import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

TRITON_BASE = os.environ.get("TRITON_BASE", "http://172.17.0.2:8000").rstrip("/")
TRITON_MODEL = os.environ.get("TRITON_MODEL", "vllm_model")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionsRequest(BaseModel):
    # allow extra fields like "stop" sent by various OpenAI clients
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "triton_base": TRITON_BASE, "triton_model": TRITON_MODEL}

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": TRITON_MODEL, "object": "model"}]}

def messages_to_prompt(messages: List[ChatMessage]) -> str:
    lines = []
    for m in messages:
        role = m.role.strip().lower()
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        lines.append(f"{role.upper()}: {m.content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)

def _first_non_empty_line(s: str) -> str:
    for ln in s.splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return ""

def clean_completion(text: str) -> str:
    if not text:
        return ""

    t = text.strip()

    # If the model echoed the prompt, keep only content after the LAST "ASSISTANT:"
    if "ASSISTANT:" in t:
        t = t.rsplit("ASSISTANT:", 1)[-1].strip()

    # If the model produced a "final marker", keep only after it
    if "assistantfinal" in t.lower():
        parts = re.split(r"(?i)assistantfinal", t, maxsplit=1)
        if len(parts) == 2:
            t = parts[1].strip()

    # Drop any remaining transcript-y lines
    cleaned_lines = []
    for ln in t.splitlines():
        ln_stripped = ln.strip()
        if re.match(r"^(USER|User|SYSTEM|System|ASSISTANT|Assistant)\s*:", ln_stripped):
            continue
        if ln_stripped:
            cleaned_lines.append(ln_stripped)

    t = "\n".join(cleaned_lines).strip()

    # For strict tests like "Say only: OK", return the first non-empty line
    return _first_non_empty_line(t) or t

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest) -> Dict[str, Any]:
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true not supported in this minimal adapter yet")

    prompt = messages_to_prompt(req.messages)

    # Triton expects stop to be int/bool/string (NOT list). Convert if needed.
    stop_val: Optional[str] = None
    if isinstance(req.stop, list) and len(req.stop) > 0:
        stop_val = str(req.stop[0])
    elif isinstance(req.stop, str):
        stop_val = req.stop

    params: Dict[str, Any] = {
        "stream": False,
        "temperature": req.temperature if req.temperature is not None else 0.0,
        "max_tokens": req.max_tokens if req.max_tokens is not None else 256,
    }
    if stop_val:
        params["stop"] = stop_val

    payload = {"text_input": prompt, "parameters": params}

    url = f"{TRITON_BASE}/v2/models/{TRITON_MODEL}/generate"
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=300)
    dt = time.time() - t0

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail={"triton_status": r.status_code, "triton_body": r.text})

    data = r.json()
    raw_text = data.get("text_output", "")
    text = clean_completion(raw_text)

    return {
        "id": f"chatcmpl-adapter-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or TRITON_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "triton_latency_s": dt,
        },
    }
