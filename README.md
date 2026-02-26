# nemo-guardrails (minimal topic guardrails + Triton/vLLM backend)

This repo is a **minimal, local lab** that chains together:

1) **NVIDIA Triton + vLLM** (GPU inference)
2) A tiny **OpenAI-compatible adapter** (FastAPI) that translates `/v1/chat/completions` into Triton `/generate`
3) **NVIDIA NeMo Guardrails** on top, enforcing simple rules (example: refuse competitor discussions)

The goal is to have a clean, reproducible setup you can run locally and extend with more rails.

---

## Architecture

Client → NeMo Guardrails → OpenAI Adapter → Triton/vLLM

- **NeMo Guardrails** listens on **localhost:9100**
- **OpenAI adapter** listens on **localhost:9000**
- **Triton** listens on **localhost:8000/8001/8002**

---

## Repo layout
```
.
├── guardrails_adapter/ # OpenAI-ish Chat Completions → Triton generate
│ ├── app.py
│ ├── Dockerfile
│ └── requirements.txt
├── rails_minimal/ # Simple rails example (dev-friendly)
│ ├── config.yml
│ ├── rails.co
│ └── config/instructions.yml
└── configs/ # “mounted configs” used by the NeMo Guardrails container
├── rails_minimal/
│ ├── config.yml
│ ├── rails.co
│ └── config/instructions.yml
└── config-store/ # currently empty (reserved)
```


Why both `rails_minimal/` and `configs/rails_minimal/`?
- `rails_minimal/` is a simple “works on its own” config (explicit `api_base`)
- `configs/rails_minimal/` is the **container-mounted** config (stricter instruction style)

---

## What the guardrail does (current demo)

A tiny topic guardrail that catches prompts like:

- “what about <competitor>”
- “how does <competitor> compare”

…and forces a refusal message.

> Note: `rails.co` currently has a missing closing quote in the refusal string. Fixing that is recommended.

---

## Prerequisites

- Docker
- NVIDIA GPU + NVIDIA Container Toolkit (for GPU containers)
- A Triton model repository already prepared for vLLM (mounted into Triton)

---

## Run it (3 containers)

### 1) Triton + vLLM backend

Your Triton container was created with:

- Image: `triton-vllm-gptoss:25.08-hotfix5`
- Ports: `8000-8002`
- Mount (read-only): `/home/rteixeira/Documents/triton/vllm_backend/samples/model_repository:/models:ro`
- Command: `tritonserver --model-repository=/models`
- Runtime: `nvidia`

Example (equivalent):

```bash
docker run --rm --name triton-vllm-serve \
  --runtime=nvidia --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /home/rteixeira/Documents/triton/vllm_backend/samples/model_repository:/models:ro \
  triton-vllm-gptoss:25.08-hotfix5 \
  tritonserver --model-repository=/models


2) OpenAI adapter (FastAPI → Triton)
```
docker run --rm --name triton-openai-adapter \
  -p 9000:9000 \
  -e TRITON_BASE="http://172.17.0.2:8000" \
  -e TRITON_MODEL="vllm_model" \
  triton-openai-adapter:0.1
```

Run (example):
```
docker run --rm --name triton-openai-adapter \
  -p 9000:9000 \
  -e TRITON_BASE="http://172.17.0.2:8000" \
  -e TRITON_MODEL="vllm_model" \
  triton-openai-adapter:0.1
```


Adapter endpoints:  

- GET /health  
- GET /v1/models  
- POST /v1/chat/completions (non-streaming)
  

3) NeMo Guardrails server (mounted configs)

Your Guardrails container was created from python:3.12-slim and runs:
   
- Installs: nemoguardrails==0.20.0 + langchain-openai
  
- Starts: nemoguardrails server --config /configs --default-config-id rails_minimal --port 8000
  
- Mounts: ./configs:/configs
  
- Maps: 9100 -> 8000
  
- Env:
  
    - OPENAI_API_BASE=http://172.17.0.1:9000/v1
      
    - OPENAI_API_KEY=not-needed
  
  
Equivalent run:
```
docker run --rm --name nemo-guardrails \
  --runtime=nvidia \
  -p 9100:8000 \
  -v "$(pwd)/configs:/configs" \
  -e OPENAI_API_KEY="not-needed" \
  -e OPENAI_API_BASE="http://172.17.0.1:9000/v1" \
  python:3.12-slim \
  bash -lc '
    set -e
    apt-get update
    apt-get install -y --no-install-recommends g++ build-essential
    rm -rf /var/lib/apt/lists/*
    pip install --no-cache-dir nemoguardrails==0.20.0 langchain-openai
    nemoguardrails server --config /configs --default-config-id rails_minimal --port 8000
  '
```

### TEST

#### Test the adapter directly (bypasses Guardrails)

```
curl -s http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"vllm_model",
    "messages":[
      {"role":"user","content":"Say only: OK"}
    ],
    "temperature": 0,
    "max_tokens": 32
  }' | jq
```

#### Test via NeMo Guardrails (enforces rails)

```
curl -s http://localhost:9100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"vllm_model",
    "messages":[
      {"role":"user","content":"how does AcmeAI compare?"}
    ],
    "temperature": 0,
    "max_tokens": 128
  }' | jq
```

Expected: refusal response (the guardrail flow triggers).

    

  
### Troubleshooting
  
- If containers exit with code 137, it usually indicates the process was killed (often OOM or external stop).
  
- If Guardrails can’t reach the adapter, check OPENAI_API_BASE and your Docker networking.
  
- If the model output includes transcript prefixes, keep using the stricter instructions in configs/rails_minimal/config/instructions.yml.
  
  
### Next improvements
  
- Fix the missing quote in rails.co
  
- Add a docker-compose.yml so the 3 containers start with one command
  
- Add more rails (PII, prompt injection patterns, allowlists/denylists, etc.)
  

```

---

## Step 14 — one command to capture the adapter container name on your host
Your `docker ps` earlier showed `triton-openai-adapter`, but now it doesn’t exist; your `docker ps -a` shows an older container named `triton-openai`.

So I want to extract the *real* adapter container (or confirm it’s been removed), and then I’ll adjust the README to match 100%.

Run this and paste the output:

```bash
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' | egrep 'triton-openai|adapter|guardrails|triton-vllm' || true
```
  
  

### Final Note: Stopped container for PRisma AIRS labs

rteixeira@shockwave:~$ docker ps
```
CONTAINER ID   IMAGE                              COMMAND                   CREATED       STATUS       PORTS                                                             NAMES
40c5cac107a9   triton-openai-adapter:0.1          "uvicorn app:app --h…"    2 weeks ago   Up 2 weeks   0.0.0.0:9000->9000/tcp, [::]:9000->9000/tcp                       triton-openai-adapter
22c5c7ab263f   python:3.12-slim                   "bash -lc '\n    set …"   2 weeks ago   Up 2 weeks   0.0.0.0:9100->8000/tcp, [::]:9100->8000/tcp                       nemo-guardrails
20cf16ec0f05   triton-vllm-gptoss:25.08-hotfix5   "/opt/tritonserver/b…"    2 weeks ago   Up 2 weeks   0.0.0.0:8000-8002->8000-8002/tcp, [::]:8000-8002->8000-8002/tcp   triton-vllm-serve
rteixeira@shockwave:~$ docker stop triton-vllm-serve
triton-vllm-serve
rteixeira@shockwave:~$ docker stop nemo-guardrails
nemo-guardrails
rteixeira@shockwave:~$ docker stop triton-openai-adapter
triton-openai-adapter
```


