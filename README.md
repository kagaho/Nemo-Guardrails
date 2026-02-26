Parei os container para o lab de Prisma AIRS

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


