"""Minimal miner ASGI server.

Gate requirement (matriks §5 / FASE0_VERIFY.md row h4): the miner endpoint binds
port 7999 — matching rayonlabs/G.O.D `miner/asgi.py:53` (host 127.0.0.1, port 7999).
NOTE: poseidon's *trainer* asgi binds 8001; that is a different service. We follow
the verified miner contract: 7999.

This is scaffold — a health endpoint so the port/contract is in place. The actual
training container's ENTRYPOINT is run_image_trainer.sh (a batch job), not this server.
"""

import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7999)
