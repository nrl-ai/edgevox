"""``edgevox-serve`` entrypoint: FastAPI app + uvicorn runner.

Loads the STT/LLM/TTS singletons once at startup and exposes:
  GET  /            → static SPA (built by ``webui/``)
  GET  /api/health  → liveness + active session count
  GET  /api/info    → languages, voice, sample rates
  WS   /ws          → per-session voice pipeline (see ws.py)

Usage:
    edgevox-serve --host 127.0.0.1 --port 8765 --language en
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from edgevox.server.core import ServerCore
from edgevox.server.ws import handle_connection

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(core: ServerCore) -> FastAPI:
    app = FastAPI(title="EdgeVox Server", version="0.1.0")

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "active_sessions": len(core.sessions)}

    @app.get("/api/info")
    async def info():
        return JSONResponse(core.info())

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await handle_connection(ws, core)

    if STATIC_DIR.exists() and any(STATIC_DIR.iterdir()):
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="spa")
    else:
        log.warning(
            "Static SPA not found at %s — build it with `cd webui && npm run build`. The /ws endpoint still works.",
            STATIC_DIR,
        )

        @app.get("/")
        async def _no_spa():
            return JSONResponse(
                {
                    "error": "SPA not built",
                    "hint": "cd webui && npm install && npm run build",
                },
                status_code=503,
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="EdgeVox WebSocket server with web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument("--language", default="en", help="Speech language (default: en)")
    parser.add_argument("--stt", default=None, help="STT model (e.g. tiny, small, medium, large-v3-turbo)")
    parser.add_argument("--stt-device", default=None, help="STT device (cuda, cpu)")
    parser.add_argument("--llm", default=None, help="LLM GGUF path or hf:repo:file")
    parser.add_argument("--tts", default=None, choices=["kokoro", "piper"], help="TTS backend")
    parser.add_argument("--voice", default=None, help="TTS voice name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    core = ServerCore(
        language=args.language,
        stt_model=args.stt,
        stt_device=args.stt_device,
        llm_model=args.llm,
        tts_backend=args.tts,
        voice=args.voice,
    )
    app = create_app(core)

    import uvicorn

    log.info("EdgeVox server listening on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
