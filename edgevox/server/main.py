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
import importlib
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from edgevox.server.core import ServerCore
from edgevox.server.ws import handle_connection

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _load_agent(spec: str, core: ServerCore) -> tuple[Any, Any]:
    """Resolve ``module:factory`` → ``(agent, deps)``.

    The factory may return either an ``Agent`` (``deps=None`` implied)
    or a ``(agent, deps)`` tuple. Anything else is rejected with a
    friendly error so a misconfigured spec fails early at boot rather
    than on the first WebSocket connection.
    """
    if ":" not in spec:
        raise ValueError(f"--agent expected 'module:factory', got {spec!r}")
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr, None)
    if factory is None:
        raise ValueError(f"--agent: {module_name}.{attr} not found")
    result = factory(core)
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, None


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
    parser.add_argument(
        "--agent",
        default=None,
        help=(
            "Dotted path to an agent factory 'module:factory'. The factory receives "
            "the ServerCore and returns either an Agent or (agent, deps). When set, "
            "the WebSocket pipeline routes every turn through LLMAgent.run with hooks, "
            "tools, and ctx.deps. Example: "
            "'edgevox.examples.agents.chess_partner:build_server_agent'."
        ),
    )
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
    if args.agent:
        try:
            agent, deps = _load_agent(args.agent, core)
        except Exception as e:
            raise SystemExit(f"failed to load --agent {args.agent!r}: {e}") from e
        # Ensure the agent is bound to the shared LLM; factories can
        # skip the bind when they want to keep a per-session LLM.
        if hasattr(agent, "bind_llm") and getattr(agent, "llm", None) is None:
            agent.bind_llm(core.llm)
        core.bind_agent(agent, deps)
        log.info("agent bound: %s", args.agent)
    app = create_app(core)

    import uvicorn

    log.info("EdgeVox server listening on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
