"""WebSocket server mode for EdgeVox.

Exposes the streaming voice pipeline (VAD → STT → LLM → TTS) over a WebSocket
endpoint plus a static SPA, so a browser can hold continuous conversations with
the local AI. Coexists with the TUI; shares no runtime state.
"""
