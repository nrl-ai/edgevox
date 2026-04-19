# Introduction

EdgeVox is an **offline voice agent framework for robots** — agents, skills, workflows, and a sub-second voice pipeline, all running locally on CPU / CUDA / Metal with no cloud dependencies.

![EdgeVox TUI Screenshot](/screenshot.png)

## What is EdgeVox?

EdgeVox combines two things:

1. **An agent framework** — `@tool` and `@skill` decorators, `LLMAgent` with handoffs, behavior-tree workflows (`Sequence`, `Fallback`, `Loop`, `Parallel`, `Router`, `Supervisor`, `Orchestrator`, `Retry`, `Timeout`), cancellable skills with `GoalHandle`, and a `SafetyMonitor` that preempts before the LLM is consulted.
2. **A streaming voice pipeline** — Mic → VAD → STT → LLM → TTS → Speaker, delivering first audio in ~0.8 s. The pipeline is the substrate that agents run on top of.

Agent code is sim-agnostic: the same Python works on `ToyWorld` (stdlib), `IrSimEnvironment` (2D navigation), `MujocoArmEnvironment` (3D pick-and-place), `MujocoHumanoidEnvironment` (Unitree G1 / H1), and `ExternalROS2Environment` (any Gazebo / Isaac / real robot over ROS2).

## Key design principles

- **Voice is the interface** — sub-second streaming pipeline on an RTX 3080, runs on a Jetson Orin Nano, CPU fallback on a laptop
- **Agents are the program model** — write `@tool` and `@skill` functions; compose with workflows; delegate across agents with handoffs
- **Robots are the target** — cancellable skills, safety monitor, three simulation tiers, ROS2 bridge
- **Everything is offline** — no cloud APIs, no telemetry, no vendor lock-in

## Simulation tiers

| Tier | Sim | Dependencies | Role | Status |
|------|-----|-------------|------|--------|
| 0 | `ToyWorld` | stdlib only | unit tests, trivial examples | shipped |
| 1 | `IrSimEnvironment` | `pip install ir-sim` | 2D visual demo (matplotlib, diff-drive, LiDAR) | shipped |
| 2a | `MujocoArmEnvironment` | `pip install mujoco` | 3D physics, Franka pick-and-place | shipped |
| 2b | `MujocoHumanoidEnvironment` | `pip install mujoco` | Unitree G1 / H1 from Menagerie, procedural gait + ONNX policy slot | shipped |
| 3 | `ExternalROS2Environment` | sourced ROS2 workspace | drive Gazebo / Isaac / real robots over standard topics | shipped |

![MuJoCo Franka pick-and-place](/robot_panda.png)

![Unitree G1 humanoid](/robot_unitree_g1.png)

## Voice pipeline components

| Component | Default model | Purpose |
|-----------|--------------|---------|
| **VAD** | Silero VAD v6 | Voice activity detection (32 ms chunks) |
| **STT** | Faster-Whisper | Speech-to-text (auto-sized by VRAM) |
| **LLM** | Gemma 4 E2B IT Q4\_K\_M | Chat via llama-cpp-python |
| **TTS** | Kokoro-82M | Text-to-speech (16 languages, 56 voices) |

## Shipping a desktop app

EdgeVox is not just a library — [**RookApp**](/documentation/desktop) is a reference PySide6 desktop application built on the same `LLMAgent` you use for robots. One Python process hosts the Qt UI, llama-cpp, and a Stockfish subprocess. No browser, no web server, no Node toolchain, no Tauri.

![RookApp — PySide6 desktop chess robot](/rook_app.png)

## Next steps

**Getting started**

- [Quick start](/documentation/quickstart) — install and run in 5 minutes
- [Architecture](/documentation/architecture) — deep dive into the streaming pipeline
- [Component Design](/documentation/components) — per-module design with mermaid diagrams

**Features**

- [Languages](/documentation/languages) — all 16 supported languages and backends
- [Voice pipeline](/documentation/pipeline) — agent path vs legacy streaming path
- [Agents & tools](/documentation/agents) — full agent framework reference
- [TUI commands](/documentation/commands) — slash commands, voice switching, debugging
- [ROS2 integration](/documentation/ros2) — topics, services, action servers
- [Chess partner](/documentation/chess) — voice-controlled chess with pluggable engines
- [RookApp (desktop)](/documentation/desktop) — shipping EdgeVox as a native PySide6 app

**Harness architecture**

- [Agent loop](/documentation/agent-loop) — six fire-points, parallel dispatch, handoff short-circuit
- [Hooks](/documentation/hooks) — authoring contract, built-ins, ordering rules
- [Memory](/documentation/memory) — `MemoryStore`, `SessionStore`, `NotesFile`, `Compactor`
- [Multi-agent](/documentation/multiagent) — `Blackboard`, `BackgroundAgent`, `AgentPool`
- [Interrupt & barge-in](/documentation/interrupt) — cancel-token plumbing
- [Tool calling](/documentation/tool-calling) — parser chain + GBNF grammar roadmap
