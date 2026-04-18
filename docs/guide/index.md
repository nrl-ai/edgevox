# Introduction

EdgeVox is an **offline voice agent framework for robots** — agents, skills, workflows, and a sub-second voice pipeline, all running locally on CPU/CUDA/Metal with no cloud dependencies.

![EdgeVox TUI Screenshot](/screenshot.png)

![MuJoCo Panda Demo](/robot_mujoco.png)

## What is EdgeVox?

EdgeVox combines two things:

1. **An agent framework** — `@tool` and `@skill` decorators, `LLMAgent` with handoffs, behavior-tree workflows (`Sequence`, `Fallback`, `Loop`, `Router`), cancellable skills with `GoalHandle`, and a `SafetyMonitor` that preempts before the LLM is consulted.
2. **A streaming voice pipeline** — Mic → VAD → STT → LLM → TTS → Speaker, delivering first audio in ~0.8s. The pipeline is the substrate that agents run on top of.

Agent code is sim-agnostic: the same Python works on `ToyWorld` (stdlib), `IrSimEnvironment` (2D navigation), and `MujocoArmEnvironment` (3D pick-and-place).

## Key Design Principles

- **Voice is the interface** — sub-second streaming pipeline on an RTX 3080, runs on a Jetson Orin Nano, CPU fallback on a laptop
- **Agents are the program model** — write `@tool` and `@skill` functions; compose with workflows; delegate across agents with handoffs
- **Robots are the target** — cancellable skills, safety monitor, three simulation tiers, ROS2 bridge
- **Everything is offline** — no cloud APIs, no telemetry, no vendor lock

## Simulation Tiers

| Tier | Sim | Dependencies | Status |
|------|-----|-------------|--------|
| 0 | `ToyWorld` | stdlib only | shipped |
| 1 | `IrSimEnvironment` | `pip install ir-sim` | shipped |
| 2 | `MujocoArmEnvironment` | `pip install mujoco` | shipped |
| 3 | Gazebo Harmonic | ROS2 + Ubuntu | planned |

## Voice Pipeline Components

| Component | Default Model | Purpose |
|-----------|--------------|---------|
| **VAD** | Silero VAD v6 | Voice activity detection (32ms chunks) |
| **STT** | Faster-Whisper | Speech-to-text (auto-sizes by VRAM) |
| **LLM** | Gemma 4 E2B IT Q4_K_M | Chat via llama-cpp-python |
| **TTS** | Kokoro-82M | Text-to-speech (15 languages, 56 voices) |

## Next Steps

- [Quick Start](/guide/quickstart) — install and run in 5 minutes
- [Agents & Tools](/guide/agents) — full agent framework reference
- [Architecture](/guide/architecture) — deep dive into the streaming pipeline
- [Pipeline](/guide/pipeline) — agent path vs legacy streaming path
- [Chess Partner](/guide/chess) — voice-controlled chess with pluggable engines
- [Desktop App](/guide/desktop) — ship EdgeVox as a Tauri native app
- [Languages](/guide/languages) — all supported languages and backends
