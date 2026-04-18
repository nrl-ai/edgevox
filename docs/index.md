---
layout: home

hero:
  name: EdgeVox
  text: Voice agents for robots.
  tagline: Offline agents, skills, and workflows — sub-second voice pipeline, fully on-device.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/
    - theme: alt
      text: View on GitHub
      link: https://github.com/vietanhdev/edgevox

  image:
    src: /screenshot.png
    alt: EdgeVox TUI Screenshot

features:
  - title: Agent framework
    icon: 🧠
    details: "@tool and @skill decorators, LLMAgent with handoffs, nine composable workflows (Sequence, Fallback, Loop, Parallel, Router, Supervisor, Orchestrator, Retry, Timeout), cancellable skills with GoalHandle."
  - title: Full sim stack
    icon: 🤖
    details: "ToyWorld · IR-SIM (2D) · MuJoCo Franka (arm) · Unitree G1 / H1 (humanoid) · External ROS2 (Gazebo, Isaac, real robots). One SimEnvironment protocol across all tiers."
  - title: Sub-second voice
    icon: ⚡
    details: Streaming STT → LLM → TTS pipeline delivers first audio in ~0.8 s. 16 languages, 56 voices, 4 TTS backends (Kokoro · Piper · Supertonic · PyThaiTTS).
  - title: 100 % offline
    icon: 🔒
    details: No cloud APIs, no telemetry. Whisper STT + Gemma 4 LLM + Kokoro / Piper / Supertonic / PyThaiTTS all run on your hardware (CPU / CUDA / Metal).
  - title: Safety-first robotics
    icon: 🛡️
    details: SafetyMonitor preempts before the LLM is consulted. Stop-words halt in-flight skills in ~200 ms. The LLM never enters the reactive layer.
  - title: ROS2-native
    icon: 📡
    details: "Full ROS2 surface — voice + robot_state + agent_event topics, TF2 / Nav2 cmd_vel / LaserScan / Image, execute_skill ActionServer, std_srvs query services."
  - title: Ships as a desktop app
    icon: 💻
    details: "RookApp is a reference PySide6 build — Qt UI, LLMAgent, llama-cpp, and Stockfish all in one Python process. No browser, no web server, no Node, no Tauri."
    link: /guide/desktop
    linkText: RookApp guide →
  - title: Pluggable harness
    icon: 🧩
    details: "Swap any STT / TTS / LLM / VAD / hook / skill / tool / parser via Protocols + registries. Six hook fire-points, priority ordering, typed AgentContext."
    link: /guide/agent-loop
    linkText: Agent loop →
---

## Demos

<div class="ev-demos">
  <figure>
    <img src="/screenshot.png" alt="EdgeVox TUI — voice pipeline" />
    <figcaption><strong>Voice pipeline TUI</strong> — streaming STT · LLM · TTS with VAD barge-in</figcaption>
  </figure>
  <figure>
    <img src="/robot_panda.png" alt="MuJoCo Franka Panda pick-and-place" />
    <figcaption><strong>MuJoCo · Franka arm</strong> — voice-controlled pick-and-place</figcaption>
  </figure>
  <figure>
    <img src="/robot_unitree_g1.png" alt="Unitree G1 humanoid" />
    <figcaption><strong>Unitree G1 humanoid</strong> — procedural gait + ONNX policy slot</figcaption>
  </figure>
  <figure>
    <img src="/rook_app.png" alt="RookApp — PySide6 desktop chess robot" />
    <figcaption><strong>RookApp desktop</strong> — offline chess partner in one Python process</figcaption>
  </figure>
</div>

## Try it

```bash
edgevox                                       # voice pipeline TUI
edgevox-agent robot-panda --text-mode         # MuJoCo Franka pick-and-place
edgevox-agent robot-irsim --text-mode         # IR-SIM 2D navigation
edgevox-agent robot-humanoid --simple-ui      # Unitree G1 humanoid (auto-fetched)
edgevox-agent robot-external --text-mode      # drive any external ROS2 sim / robot
edgevox-chess-robot                           # RookApp — PySide6 desktop chess partner
```

Any `edgevox-agent` invocation composes with `--ros2` to publish `/edgevox/robot_state` + `/agent_event`, accept `cmd_vel` / `goal_pose` + `text_input`, and expose the `execute_skill` action.
