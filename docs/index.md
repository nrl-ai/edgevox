---
layout: home

hero:
  name: EdgeVox
  text: Voice agents for robots.
  tagline: Sub-second voice pipeline. Plug-and-play harness. Fully on-device.
  actions:
    - theme: brand
      text: Get Started
      link: /documentation/
    - theme: alt
      text: View on GitHub
      link: https://github.com/nrl-ai/edgevox

  image:
    src: /screenshot.png
    alt: EdgeVox TUI Screenshot
---

<!-- ===== Marquee value-prop strip ===== -->
<div class="ev-marquee" aria-hidden="true">
  <div class="ev-marquee-track">
    <span>offline by default</span>
    <span>runs on a laptop</span>
    <span>no cloud required</span>
    <span>your data stays put</span>
    <span>plug-and-play hooks</span>
    <span>16 languages</span>
    <span>sub-second pipeline</span>
    <span>open since 2026</span>
    <span>offline by default</span>
    <span>runs on a laptop</span>
    <span>no cloud required</span>
    <span>your data stays put</span>
    <span>plug-and-play hooks</span>
    <span>16 languages</span>
    <span>sub-second pipeline</span>
    <span>open since 2026</span>
  </div>
</div>

<!-- ===== § 01 — Four corners ===== -->
<header class="ev-corners-header">
  <div>
    <div class="marker">§ 01 — Four corners</div>
    <h2>What EdgeVox actually is.</h2>
  </div>
  <p class="lede">A small framework with a wide surface — the harness, the voice pipeline, the robot bridge, and a reference desktop app.</p>
</header>

<div class="ev-corners">
  <div class="ev-corner">
    <div class="ev-corner-head">
      <span class="marker">§ 01</span>
      <!-- Lucide: bot -->
      <svg class="ev-icon ev-icon-lg" viewBox="0 0 24 24"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>
    </div>
    <h3>Agents &amp; tools.</h3>
    <p><code>@tool</code> and <code>@skill</code> decorators, <code>LLMAgent</code> with handoffs, nine composable workflows (Sequence, Fallback, Loop, Parallel, Router, Supervisor, Orchestrator, Retry, Timeout), cancellable skills with <code>GoalHandle</code>.</p>
    <a class="ev-corner-link" href="/documentation/agent-loop">agent loop</a>
  </div>

  <div class="ev-corner featured">
    <div class="ev-corner-head">
      <span class="marker">§ 02</span>
      <!-- Lucide: audio-waveform -->
      <svg class="ev-icon ev-icon-lg" viewBox="0 0 24 24"><path d="M2 13a2 2 0 0 0 2-2V7a2 2 0 0 1 4 0v13a2 2 0 0 0 4 0V4a2 2 0 0 1 4 0v13a2 2 0 0 0 4 0v-4a2 2 0 0 1 2-2"/></svg>
    </div>
    <h3>Voice pipeline.</h3>
    <p>Streaming STT → LLM → TTS, sub-second target on consumer GPUs. 16 languages, 56 voices, four TTS backends (Kokoro · Piper · Supertonic · PyThaiTTS). VAD barge-in halts mid-phrase.</p>
    <a class="ev-corner-link" href="/documentation/pipeline">voice pipeline</a>
  </div>

  <div class="ev-corner">
    <div class="ev-corner-head">
      <span class="marker">§ 03</span>
      <!-- Lucide: cpu -->
      <svg class="ev-icon ev-icon-lg" viewBox="0 0 24 24"><rect width="16" height="16" x="4" y="4" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>
    </div>
    <h3>Robotics &amp; sim.</h3>
    <p>ROS2-native — voice + <code>robot_state</code> + <code>agent_event</code>, TF2, Nav2 <code>cmd_vel</code>, <code>execute_skill</code> action server. ToyWorld · IR-SIM · MuJoCo Franka · Unitree G1/H1 · external Gazebo/Isaac.</p>
    <a class="ev-corner-link" href="/documentation/robotics">robotics &amp; sim</a>
  </div>

  <div class="ev-corner">
    <div class="ev-corner-head">
      <span class="marker">§ 04</span>
      <!-- Lucide: monitor -->
      <svg class="ev-icon ev-icon-lg" viewBox="0 0 24 24"><rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/><line x1="12" x2="12" y1="17" y2="21"/></svg>
    </div>
    <h3>Ships as an app.</h3>
    <p>RookApp is the reference PySide6 build — Qt UI + LLMAgent + llama.cpp + Stockfish in one Python process. No browser, no web server, no Tauri. The framework runs end-user products, not just demos.</p>
    <a class="ev-corner-link" href="/documentation/desktop">rookapp guide</a>
  </div>
</div>

<!-- ===== § 02 — Demos ===== -->
<header class="ev-section">
  <div>
    <div class="marker">§ 02 — Demos</div>
    <h2>Things you can run today.</h2>
  </div>
  <p class="lede">One <code>pip install</code>, one process, one warm laptop. Each shot below is a real screen from this repo.</p>
</header>

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

<!-- ===== § 03 — Try it ===== -->
<header class="ev-section">
  <div>
    <div class="marker">§ 03 — Try it</div>
    <h2>Six entrypoints, one install.</h2>
  </div>
  <p class="lede">Every <code>edgevox-agent</code> invocation composes with <code>--ros2</code> for the full topic surface.</p>
</header>

```bash
edgevox                                       # voice pipeline TUI
edgevox-agent robot-panda --text-mode         # MuJoCo Franka pick-and-place
edgevox-agent robot-irsim --text-mode         # IR-SIM 2D navigation
edgevox-agent robot-humanoid --simple-ui      # Unitree G1 humanoid (auto-fetched)
edgevox-agent robot-external --text-mode      # drive any external ROS2 sim / robot
edgevox-chess-robot                           # RookApp — PySide6 desktop chess partner
```

Any `edgevox-agent` invocation composes with `--ros2` to publish `/edgevox/robot_state` + `/agent_event`, accept `cmd_vel` / `goal_pose` + `text_input`, and expose the `execute_skill` action.

<!-- ===== § 04 — Principles ===== -->
<header class="ev-section">
  <div>
    <div class="marker">§ 04 — Principles</div>
    <h2>What we will and won't do.</h2>
  </div>
  <p class="lede">A small set of rules the codebase actually enforces — not aspirations.</p>
</header>

<div class="ev-principles">
  <div class="ev-principle">
    <div class="num">01</div>
    <div class="title">Plug-and-play, not patchable.</div>
    <div class="body">Every layer — STT, TTS, LLM, VAD, hooks, skills, tools, parsers — swaps via Protocols and registries. New behaviour lands as a plugin, never a conditional in core.</div>
  </div>
  <div class="ev-principle">
    <div class="num">02</div>
    <div class="title">Offline by default.</div>
    <div class="body">No cloud APIs, no telemetry, no analytics. Whisper, Gemma, Kokoro, Piper, Supertonic, PyThaiTTS — every model runs on your hardware. Period.</div>
  </div>
  <div class="ev-principle">
    <div class="num">03</div>
    <div class="title">Streaming is the contract.</div>
    <div class="body">STT &lt; 0.5 s, LLM first token &lt; 0.4 s, TTS first chunk &lt; 0.1 s. No blocking calls hold the loop. Latency regressions block the merge.</div>
  </div>
  <div class="ev-principle">
    <div class="num">04</div>
    <div class="title">Hardware-aware, never hardware-bound.</div>
    <div class="body">CUDA, Metal, CPU — every backend degrades gracefully. A missing accelerator is a config decision, not a crash.</div>
  </div>
  <div class="ev-principle">
    <div class="num">05</div>
    <div class="title">Safety preempts the LLM.</div>
    <div class="body">SafetyMonitor halts skills <em>before</em> the LLM is consulted — stop-words land in the reactive layer, the halt path doesn't wait on a model round-trip.</div>
  </div>
  <div class="ev-principle">
    <div class="num">06</div>
    <div class="title">MIT, no copyleft contamination.</div>
    <div class="body">License of every dependency is verified at add-time. GPL/AGPL/SSPL are refused. Your downstream stays unencumbered.</div>
  </div>
</div>
