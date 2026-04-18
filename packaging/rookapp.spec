# PyInstaller spec for RookApp.
#
# Run from the repo root::
#
#     pyinstaller packaging/rookapp.spec --clean --noconfirm
#
# Produces ``dist/RookApp/RookApp[.exe]`` (onedir) — faster startup
# than onefile and easier to debug when something is missing. The CI
# workflow zips the dir per-OS; releases attach the zip.
#
# Why onedir over onefile:
#   * PyInstaller onefile extracts to a temp dir at launch, which on
#     Windows AV-scans the whole bundle every time and on macOS breaks
#     codesigning paths. onedir boots in ~100 ms.
#   * Missing hidden imports fail as "module X not found" with an
#     obvious file tree to inspect, instead of a cryptic runtime error
#     from the onefile bootloader.
#   * Assets are addressable relative to the exe without the
#     ``sys._MEIPASS`` dance.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ``spec`` files execute from the directory containing the .spec, so
# anchor on the repo root instead.
ROOT = Path(SPECPATH).parent
ENTRY = ROOT / "rookapp_entry.py"
ASSETS = ROOT / "edgevox" / "apps" / "chess_robot_qt" / "assets"

# --- Assets: piece SVGs + Lottie faces + attribution files.
datas = collect_data_files(
    "edgevox.apps.chess_robot_qt",
    includes=["assets/**/*.svg", "assets/**/*.json", "assets/**/*.md"],
)

# --- Hidden imports the static analyser misses.
#
# llama-cpp-python's Python layer imports its C extension dynamically
# via ``ctypes``, so PyInstaller's modulegraph can't see the sub-
# modules. We collect the whole package and let PyInstaller's
# llama-cpp hook (shipped with recent PyInstaller) handle the shared
# library.
hiddenimports = []
hiddenimports += collect_submodules("edgevox.apps.chess_robot_qt")
hiddenimports += collect_submodules("edgevox.agents")
hiddenimports += collect_submodules("edgevox.llm")
hiddenimports += collect_submodules("edgevox.integrations.chess")
hiddenimports += collect_submodules("edgevox.examples.agents.chess_robot")
hiddenimports += collect_submodules("edgevox.core")
hiddenimports += collect_submodules("edgevox.audio")
hiddenimports += [
    "qtawesome",
    "chess",
    "chess.engine",
    "chess.pgn",
    "llama_cpp",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
]

# --- Excludes: heavy deps RookApp never touches. These come in via
# ``[project.dependencies]`` for the wider EdgeVox framework (STT/TTS
# backends, wake-word detection, FastAPI web UI) but the chess robot
# Qt app doesn't load any of them at startup, so skipping them keeps
# the bundle under 400 MB instead of 1.2 GB.
excludes = [
    "torch",
    "tensorflow",
    "transformers",
    "ctranslate2",
    "faster_whisper",
    "sherpa_onnx",
    "kokoro_onnx",
    "piper",
    "piper_phonemize",
    "supertonic",
    "pythaitts",
    "pymicro_wakeword",
    "fastapi",
    "uvicorn",
    "websockets",
    "textual",
    "IPython",
    "jupyter",
    "pytest",
    "matplotlib",
    "PIL.ImageQt",  # Qt5/6 ambiguity causes a warning on some Pillow builds
    "tkinter",
]


block_cipher = None

a = Analysis(
    [str(ENTRY)],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RookApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX corrupts PySide6 plugins on Linux/Windows.
    console=False,  # GUI app — no console window on Windows.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RookApp",
)
