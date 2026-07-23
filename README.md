<div align="center">

# SARA AI

### An offline-first, bilingual voice assistant with real-time echo cancellation, semantic long-term memory, and hybrid intent resolution

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square&logo=windows&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-success?style=flat-square)](#)

[Overview](#overview) · [Architecture](#architecture) · [Engineering Highlights](#engineering-highlights) · [Getting Started](#getting-started) · [Configuration](#configuration) · [Roadmap](#roadmap)

</div>

---

## Overview

SARA AI is a desktop voice assistant built to run **fully offline** on consumer hardware — no cloud STT/TTS, no mandatory API keys, no per-request billing. It targets a single constraint that most voice-assistant tutorials ignore: **one shared GPU, three latency-sensitive workloads, running concurrently, for hours at a time, without leaking memory or deadlocking.**

Everything in this repository was built and hardened against that constraint on a single RTX 3050 (4GB VRAM), which shapes almost every architectural decision described below.

```text
"Hey Sara, what's the weather in Ajmer, and remind me to call mom at 6pm"
        │
        ▼
┌───────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌────────────┐
│  Wake Word     │───▶│  faster-     │───▶│  Hybrid Intent    │───▶│  Kokoro    │
│  Detection     │    │  whisper STT │    │  Resolution       │    │  ONNX TTS  │
│  (STT-fallback)│    │  (GPU, FP16) │    │  (regex → LLM     │    │  (GPU)     │
└───────────────┘    └──────────────┘    │   tool-calling)   │    └────────────┘
        ▲                     ▲          └────────┬──────────┘          │
        │                     │                   │                     │
        │              ┌──────┴──────┐    ┌────────▼────────┐           │
        └──────────────│  WebRTC AEC  │◀───│  Ollama / Gemini │           │
                        │  (shared     │    │  + RAG memory    │           │
                        │  processor)  │    │  (SQLite +       │           │
                        └──────────────┘    │  cosine search)  │           │
                               ▲             └──────────────────┘          │
                               └─────────────────────────────────────────┘
                                     far-end reference feed (24kHz→16kHz)
```

---

## Engineering Highlights

*This section exists because a feature list doesn't show how a system was built — these do.*

### 1. Real-time audio pipeline under GPU contention

Three GPU-bound models (`faster-whisper`, Kokoro ONNX, Ollama) share one 4GB GPU. Rather than serializing them (which would make wake-word response feel sluggish), the system uses:

- **A single shared `AECProcessor` instance** wired into *both* the TTS output callback (far-end/reference feed) and the STT input callback (near-end cancellation) — two independent adaptive-filter instances would have unrelated internal state and silently produce zero cancellation.
- **Off-thread AEC feeding.** An earlier revision called `feed_far_end()` directly inside the real-time `sounddevice` output callback. Under GPU load, that call (lock acquisition + 24kHz→16kHz resample + a native WebRTC APM call) could occasionally miss its block deadline, causing an audible glitch — which then became a *new* echo source picked up by the mic. Fixed by moving the actual resample+feed work to a dedicated background thread; the real-time callback now does a single lock-free `queue.put_nowait()` and returns immediately.
- **Lazy background construction.** `SpeechToText`, `TextToSpeech`, and the SQLite layer are built concurrently via a `ThreadPoolExecutor`, while the LLM client and `ReminderManager` defer construction to background threads via a custom `_Lazy` wrapper — first UI paint and wake-word readiness are never blocked on an Ollama cold-start.

### 2. A single-writer, WAL-mode persistence layer

All state (preferences, conversation log, reminders, long-term memory) lives in one canonical SQLite file resolved once in `config.py` — not independently by each consuming module. This sounds trivial; it wasn't always true. An earlier revision had three separate modules each computing their own default DB path (`os.getcwd()`-relative, `__file__`-relative, and project-root-relative), which meant **the physical database file in use silently depended on the directory the process happened to be launched from.** Fixed by making every module import one shared `Config.DB_PATH`, with a written, idempotent migration script for anyone with an already-split installation.

Within that one file: a dedicated writer thread owns the only connection that ever performs writes (serialized through a `queue.Queue`, avoiding SQLite's `database is locked` under concurrent access), while reads use one lazily-opened connection per calling thread under WAL mode — readers never block on the writer.

### 3. Long-term semantic memory (RAG) without a vector-database dependency

Conversation history is capped (a fixed-size deque) for LLM context-window reasons, but that shouldn't mean everything older is forgotten. `sara/core/rag.py` implements retrieval-augmented recall using:

- **Ollama's own `/api/embeddings` endpoint** (plain HTTP, no coupling to a specific client library version) instead of pulling in a heavyweight embedding model — the system already runs Ollama for chat, so this adds zero new GPU/RAM footprint.
- **An in-memory `(N, D)` cosine-similarity matrix**, rebuilt from SQLite once at startup and appended to incrementally — appropriate for a single-user assistant's memory scale (thousands of entries), without the operational overhead of a real vector database.
- **A hard timeout on the retrieval-time embedding call.** Retrieval sits directly on the hot path before every LLM response; if the embedding backend is slow or unavailable, `search()` degrades to an empty list rather than stalling the user's turn.
- **Fire-and-forget ingestion** through a background writer thread, mirroring the same producer/consumer pattern used for conversation logging — a slow embed call never blocks the response the user is already hearing.

### 4. Hybrid intent resolution: deterministic first, LLM as a fallback, never a replacement

A ~100-pattern regex intent router (`intent.py`) handles the overwhelming majority of commands with zero LLM latency and 100% determinism. Rather than trying to make regex cover every possible phrasing (a losing battle) or routing everything through an LLM (unacceptable latency for "set volume to 50"), the system adds **one bounded-time structured tool-calling request** — but *only* when the regex layer found nothing at all:

```python
# intent.py finds nothing → exactly the point where the old code
# went straight to a generic conversational reply. tool_router gets
# ONE bounded-time shot at resolving a real action first.
resolved = resolve_tool_call(user_input, brain.model_name)
if resolved:
    fake_match = build_fake_match(resolved["name"], resolved["arguments"])
    handler = _INTENT_HANDLERS.get(TOOL_NAME_TO_INTENT[resolved["name"]])
    result = handler(fake_match, ctx)   # reuses the EXACT SAME executor
```

The resolved tool call is translated into the same `re.Match`-shaped object the regex layer would have produced, so it's dispatched through the **identical executor functions** — zero duplicated business logic between the fast path and the LLM-assisted path. If the installed Ollama client/model doesn't support tool calling at all, this fails silently and the system falls back to its original behavior — a strictly additive feature with no regression surface.

### 5. Systematic thread-safety auditing

Every long-lived background thread (`TTSWorker`, `_WakeWatcher`, `AsyncDBWriter`, the AEC far-end feeder) follows the same discipline: **the outer loop body is wrapped, not just the inner call**, so an unexpected exception logs and continues instead of silently killing the thread. This mattered in practice — an uncaught exception in the wake-word polling loop used to kill the *only* thread that ever signals a wake event, permanently and silently, with the failure surfacing only as "the assistant stopped responding to its name" with no visible error.

A lazy-initialization wrapper (`_Lazy`) that defers expensive construction (LLM client, reminder manager) to a background thread had a related bug class: if the factory raised, waiting callers blocked forever rather than seeing the error. Fixed by always releasing waiters via `finally`, with the stored exception re-raised (with context) on first access.

### 6. Calculator sandboxing without executor-based timeouts

User-facing arithmetic is evaluated via a restricted `eval()` — the naive fix for "what if someone submits `2**999999999999`" is to run it in a thread pool with a timeout. That was tried and rejected here: `ThreadPoolExecutor` workers are non-daemon, so a genuinely runaway `eval()` call left a thread that Python cannot forcibly kill, which blocked the *entire process* from exiting even after the nominal timeout had elapsed (verified by testing, not assumed). The fix instead rejects pathological input **before** `eval()` ever runs — capping operand digit count, banning power towers, and bounding exponent magnitude by absolute value (a negative exponent is exactly as expensive to compute as its positive counterpart, since Python evaluates it via the positive power first) — so `eval()` itself is always fast and no timeout machinery is needed at all.

### 7. Production observability

A centralized rotating log (`logging_config.py`) captures every module's logger — previously, `logger.error()` calls scattered across the codebase went nowhere, since no root logger was ever configured (Python's silent `lastResort` handler swallowed them). A pre-flight `health_check.py` runs before any expensive model loading: missing model files, an unreachable LLM backend, no audio device, and a non-writable DB path are now reported *before* the app limps into a half-broken state, both to the log and as a summarized GUI notification.

---

## Architecture

Each module below used to be a single large file; every one is now a small
package (folder) of focused files behind an unchanged public import path,
so `from sara.audio.tts import TextToSpeech` etc. still works exactly as
before — see each package's `__init__.py` docstring for the internal
breakdown.

```
sara/
├── audio/
│   ├── aec.py             # WebRTC Acoustic Echo Cancellation (shared far/near-end processor)
│   ├── stt/                # faster-whisper (GPU FP16) + VAD-based endpointing + wake-word fallback
│   │   ├── helpers.py        # RMS energy, language detection, hallucination filter
│   │   ├── buffers.py          # VAD, ring/pre-buffers, noise floor, collection state
│   │   └── engine.py             # SpeechToText (public class)
│   └── tts/                 # Kokoro ONNX streaming synthesis, persistent output stream
│       ├── voice_params.py    # per-language speed/pitch presets
│       ├── text_prep.py         # text normalization + adaptive chunk splitting
│       ├── cache.py               # short-phrase synthesis cache
│       ├── synth.py                 # Kokoro ONNX call + volume shaping
│       ├── player.py                  # persistent playback worker
│       └── engine.py                    # TextToSpeech (public class)
├── core/
│   ├── llm/                # Ollama/Gemini streaming client, sentence-boundary chunking, retry/backoff
│   │   ├── prompt.py          # system-prompt construction
│   │   ├── streaming.py         # sentence/clause-boundary helpers
│   │   ├── clients.py             # lazy Ollama/Gemini client construction
│   │   └── engine.py                # SaraLLM (public class)
│   ├── intent.py            # Deterministic regex intent router (gated + memoized)
│   ├── memory.py             # Single-writer, WAL-mode SQLite preferences/conversation store
│   ├── rag.py                 # Long-term semantic memory (embeddings + cosine retrieval)
│   └── tool_router.py          # optional: LLM tool-calling fallback for unmatched phrasing
├── tools/
│   ├── reminders.py         # Persistent reminders with background due-polling
│   ├── system/                # ~70 Windows system-control actions, grouped by category
│   │   ├── apps.py, power.py, window_mgmt.py, media_keys.py, shortcuts.py,
│   │   │   connectivity.py, files_notes.py, timers.py, folders.py,
│   │   │   settings_pages.py, system_info.py, dispatch.py, _shared.py
│   ├── web.py                  # Search, weather, news, YouTube/Spotify, page summarization
│   ├── vision.py                 # Screenshot capture + Gemini Vision description
│   └── clipboard.py                # Clipboard read/write
└── gui/
    ├── app/                   # pywebview bridge, composed from focused mixins
    │   ├── events.py            # window-lifecycle state + Python->JS push bridge
    │   ├── helpers.py             # export shaping, weather fallback fetch, pref writer
    │   ├── core.py                  # ApiCoreMixin: init, system stats, weather, window, wake/stop
    │   ├── reminders.py                # ApiRemindersMixin: Calendar/Reminders CRUD
    │   ├── settings.py                   # ApiSettingsMixin: mute/focus/mic/speed/wifi/language
    │   ├── notes.py                        # ApiNotesMixin: Quick Notes + memory export
    │   ├── media.py                          # ApiMediaMixin: media player controls
    │   ├── engine.py                           # Api, composed from the mixins above
    │   └── bootstrap.py                          # main(): window creation + entry point
    ├── index.html / js/app.js  # Dashboard, chat, calendar, voice-control, memory pages
    └── style/style.css           # Design system (dark/light themes, motion-reduced variants)

sara/orchestrator/       # Everything main.py used to do inline — now split by concern:
│   ├── lazy.py               # _Lazy background construction wrapper
│   ├── state.py                # LanguageState, AssistantState (GUI-driven toggles)
│   ├── ollama_manager.py         # start/stop/health-check the local Ollama server
│   ├── ui_bridge.py                # ui_update(kind, *args) wrapper + event coalescing
│   ├── tts_worker.py                 # TTSWorker (speak/barge-in coordination)
│   ├── db_writer.py                    # AsyncDBWriter (fire-and-forget conversation log)
│   ├── calc_utils.py                     # safe calculator eval + duration parsing
│   ├── network_utils.py                    # bounded-timeout wrapper for network tools
│   ├── text_utils.py                         # name-extraction / phrase-matching helpers
│   ├── history.py                              # restore conversation history + preferences
│   ├── intent_handlers.py                        # one handler per fast-path regex intent
│   └── core_wiring.py                               # build_core_objects() + run_sara_logic()

main.py                # Thin entry point: setup_logging -> build_core_objects -> launch GUI
                          (previously gui_main.py)
config.py              # Centralized, validated configuration (single source of truth for all tuning)
logging_config.py        # Rotating file + console logging setup
health_check.py            # Pre-flight startup diagnostics
```

---

## Getting Started

### Prerequisites

- Windows 10/11
- Python 3.10+
- An NVIDIA GPU with CUDA (optional but strongly recommended — CPU-only works, noticeably slower)
- [Ollama](https://ollama.com) installed separately, if using the local LLM backend

### Installation

```bash
git clone https://github.com/<your-username>/sara-ai.git
cd sara-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Place the Kokoro TTS model files (`kokoro-v1.0.onnx`, `voices-v1.0.bin`) under `models/`, then:

```bash
python main.py
```

### Building a standalone `.exe`

```bash
pip install -r requirements-build.txt
pyinstaller sara_ai.spec
```

See [`BUILD.md`](./BUILD.md) for the full packaging guide, including known rough edges around CUDA execution providers and `comtypes` caching in a frozen build.

---

## Configuration

All tuning lives in `.env` (see [`.env.example`](./.env.example) for the full, documented list — ~70 settings covering LLM backend, TTS voice/speed per language, AEC parameters, Whisper decoding thresholds, RAG retrieval, and tool-calling). Every setting has a safe default; nothing needs to be set to run the app.

| Category | Example settings |
|---|---|
| LLM backend | `LLM_BACKEND`, `OLLAMA_MODEL`, `GEMINI_API_KEY` |
| Voice | `KOKORO_VOICE_EN`, `KOKORO_VOICE_HI`, `KOKORO_SPEED_*` |
| Echo cancellation | `AEC_ENABLED`, `AEC_STREAM_DELAY_MS`, `AEC_ENABLE_NS` |
| Speech recognition | `WHISPER_MODEL_SIZE`, `STT_NO_SPEECH_THRESHOLD` |
| Long-term memory | `RAG_ENABLED`, `EMBEDDING_MODEL`, `RAG_TOP_K` |
| Tool-calling | `TOOL_CALLING_ENABLED`, `TOOL_CALLING_TIMEOUT_S` |

---

## Roadmap

- [x] Real-time WebRTC acoustic echo cancellation
- [x] Single-writer WAL-mode persistence, consolidated DB path
- [x] Long-term semantic memory (RAG) over conversation history
- [x] LLM tool-calling fallback for unmatched natural-language commands
- [x] Rotating logs + pre-flight health diagnostics
- [x] PyInstaller packaging
- [ ] Custom-trained wake-word model (currently STT-fallback based)
- [ ] Cross-platform support (currently Windows-only, tightly coupled to `winreg`/`pycaw`/`ctypes.windll`)
- [ ] Proactive suggestions (calendar-aware, battery/system-state-aware)

---

## License

MIT — see [LICENSE](./LICENSE).

---

<div align="center">

Built to explore what a genuinely offline, resource-constrained voice assistant looks like when the engineering constraints are taken seriously.

</div>
#   S A R A - A I - A u t o m a t i o n - P o w e r e d - P e r s o n a l - D e s k t o p - A s s i s t a n t  
 #   S A R A - A I - A u t o m a t i o n - P o w e r e d - P e r s o n a l - D e s k t o p - A s s i s t a n t  
 