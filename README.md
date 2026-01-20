# 🤖 Workflow Automation System

> **Learn once, automate forever.** Record a human demonstration, and let AI replay it with new inputs.

A demonstration-based workflow automation system that observes a human performing a computer task, learns the workflow from that demonstration, and fully automates it with new parameters — no code changes required.

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)
![macOS](https://img.shields.io/badge/Platform-macOS-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎥 **Demo-based Learning** | Record yourself performing a task — no scripting required |
| 🎙️ **Voice Narration** | Speak while you work to help AI understand your intent |
| 🌐 **Browser + Desktop** | Seamlessly automate across Chrome and native macOS apps |
| 🎯 **Goal-Oriented Execution** | AI infers goals, not just actions — handles UI changes gracefully |
| 🔄 **Parameterization** | Automatically detects variable inputs for reuse with new data |
| 📊 **Smart Extraction** | Uses Gemini Vision to extract structured data from any page |
| 🛡️ **Safety Guardrails** | Blocks dangerous operations (system shutdown, file deletion, etc.) |

---

## 🚀 Quick Start

### Prerequisites

- **macOS** (required for desktop automation via accessibility APIs)
- **Python 3.11+**
- **Google Chrome** (for browser automation)

### Installation

```bash
cd workflow-automation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

### Environment Setup

```bash
# Required for LLM-powered compilation
export OPENAI_API_KEY="your-openai-key"

# Required for visual analysis and extraction
export GOOGLE_API_KEY="your-gemini-key"
```

### macOS Permissions

Grant these permissions in **System Preferences > Security & Privacy**:
- ✅ **Screen Recording** — for capturing screenshots
- ✅ **Accessibility** — for tracking mouse/keyboard events
- ✅ **Microphone** — for voice narration (optional but recommended)

---

## 📋 The Three-Step Workflow

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   RECORD    │ ───▶ │   COMPILE   │ ───▶ │   REPLAY    │
│  (observe)  │      │   (learn)   │      │  (automate) │
└─────────────┘      └─────────────┘      └─────────────┘
```

### Step 1: Record a Demonstration

```bash
python -m src.cli.record
```

Perform your task naturally while optionally narrating your intent. The system captures:
- Mouse clicks and keyboard input
- Screenshots at key moments
- Browser navigation and page content
- Voice narration (transcribed via Whisper)

**Voice Tips for Better Results:**
```
✓ "I'm searching for BEST PIZZA PLACES in SAN FRANCISCO"
  → Creates parameters: search_topic, location

✓ "I want to extract the RESTAURANT NAME, RATING, and ADDRESS"
  → Creates extraction schema for these fields

✓ "Now I'm saving this to my NOTES app"
  → Helps identify the goal of this step
```

**Recording Options:**
```bash
# Start with specific URL
python -m src.cli.record --url https://yelp.com

# Desktop only (no browser)
python -m src.cli.record --no-browser

# Without voice (not recommended)
python -m src.cli.record --no-voice
```

---

### Step 2: Compile to Workflow

```bash
python -m src.cli.compile \
    --session session_abc123 \
    --name restaurant_search \
    --goals
```

The compiler analyzes your recording and creates a reusable workflow recipe.

#### Compilation Modes

| Mode | Flag | Description | Recommended |
|------|------|-------------|-------------|
| **Goal-based** | `--goals` | AI infers goals with multiple strategies. Most robust. | ✅ **Yes** |
| Traditional | (default) | Direct action replay. Less flexible. | No |
| Heuristics only | `--no-llm --no-gemini` | No AI, pattern matching only. Fastest but least accurate. | No |

**⭐ Best Performance: Always use `--goals`**

The goal-based mode:
- Infers user **intent**, not just actions
- Creates **multiple strategies** per step (fallbacks if one fails)
- Handles **UI changes** gracefully (button moved? finds it anyway)
- Uses **success criteria** to verify each step worked

**Compilation Options:**
```bash
# Full AI analysis (recommended)
python -m src.cli.compile --session session_abc123 --name my_workflow --goals

# Without Gemini visual analysis
python -m src.cli.compile --session session_abc123 --name my_workflow --goals --no-gemini

# Heuristics only (no API calls, fastest)
python -m src.cli.compile --session session_abc123 --name my_workflow --no-llm --no-gemini
```

---

### Step 3: Replay with New Inputs

```bash
python -m src.cli.replay \
    --recipe artifacts/recipes/restaurant_search_goals.json \
    --goals \
    --params '{"search_topic": "best sushi restaurants", "location": "NYC"}'
```

The system executes the workflow with your new parameters, handling:
- Dynamic element location
- Page load timing
- CAPTCHA detection
- Fallback strategies when primary approach fails

**Replay Options:**
```bash
# Dry run (preview steps without executing)
python -m src.cli.replay --recipe my_workflow_goals.json --goals --dry-run

# Headless mode (no visible browser)
python -m src.cli.replay --recipe my_workflow_goals.json --goals --headless \
    --params '{"query": "test"}'

# Custom starting URL
python -m src.cli.replay --recipe my_workflow_goals.json --goals \
    --url https://google.com --params '{"query": "test"}'
```

---

## 🏗️ Architecture

```
workflow-automation/
├── src/
│   ├── cli/                    # Command-line interfaces
│   │   ├── record.py           # Recording CLI
│   │   ├── compile.py          # Compilation CLI
│   │   └── replay.py           # Replay CLI
│   │
│   ├── observer/               # Recording components
│   │   ├── session_recorder.py # Main recording orchestrator
│   │   ├── browser_observer.py # Playwright-based browser capture
│   │   ├── desktop_observer.py # macOS accessibility tree capture
│   │   ├── input_observer.py   # Mouse/keyboard event capture
│   │   └── voice_recorder.py   # Audio recording
│   │
│   ├── compiler/               # Workflow compilation
│   │   ├── workflow_compiler.py # Main compiler
│   │   ├── goal_inferrer.py    # Converts actions → goals
│   │   ├── parameter_detector.py # Finds variable inputs
│   │   └── template_detector.py  # Identifies patterns
│   │
│   ├── executor/               # Workflow execution
│   │   ├── workflow_executor.py # Main executor
│   │   ├── goal_executor.py    # Goal-based execution engine
│   │   ├── browser_executor.py # Browser automation
│   │   └── desktop_executor.py # Desktop automation
│   │
│   ├── models/                 # Data models
│   │   ├── session_artifact.py # Raw recording data
│   │   ├── workflow_recipe.py  # Traditional workflow format
│   │   ├── goal_step.py        # Goal-based workflow format
│   │   └── element_reference.py # Multi-strategy element location
│   │
│   └── utils/                  # Utilities
│       ├── gemini_client.py    # Google Gemini API wrapper
│       ├── llm_client.py       # OpenAI API wrapper
│       ├── safety_guard.py     # Dangerous operation blocking
│       ├── audit_log.py        # Execution logging
│       └── rate_limiter.py     # API rate limiting
│
├── artifacts/
│   ├── sessions/               # Recorded session data
│   ├── recipes/                # Compiled workflow recipes
│   └── audit_logs/             # Execution audit trails
│
└── requirements.txt
```

---

## 🎯 Goal-Based Execution (Recommended)

The `--goals` compilation mode creates **goal-oriented workflows** that are significantly more robust than traditional action replay.

### How It Works

```
Traditional:                    Goal-Based:
─────────────                   ───────────
1. Click (x=450, y=230)         1. GOAL: Enter search query
2. Type "pizza"                    Strategy A: Type in focused element
3. Press Enter                     Strategy B: Find input by placeholder
                                   Strategy C: Gemini visual locate
                                   Success: Text appears in search box
```

### Goal Types

| Goal Type | Description | Example |
|-----------|-------------|---------|
| `navigate` | Go to a URL | Open google.com |
| `write` | Enter text | Type search query |
| `select` | Click an element | Click search result |
| `extract` | Get data from page | Extract restaurant name, rating |
| `launch` | Open desktop app | Switch to Notes |
| `shortcut` | Keyboard shortcut | Cmd+V to paste |
| `save` | Save document | Cmd+S |

### Multi-Strategy Fallback

Each goal has multiple strategies ranked by priority:

```json
{
  "goal_type": "select",
  "goal_description": "Click the first search result",
  "strategies": [
    {"name": "css_selector", "priority": 90, "selector": "h3.LC20lb"},
    {"name": "text_match", "priority": 80, "text_match": "contains search term"},
    {"name": "gemini_visual", "priority": 60, "visual_description": "first blue link"}
  ]
}
```

If `css_selector` fails (element not found), automatically tries `text_match`, then `gemini_visual`.

---

## 🛡️ Safety Features

The system includes guardrails to prevent dangerous operations:

### Blocked Operations

| Category | Examples |
|----------|----------|
| **System** | Shutdown, restart, logout |
| **Destructive** | `rm -rf /`, format disk, empty trash |
| **Security** | Disable SIP, modify keychain |
| **Browser** | Clear all browser data, reset settings |

### Blocked Shortcuts

- `Cmd+Option+Esc` (Force Quit)
- `Cmd+Shift+Q` (Logout)
- `Cmd+Option+Control+Eject` (Shutdown)

These operations are blocked at the executor level and logged for audit.

---

## 📊 Example Workflows

### Restaurant Search → Notes

```bash
# Record
python -m src.cli.record --url https://google.com

# During recording:
# 1. Search "best pizza places in sf"
# 2. Click a result
# 3. Copy restaurant info
# 4. Open Notes app
# 5. Paste and save

# Compile
python -m src.cli.compile --session session_xxx --name restaurant_search --goals

# Replay with new inputs
python -m src.cli.replay \
    --recipe artifacts/recipes/restaurant_search_goals.json \
    --goals \
    --params '{"query": "best sushi restaurants in nyc"}'
```

### Job Application Tracker

```bash
# Record yourself:
# 1. Open LinkedIn Jobs
# 2. Search for a role
# 3. Extract job details
# 4. Log to spreadsheet

# Replay to track multiple jobs automatically
python -m src.cli.replay \
    --recipe job_tracker_goals.json \
    --goals \
    --params '{"job_title": "Software Engineer", "location": "Remote"}'
```

---

## 🔧 Configuration

Edit `src/utils/config.py` or set environment variables:

```python
# API Keys
OPENAI_API_KEY      # For GPT-4o text analysis
GOOGLE_API_KEY      # For Gemini vision analysis

# Directories
SESSIONS_DIR        # Where recordings are saved
RECIPES_DIR         # Where compiled workflows are saved

# Defaults
BROWSER_DEFAULT_URL # Starting URL for recordings
```

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Permission denied" | Grant Screen Recording + Accessibility in System Preferences |
| "No module named 'pydantic'" | Activate virtual environment: `source .venv/bin/activate` |
| Browser crashes | Persistent profiles disabled by default; if enabled, delete `artifacts/.browser_profile` |
| CAPTCHA appears | System auto-detects and pauses; complete manually then continue |
| Element not found | Goal-based mode tries multiple strategies; check if page structure changed significantly |

### Debug Mode

```bash
# Verbose logging
LOG_LEVEL=DEBUG python -m src.cli.replay --recipe my_workflow_goals.json --goals
```

---

## 📈 Performance Tips

1. **Always use `--goals`** — Goal-based workflows are 3-5x more robust than traditional
2. **Narrate while recording** — Voice context dramatically improves parameter detection
3. **Keep recordings short** — 2-3 minutes is ideal; break complex flows into smaller workflows
4. **Use Gemini for extraction** — Visual extraction handles dynamic content better than selectors
5. **Test with dry-run first** — `--dry-run` shows steps without executing

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Playwright](https://playwright.dev/) — Browser automation
- [PyAutoGUI](https://pyautogui.readthedocs.io/) — Desktop automation
- [OpenAI Whisper](https://openai.com/whisper) — Voice transcription
- [Google Gemini](https://ai.google.dev/) — Visual analysis and computer use
- [macapptree](https://github.com/nicholascm/macapptree) — macOS accessibility tree

---

<p align="center">
  <b>Built with ❤️ for automating the mundane</b>
</p>
