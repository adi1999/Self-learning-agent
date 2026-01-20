# Design Document: Demo-Driven Workflow Learning & Automation
 
**Platform:** macOS  
**Last Updated:** January 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture](#3-architecture)
4. [Core Components](#4-core-components)
5. [Data Models](#5-data-models)
6. [Workflow Pipeline](#6-workflow-pipeline)
7. [Goal-Based Execution Engine](#7-goal-based-execution-engine)
8. [AI/ML Integration](#8-aiml-integration)
9. [Safety & Guardrails](#9-safety--guardrails)
10. [Design Decisions & Tradeoffs](#10-design-decisions--tradeoffs)
11. [Limitations & Future Work](#11-limitations--future-work)

---

## 1. Executive Summary

### Problem Statement

Users frequently perform repetitive computer tasks that span multiple applications (browser + desktop). Traditional automation solutions require:
- Technical scripting knowledge
- Brittle element selectors that break with UI changes
- Separate tools for browser vs. desktop automation

### Solution

A **demonstration-based workflow automation system** that:

1. **Observes** a human performing a real computer task (~2-3 minutes)
2. **Learns** the workflow from that demonstration using AI
3. **Automates** the task with new inputs — no code changes required

### Key Innovation: Goal-Oriented Execution

Instead of replaying exact actions (click at x=450, y=230), the system:
- Infers **goals** from actions ("enter search query", "click first result")
- Creates **multiple strategies** to achieve each goal
- Uses **success criteria** to verify goals are met
- **Adapts** when UI changes or elements move

---

## 2. System Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER DEMONSTRATION                               │
│  Browser: Search "best pizza sf" → Click result → Copy info             │
│  Desktop: Open Notes → Paste → Save                                      │
│  Voice: "I want to find restaurants and save them to notes"             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           RECORDING PHASE                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │   Browser    │ │   Desktop    │ │    Input     │ │    Voice     │   │
│  │   Observer   │ │   Observer   │ │   Observer   │ │   Recorder   │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
│                                    │                                     │
│                    ┌───────────────┴───────────────┐                    │
│                    │      Session Artifact         │                    │
│                    │   (screenshots, events, audio)│                    │
│                    └───────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          COMPILATION PHASE                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │    Intent    │ │  Parameter   │ │    Goal      │ │   Gemini     │   │
│  │  Classifier  │ │   Detector   │ │   Inferrer   │ │  Enrichment  │   │
│  │   (GPT-4o)   │ │   (GPT-4o)   │ │   (GPT-4o)   │ │   (Vision)   │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
│                                    │                                     │
│                    ┌───────────────┴───────────────┐                    │
│                    │      Goal Workflow Recipe     │                    │
│                    │  (goals, strategies, params)  │                    │
│                    └───────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION PHASE                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │    Goal      │ │   Browser    │ │   Desktop    │ │   Gemini     │   │
│  │   Executor   │ │   Executor   │ │   Executor   │ │   Agent      │   │
│  │  (orchestr.) │ │ (Playwright) │ │ (PyAutoGUI)  │ │  (fallback)  │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
│                                    │                                     │
│                    ┌───────────────┴───────────────┐                    │
│                    │      Execution Result         │                    │
│                    │   (success, extracted data)   │                    │
│                    └───────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Architecture

### Directory Structure

```
workflow-automation/
├── src/
│   ├── cli/                        # Command-Line Interfaces
│   │   ├── record.py               # Recording CLI with user guide
│   │   ├── compile.py              # Compilation CLI with mode selection
│   │   └── replay.py               # Replay CLI with parameter info
│   │
│   ├── observer/                   # Recording Components
│   │   ├── session_recorder.py     # Main orchestrator
│   │   ├── browser_observer.py     # Playwright-based browser capture
│   │   ├── desktop_observer.py     # macOS accessibility tree (macapptree)
│   │   ├── input_observer.py       # pynput mouse/keyboard capture
│   │   └── voice_recorder.py       # PyAudio recording + Whisper transcription
│   │
│   ├── interpreter/                # Recording Analysis
│   │   ├── intent_classifier.py    # LLM-based action intent classification
│   │   ├── segmenter.py            # Groups events into semantic steps
│   │   └── voice_analyzer.py       # Extracts hints from voice transcription
│   │
│   ├── compiler/                   # Workflow Compilation
│   │   ├── workflow_compiler.py    # Main compiler orchestrator
│   │   ├── goal_inferrer.py        # Converts actions → goals (KEY COMPONENT)
│   │   ├── parameter_detector.py   # Identifies variable inputs
│   │   └── template_detector.py    # Finds patterns for parameterization
│   │
│   ├── executor/                   # Workflow Execution
│   │   ├── workflow_executor.py    # Main execution orchestrator
│   │   ├── goal_executor.py        # Goal-based execution engine (KEY COMPONENT)
│   │   ├── browser_executor.py     # Playwright browser automation
│   │   ├── desktop_executor.py     # PyAutoGUI + accessibility automation
│   │   └── gemini_agent.py         # Gemini Computer Use fallback
│   │
│   ├── models/                     # Data Models (Pydantic)
│   │   ├── session_artifact.py     # Raw recording data
│   │   ├── semantic_trace.py       # Interpreted semantic steps
│   │   ├── workflow_recipe.py      # Traditional workflow format
│   │   ├── goal_step.py            # Goal-based workflow format (RECOMMENDED)
│   │   └── element_reference.py    # Multi-strategy element location
│   │
│   └── utils/                      # Shared Utilities
│       ├── config.py               # Configuration management
│       ├── logger.py               # Structured logging
│       ├── llm_client.py           # OpenAI GPT-4o wrapper
│       ├── gemini_client.py        # Google Gemini wrapper (vision + computer use)
│       ├── safety_guard.py         # Dangerous operation blocking
│       ├── audit_log.py            # Execution audit trail
│       └── rate_limiter.py         # API rate limiting
│
├── artifacts/
│   ├── sessions/                   # Recorded session data
│   │   └── session_<id>/
│   │       ├── session.json        # Metadata and timeline
│   │       ├── screenshots/        # Captured screenshots
│   │       └── voice.wav           # Audio recording
│   │
│   ├── recipes/                    # Compiled workflow recipes
│   │   ├── *_goals.json            # Goal-based workflows (recommended)
│   │   └── *.json                  # Traditional workflows
│   │
│   └── audit_logs/                 # Execution audit trails
│
└── requirements.txt
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Browser Automation** | Playwright | Reliable browser control, DOM access, screenshots |
| **Desktop Automation** | PyAutoGUI + macapptree | Mouse/keyboard control, accessibility tree |
| **Voice Recording** | PyAudio | Real-time audio capture |
| **Voice Transcription** | OpenAI Whisper | Speech-to-text |
| **Intent Classification** | GPT-4o | Understanding action purpose |
| **Visual Analysis** | Gemini 2.0 Flash | Page analysis, element finding, extraction |
| **Agentic Fallback** | Gemini Computer Use | When all strategies fail |
| **Data Validation** | Pydantic | Type-safe data models |

---

## 4. Core Components

### 4.1 Session Recorder

**Purpose:** Orchestrates all observers during recording.

**Responsibilities:**
- Starts/stops all observers in sync
- Manages session lifecycle
- Handles graceful shutdown (Ctrl+C)
- Saves session artifact

```python
class SessionRecorder:
    def __init__(self, output_dir, use_browser, initial_url, enable_voice):
        self.browser_observer = BrowserObserver()
        self.desktop_observer = DesktopObserver()
        self.input_observer = InputObserver()
        self.voice_recorder = VoiceRecorder() if enable_voice else None
    
    def start(self) -> SessionArtifact:
        # Start all observers
        # Wait for Ctrl+C
        # Stop all observers
        # Save and return artifact
```

### 4.2 Goal Inferrer

**Purpose:** Converts raw recorded actions into goal-oriented steps.

**Key Innovation:** Looks at OUTCOMES, not just ACTIONS.

```python
class GoalInferrer:
    def infer_goals_from_semantic_steps(
        self,
        semantic_steps: List[SemanticStep],
        voice_context: Optional[VoiceContext],
        extraction_schemas: Dict,
        detected_parameters: List[ParameterCandidate]
    ) -> List[GoalStep]:
        
        # 1. Analyze step sequence with LLM to understand intent
        step_analysis = self._analyze_step_sequence_with_llm(steps, voice_context)
        
        # 2. Merge behavioral sequences (click+type → search)
        merged_steps = self._merge_focus_clicks(steps, step_analysis)
        
        # 3. Convert each step to a goal with multiple strategies
        goals = []
        for step in merged_steps:
            goal = self._create_goal_from_step(step, step_analysis)
            goals.append(goal)
        
        return goals
```

**Goal Types:**

| Type | Description | Example |
|------|-------------|---------|
| `navigate` | Go to URL | Open google.com |
| `write` | Enter text input | Type search query |
| `select` | Click element | Click search result |
| `extract` | Get data from page | Get restaurant name, rating |
| `launch` | Open/switch app | Switch to Notes |
| `shortcut` | Keyboard shortcut | Cmd+V to paste |
| `save` | Save document | Cmd+S |
| `scroll` | Scroll page | Scroll to see more results |

### 4.3 Goal Executor

**Purpose:** Executes goal-oriented workflows with multi-strategy fallback.

**Key Innovation:** Tries multiple strategies until success criteria are met.

```python
class GoalExecutor:
    def execute_goal(self, goal: GoalStep) -> GoalResult:
        # Sort strategies by priority
        strategies = sorted(goal.strategies, key=lambda s: s.priority, reverse=True)
        
        for strategy in strategies:
            # Safety check
            if not safety_guard.check(strategy):
                continue
            
            # Execute strategy
            result = self._execute_strategy(goal, strategy)
            
            # Check success criteria
            if self._check_success_criteria(goal.success_criteria):
                return GoalResult(success=True, strategy_used=strategy.name)
        
        # All strategies failed - try Gemini agent fallback
        if goal.fallback_to_agent:
            return self._gemini_agent_fallback(goal)
        
        return GoalResult(success=False)
```

### 4.4 Browser Executor

**Purpose:** Executes browser-specific actions using Playwright.

**Features:**
- Multiple element location strategies
- CAPTCHA detection and pause
- Human-like typing with realistic delays
- Screenshot capture for debugging
- Gemini-powered visual element finding

```python
class BrowserExecutor:
    def _find_element(self, ref: ElementReference) -> Optional[Locator]:
        # Strategy 1: CSS Selector (fastest)
        if ref.selector:
            element = self.page.locator(ref.selector).first
            if element.is_visible():
                return element
        
        # Strategy 2: Text content match
        if ref.text_content:
            element = self.page.get_by_text(ref.text_content).first
            if element.is_visible():
                return element
        
        # Strategy 3: ARIA role + name
        if ref.role:
            element = self.page.get_by_role(ref.role, name=ref.name)
            if element.is_visible():
                return element
        
        # Strategy 4: Gemini visual (slowest but most robust)
        if ref.visual_description:
            coords = self.gemini.find_element(screenshot, ref.visual_description)
            return coords
        
        return None
```

### 4.5 Desktop Executor

**Purpose:** Executes desktop-specific actions using PyAutoGUI and accessibility APIs.

**Features:**
- App activation via AppleScript
- Accessibility tree element finding (macapptree)
- Reliable clipboard operations
- Keyboard shortcut execution
- Gemini visual fallback for element location

```python
class DesktopExecutor:
    def _execute_click(self, step: WorkflowStep) -> DesktopStepResult:
        # Strategy 1: Accessibility tree (most reliable)
        if step.element_reference and step.element_reference.accessibility_role:
            tree = macapptree.get_app_tree(step.app_name)
            element = self._find_in_tree(tree, step.element_reference)
            if element:
                click_at(element.position)
                return DesktopStepResult(success=True)
        
        # Strategy 2: Coordinates (if recorded)
        if step.element_reference and step.element_reference.coordinates:
            click_at(step.element_reference.coordinates)
            return DesktopStepResult(success=True)
        
        # Strategy 3: Gemini visual
        screenshot = capture_screen()
        coords = self.gemini.find_element(screenshot, step.element_reference.visual_description)
        if coords:
            click_at(coords)
            return DesktopStepResult(success=True)
        
        return DesktopStepResult(success=False)
```

---

## 5. Data Models

### 5.1 Session Artifact (Recording Output)

```python
class SessionArtifact(BaseModel):
    """Raw output from recording phase."""
    session_id: str
    start_time: datetime
    end_time: datetime
    
    # Timeline of all events
    timeline: List[TimelineEvent]
    
    # Metadata
    initial_url: Optional[str]
    apps_used: List[str]
    
    # Voice
    voice_audio_path: Optional[Path]
    voice_transcription: Optional[VoiceTranscription]

class TimelineEvent(BaseModel):
    timestamp: float
    event_type: str  # "click", "type", "navigate", "screenshot", etc.
    
    # Platform context
    platform: str  # "browser" or "desktop"
    app_name: str
    url: Optional[str]
    
    # Event-specific data
    input_event: Optional[InputEvent]
    navigation_outcome: Optional[NavigationOutcome]
    screenshot_path: Optional[Path]
    element_info: Optional[ElementInfo]
```

### 5.2 Goal Workflow (Recommended Format)

```python
class GoalWorkflow(BaseModel):
    """Goal-oriented workflow - recommended format."""
    workflow_id: str
    name: str
    description: str
    
    # Parameters with defaults
    parameters: Dict[str, str]  # {"search_topic": "best pizza", "location": "sf"}
    
    # Goal steps
    steps: List[GoalStep]
    
    # Metadata
    version: str = "2.0"
    created_from_session: str
    voice_analyzed: bool
    gemini_enriched: bool

class GoalStep(BaseModel):
    """A single goal to achieve."""
    step_id: str
    step_number: int
    
    # Goal definition
    goal_type: GoalType  # navigate, write, select, extract, launch, shortcut, save
    goal_description: str  # "Type: {{search_topic}} in {{location}}"
    
    # Context
    platform: str  # "browser" or "desktop"
    app_name: str
    
    # Success criteria
    success_criteria: SuccessCriteria
    
    # Multiple strategies (tried in priority order)
    strategies: List[Strategy]
    
    # Template for parameterization
    template: Optional[str]  # "{{search_topic}} in {{location}}"
    
    # Extraction schema (for extract goals)
    extraction_schema: Optional[Dict[str, ExtractionField]]
    
    # Fallback
    fallback_to_agent: bool = True
    agent_goal_prompt: Optional[str]

class Strategy(BaseModel):
    """A single strategy to achieve a goal."""
    name: str  # "css_selector", "text_match", "gemini_visual", etc.
    priority: int  # Higher = try first
    
    # Element location
    selector: Optional[str]
    text_match: Optional[str]
    role: Optional[str]
    visual_description: Optional[str]
    coordinates: Optional[Tuple[int, int]]
    
    # Action parameters
    input_value: Optional[str]  # For write goals
    shortcut_keys: Optional[str]  # For shortcut goals
    
    # Accessibility
    accessibility_role: Optional[str]
    accessibility_name: Optional[str]

class SuccessCriteria(BaseModel):
    """How to verify a goal was achieved."""
    url_contains: Optional[str]
    url_changed: bool = False
    page_contains_text: Optional[str]
    element_visible: Optional[str]
    extracted_fields: List[str] = []
    min_extracted_count: int = 0
    app_active: Optional[str]
    timeout_success: bool = False  # Consider success if no error after timeout
```

### 5.3 Element Reference (Multi-Strategy Location)

```python
class ElementReference(BaseModel):
    """Multi-strategy element identification."""
    
    # Primary identifiers (browser)
    selector: Optional[str]  # CSS selector
    xpath: Optional[str]
    text_content: Optional[str]
    
    # ARIA attributes
    role: Optional[str]  # button, textbox, link, etc.
    name: Optional[str]  # Accessible name
    
    # Visual identification (Gemini)
    visual_description: Optional[str]
    screenshot_region: Optional[Tuple[int, int, int, int]]
    
    # Coordinates (fallback)
    coordinates: Optional[Tuple[int, int]]
    
    # Desktop accessibility
    accessibility_role: Optional[str]
    accessibility_name: Optional[str]
    accessibility_path: Optional[List[str]]
```

---

## 6. Workflow Pipeline

### 6.1 Recording Phase

```
User Action                    Observers                      Output
───────────                    ─────────                      ──────
Click search box        →      InputObserver captures         TimelineEvent(
                               coordinates, timestamp          type="click",
                                                               coordinates=(450, 230))

Type "pizza sf"         →      InputObserver captures         TimelineEvent(
                               keystrokes                      type="type",
                                                               text="pizza sf")

Page loads              →      BrowserObserver captures       TimelineEvent(
                               URL change, screenshot          type="navigate",
                                                               url="google.com/search?q=...")

Voice: "searching       →      VoiceRecorder captures         voice.wav
for restaurants"               audio stream                    

Ctrl+C                  →      SessionRecorder stops          SessionArtifact(
                               all observers                   timeline=[...],
                                                               voice_audio_path=...)
```

### 6.2 Compilation Phase

```
SessionArtifact
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  1. SEGMENTATION                                                         │
│     Group raw events into semantic steps by:                            │
│     - App switching                                                      │
│     - Significant time gaps (>2s)                                        │
│     - Action completion (submit, navigation)                             │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. INTENT CLASSIFICATION (GPT-4o)                                       │
│     For each step, determine:                                            │
│     - Intent: search, navigate, copy, paste, save, etc.                 │
│     - Target: what element/content was interacted with                  │
│     - Context: why this action (voice hints help here)                  │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. VOICE ANALYSIS (Whisper + GPT-4o)                                    │
│     Extract from transcription:                                          │
│     - Task goal: "find restaurants and save to notes"                   │
│     - Parameter hints: "BEST PIZZA" → search_topic                      │
│     - Extraction hints: "NAME, RATING, ADDRESS"                         │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. PARAMETER DETECTION (GPT-4o)                                         │
│     Identify which values should be parameters:                          │
│     - Typed text in search contexts                                      │
│     - Values mentioned in voice narration                                │
│     - Create template: "{{search_topic}} in {{location}}"               │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  5. GOAL INFERENCE (GPT-4o)                                              │
│     Convert steps to goals:                                              │
│     - Merge focus clicks with typing (click+type → "write" goal)        │
│     - Determine flexible vs rigid success criteria                       │
│     - Create multiple strategies per goal                                │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  6. GEMINI ENRICHMENT (Vision)                                           │
│     Analyze screenshots to:                                              │
│     - Create extraction schemas                                          │
│     - Add visual descriptions for elements                               │
│     - Identify page types                                                │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
GoalWorkflow Recipe
```

### 6.3 Execution Phase

```
GoalWorkflow + Parameters
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  1. PARAMETER SUBSTITUTION                                               │
│     Replace {{param}} in all templates with provided values             │
│     "{{search_topic}} in {{location}}" → "best sushi in NYC"            │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FOR EACH GOAL:                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  2. SAFETY CHECK                                                    │ │
│  │     Verify action is not blocked (no shutdown, delete, etc.)       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  3. PLATFORM ROUTING                                                │ │
│  │     Browser goals → BrowserExecutor                                 │ │
│  │     Desktop goals → DesktopExecutor                                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  4. STRATEGY EXECUTION (in priority order)                          │ │
│  │     For each strategy until success:                                │ │
│  │       - Execute the strategy                                        │ │
│  │       - Check success criteria                                      │ │
│  │       - If success → next goal                                      │ │
│  │       - If fail → try next strategy                                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  5. GEMINI AGENT FALLBACK (if all strategies fail)                  │ │
│  │     Use Gemini Computer Use model for agentic control              │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
Execution Result (success, extracted_data, audit_log)
```

---

## 7. Goal-Based Execution Engine

### Why Goal-Based?

| Traditional Approach | Goal-Based Approach |
|---------------------|---------------------|
| "Click at (450, 230)" | "Enter search query" |
| Breaks if button moves | Finds button wherever it is |
| Single strategy | Multiple fallback strategies |
| No verification | Success criteria check |
| Brittle | Robust |

### Multi-Strategy Resolution

Each goal has multiple strategies sorted by priority:

```json
{
  "goal_type": "select",
  "goal_description": "Click the first search result",
  "strategies": [
    {
      "name": "css_selector",
      "priority": 90,
      "selector": "h3.LC20lb:first-child"
    },
    {
      "name": "text_match", 
      "priority": 80,
      "text_match": "contains query terms"
    },
    {
      "name": "role_based",
      "priority": 70,
      "role": "link",
      "visual_description": "in main content area"
    },
    {
      "name": "gemini_visual",
      "priority": 60,
      "visual_description": "first blue link in search results"
    },
    {
      "name": "coordinates",
      "priority": 30,
      "coordinates": [450, 280]
    }
  ],
  "success_criteria": {
    "url_changed": true
  }
}
```

**Execution Order:**
1. Try `css_selector` (fastest) → Check URL changed
2. If fail, try `text_match` → Check URL changed
3. If fail, try `role_based` → Check URL changed
4. If fail, try `gemini_visual` (slowest but robust) → Check URL changed
5. If fail, try `coordinates` (last resort) → Check URL changed
6. If all fail and `fallback_to_agent: true` → Gemini Computer Use

### Success Criteria Types

| Criteria | Use Case | Example |
|----------|----------|---------|
| `url_changed` | Navigation clicks | Click search result |
| `url_contains` | Specific navigation | Navigate to checkout |
| `page_contains_text` | Content verification | Form submitted |
| `element_visible` | UI state change | Modal opened |
| `extracted_fields` | Data extraction | Got restaurant name |
| `app_active` | App switching | Notes is frontmost |
| `timeout_success` | Best-effort actions | Type succeeded if no error |

---

## 8. AI/ML Integration

### 8.1 OpenAI GPT-4o

**Used for:** Text analysis at compile time

| Task | Input | Output |
|------|-------|--------|
| Intent Classification | Step sequence + context | Intent labels (search, navigate, copy) |
| Parameter Detection | Typed values + voice | Parameter candidates with names |
| Goal Inference | Semantic steps | Goal type, description, strategies |
| Voice Analysis | Transcription | Task goal, hints, parameters |

**Example Prompt (Intent Classification):**
```
Analyze this user action:
- App: Google Chrome
- URL: google.com
- Action: typed "best pizza sf"
- Context: after clicking search box

Classify the intent: search | navigate | fill_form | copy | paste | ...
```

### 8.2 Google Gemini

**Models Used:**
- `gemini-2.0-flash` — Vision analysis (fast, cheap)
- `gemini-2.5-computer-use-preview` — Agentic fallback (slower, powerful)

| Task | Model | Input | Output |
|------|-------|-------|--------|
| Element Finding | Flash | Screenshot + description | Coordinates (x, y) |
| Page Type Classification | Flash | Screenshot | Page type (search, detail, form) |
| Data Extraction | Flash | Screenshot + schema | Structured data |
| Extraction Schema Generation | Flash | Screenshot | Field definitions |
| Agentic Fallback | Computer Use | Screenshot + goal | Action to take |

**Example (Element Finding):**
```python
gemini.find_element(
    screenshot=page_screenshot,
    description="the search button, usually a magnifying glass icon"
)
# Returns: (892, 145) or None
```

### 8.3 OpenAI Whisper

**Used for:** Voice transcription

```python
# Transcribe recorded audio
transcription = whisper.transcribe(
    audio_path="voice.wav",
    model="whisper-1"
)
# Returns: "I'm searching for best pizza places in San Francisco 
#           and I want to save the name rating and address"
```

---

## 9. Safety & Guardrails

### 9.1 Safety Guard

Prevents execution of dangerous operations at the executor level.

```python
class SafetyGuard:
    # Blocked keyboard shortcuts
    BLOCKED_SHORTCUTS = {
        ("command", "option", "control", "eject"),  # Shutdown
        ("command", "control", "eject"),            # Restart
        ("command", "option", "escape"),            # Force Quit
        ("command", "shift", "q"),                  # Logout
    }
    
    # Blocked typed commands (in Terminal)
    BLOCKED_TYPE_PATTERNS = [
        r"sudo\s+rm\s+-rf\s+/",      # Delete everything
        r"rm\s+-rf\s+~",              # Delete home
        r"mkfs\.",                    # Format disk
        r"shutdown",                  # Shutdown
        r"diskutil\s+eraseDisk",      # Erase disk
    ]
    
    # Blocked URLs
    BLOCKED_URL_PATTERNS = [
        r"chrome://settings/clearBrowserData",
        r"chrome://settings/reset",
    ]
    
    def check_shortcut(self, keys: Tuple[str, ...]) -> SafetyCheck:
        """Returns SafetyCheck(allowed=False) if blocked."""
        
    def check_typed_text(self, text: str, app_name: str) -> SafetyCheck:
        """Checks typed text in Terminal-like apps."""
        
    def check_url(self, url: str) -> SafetyCheck:
        """Checks navigation URLs."""
```

### 9.2 Audit Logging

All executions are logged for debugging and compliance.

```python
class AuditLog:
    def log_step(self, entry: AuditEntry):
        """Log each step execution."""
        # Logs to: artifacts/audit_logs/<workflow_id>_<timestamp>.jsonl

@dataclass
class AuditEntry:
    timestamp: str
    workflow_id: str
    step_id: str
    action_type: str
    parameters: Dict[str, Any]
    result: str  # "success", "failed", "blocked"
    duration_ms: int
    error: Optional[str]
```

### 9.3 Rate Limiting

Prevents API quota exhaustion.

```python
class RateLimiter:
    def __init__(self, calls_per_minute: int = 30):
        self.calls_per_minute = calls_per_minute
        
    def acquire(self):
        """Blocks if rate limit exceeded."""
        # Ensures max 30 calls/minute to Gemini
```

---

## 10. Design Decisions & Tradeoffs

### 10.1 Goal-Based vs Action-Based

**Decision:** Use goal-based execution as the primary approach.

**Rationale:**
- UI elements move; goals don't change
- Multiple strategies provide robustness
- Success criteria verify outcomes
- Easier to debug ("goal failed" vs "click failed at x=450")

**Tradeoff:**
- More complex compilation
- Higher latency (multiple strategy attempts)
- Requires AI for goal inference

### 10.2 Compile-Time vs Runtime AI

**Decision:** Use AI primarily at compile time, not runtime.

**Rationale:**
- Faster execution (no LLM calls during replay)
- Predictable behavior (same recipe = same execution)
- Lower cost (LLM calls only during compilation)
- Offline replay capability

**Tradeoff:**
- Less adaptive to runtime changes
- Requires recompilation for significant UI changes

**Exception:** Gemini visual finding is used at runtime as a fallback.

### 10.3 Fresh Browser Context vs Persistent Profile

**Decision:** Default to fresh browser context (no persistent profile).

**Rationale:**
- Avoids profile corruption crashes
- Cleaner state for each execution
- More reproducible results

**Tradeoff:**
- Can't maintain login state across runs
- May trigger more CAPTCHAs (no cookie history)

### 10.4 Parameter Detection Strategy

**Decision:** Only include parameters that are actually used in templates.

**Rationale:**
- Cleaner workflow recipes
- No confusion about unused parameters
- Parameters are clearly mapped to steps

**Tradeoff:**
- Some detected values may be discarded
- Requires template matching during compilation

---

## 11. Limitations & Future Work

### Current Limitations

| Limitation | Description | Workaround |
|------------|-------------|------------|
| **macOS only** | Desktop automation uses macapptree | Use browser-only mode on other platforms |
| **Single monitor** | Screenshot capture assumes primary display | Use primary monitor for demos |
| **English voice** | Whisper optimized for English | Speak clearly in English |
| **No auth persistence** | Fresh browser loses login state | Add manual login step to workflow |
| **Limited CAPTCHA handling** | Pauses for manual completion | Complete CAPTCHA manually when prompted |

### Future Enhancements

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| **Workflow versioning** | Track changes to recipes over time | Medium |
| **Workflow testing framework** | Automated validation of compiled workflows | Medium |
| **Multi-monitor support** | Handle multiple displays | Low |
| **Windows/Linux support** | Port desktop automation to other platforms | Medium |
| **Workflow chaining** | Compose multiple workflows | Low |
| **Visual workflow editor** | GUI for editing compiled workflows | Low |
| **Cloud execution** | Run workflows on remote machines | Medium |
| **Webhook triggers** | Start workflows via HTTP | Low |

---

## Appendix A: CLI Reference

### Record Command

```bash
python -m src.cli.record [OPTIONS]

Options:
  --url TEXT          Starting URL (default: https://google.com)
  --no-browser        Desktop recording only
  --no-voice          Disable voice recording
  --output-dir PATH   Session output directory

Examples:
  python -m src.cli.record
  python -m src.cli.record --url https://yelp.com
  python -m src.cli.record --no-voice
```

### Compile Command

```bash
python -m src.cli.compile [OPTIONS]

Required:
  --session TEXT      Session ID to compile
  --name TEXT         Workflow name

Options:
  --goals             Compile to goal-based workflow (RECOMMENDED)
  --description TEXT  Workflow description
  --output PATH       Output file path
  --no-llm            Disable LLM analysis
  --no-gemini         Disable Gemini enrichment

Examples:
  python -m src.cli.compile --session session_abc123 --name my_workflow --goals
  python -m src.cli.compile --session session_abc123 --name my_workflow --no-gemini
```

### Replay Command

```bash
python -m src.cli.replay [OPTIONS]

Required:
  --recipe PATH       Path to workflow recipe

Options:
  --goals             Load as goal-based workflow
  --params JSON       Parameters as JSON string
  --params-file PATH  Parameters from JSON file
  --url TEXT          Override starting URL
  --headless          Run browser in headless mode
  --dry-run           Show steps without executing

Examples:
  python -m src.cli.replay --recipe my_workflow_goals.json --goals \
      --params '{"query": "test"}'
  python -m src.cli.replay --recipe my_workflow_goals.json --goals --dry-run
```

---

## Appendix B: Example Workflow Recipe

```json
{
  "workflow_id": "restaurant_search",
  "name": "restaurant_search",
  "description": "Search for restaurants and save to Notes",
  "parameters": {
    "search_topic": "best pizza places",
    "location": "sf"
  },
  "steps": [
    {
      "step_id": "goal_1",
      "step_number": 1,
      "goal_type": "write",
      "goal_description": "Type: {{search_topic}} in {{location}}",
      "platform": "browser",
      "app_name": "Google Chrome",
      "success_criteria": {
        "timeout_success": true
      },
      "strategies": [
        {
          "name": "focused_type",
          "priority": 70,
          "input_value": "{{search_topic}} in {{location}}"
        }
      ],
      "template": "{{search_topic}} in {{location}}"
    },
    {
      "step_id": "goal_2",
      "step_number": 2,
      "goal_type": "select",
      "goal_description": "Click first search result",
      "platform": "browser",
      "app_name": "Google Chrome",
      "success_criteria": {
        "url_changed": true
      },
      "strategies": [
        {
          "name": "css_selector",
          "priority": 90,
          "selector": "h3.LC20lb"
        },
        {
          "name": "gemini_visual",
          "priority": 60,
          "visual_description": "first blue link in search results"
        }
      ]
    },
    {
      "step_id": "goal_3",
      "step_number": 3,
      "goal_type": "extract",
      "goal_description": "Extract restaurant data",
      "platform": "browser",
      "app_name": "Google Chrome",
      "success_criteria": {
        "min_extracted_count": 1
      },
      "strategies": [
        {
          "name": "gemini_vision_extract",
          "priority": 100,
          "visual_description": "Extract structured data"
        }
      ],
      "extraction_schema": {
        "restaurant_name": {
          "description": "Name of the restaurant",
          "visual_hint": "Large text at top"
        },
        "rating": {
          "description": "Star rating",
          "visual_hint": "Stars near the name"
        },
        "address": {
          "description": "Street address",
          "visual_hint": "Below the name"
        }
      }
    },
    {
      "step_id": "goal_4",
      "step_number": 4,
      "goal_type": "launch",
      "goal_description": "Switch to Notes",
      "platform": "desktop",
      "app_name": "Notes",
      "success_criteria": {
        "app_active": "Notes"
      },
      "strategies": [
        {
          "name": "activate_app",
          "priority": 100
        }
      ]
    },
    {
      "step_id": "goal_5",
      "step_number": 5,
      "goal_type": "shortcut",
      "goal_description": "Paste extracted content",
      "platform": "desktop",
      "app_name": "Notes",
      "success_criteria": {
        "timeout_success": true
      },
      "strategies": [
        {
          "name": "paste_content",
          "priority": 100,
          "shortcut_keys": "command+v"
        }
      ],
      "template": "{{extracted_content}}"
    },
    {
      "step_id": "goal_6",
      "step_number": 6,
      "goal_type": "save",
      "goal_description": "Save document",
      "platform": "desktop",
      "app_name": "Notes",
      "success_criteria": {
        "timeout_success": true
      },
      "strategies": [
        {
          "name": "save_shortcut",
          "priority": 100,
          "shortcut_keys": "command+s"
        }
      ]
    }
  ],
  "version": "2.0",
  "voice_analyzed": true,
  "gemini_enriched": true
}
```

---

