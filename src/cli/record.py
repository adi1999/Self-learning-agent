#!/usr/bin/env python3
"""CLI entry point for recording workflows."""
import argparse
import sys
from pathlib import Path

# Add src to path if running as script
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(src_path))

from src.observer.session_recorder import SessionRecorder
from src.utils.config import config


RECORDING_GUIDE = """
╔════════════════════════════════════════════════════════════════════════════╗
║                        📹 WORKFLOW RECORDING GUIDE                          ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  🎯 WHAT TO DO:                                                            ║
║  ─────────────────────────────────────────────────────────────────────────║
║  1. Perform your task naturally (browser + desktop apps)                   ║
║  2. The system captures: clicks, typing, screenshots, and voice           ║
║  3. Press Ctrl+C when done                                                 ║
║                                                                            ║
║  🎙️ VOICE NARRATION (Highly Recommended):                                 ║
║  ─────────────────────────────────────────────────────────────────────────║
║  Speaking while you work helps the AI understand your INTENT:              ║
║                                                                            ║
║  ✓ "I'm searching for BEST PIZZA PLACES in SAN FRANCISCO"                 ║
║    → Creates parameters: search_topic="best pizza places", location="sf"   ║
║                                                                            ║
║  ✓ "I want to extract the RESTAURANT NAME, RATING, and ADDRESS"           ║
║    → Creates extraction schema for these specific fields                   ║
║                                                                            ║
║  ✓ "Now I'm saving this to my NOTES app"                                  ║
║    → Helps identify the goal of this step                                  ║
║                                                                            ║
║  🚫 AVOID:                                                                 ║
║  ─────────────────────────────────────────────────────────────────────────║
║  • Don't rush - give UI time to load between actions                       ║
║  • Don't use keyboard shortcuts unless that's how you'd normally do it    ║
║  • Avoid unnecessary clicks or typos (they'll be recorded!)               ║
║                                                                            ║
║  💡 PRO TIPS:                                                              ║
║  ─────────────────────────────────────────────────────────────────────────║
║  • Say parameter values OUT LOUD when typing them                          ║
║  • Mention what DATA you want to extract from pages                       ║
║  • ~2-3 minutes is ideal recording length                                  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""


def main():
    """Record a workflow session."""
    parser = argparse.ArgumentParser(
        description="Record a workflow demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record with browser + voice (recommended)
  python -m src.cli.record
  
  # Record with specific starting URL
  python -m src.cli.record --url https://google.com
  
  # Record desktop only (no browser)
  python -m src.cli.record --no-browser
  
  # Record without voice (not recommended)
  python -m src.cli.record --no-voice

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW: record → compile → replay
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: RECORD (this command)
  python -m src.cli.record
  
Step 2: COMPILE (creates reusable workflow)
  python -m src.cli.compile --session session_abc123 --name my_workflow --goals
  
Step 3: REPLAY (automate with new inputs)
  python -m src.cli.replay --recipe artifacts/recipes/my_workflow_goals.json \\
      --goals --params '{"search_topic": "new value"}'
        """
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.sessions_dir,
        help=f"Output directory (default: {config.sessions_dir})"
    )
    
    parser.add_argument(
        "--url",
        type=str,
        default=config.browser_default_url,
        help=f"Initial browser URL (default: {config.browser_default_url})"
    )
    
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't launch browser (desktop recording only)"
    )
    
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice recording"
    )
    
    args = parser.parse_args()
    
    # Print recording guide
    print(RECORDING_GUIDE)
    
    # Print permissions reminder
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║  ⚠️  REQUIRED PERMISSIONS (macOS)                                          ║")
    print("╠════════════════════════════════════════════════════════════════════════════╣")
    print("║  System Preferences > Security & Privacy:                                  ║")
    print("║    ✓ Screen Recording - to capture screenshots                             ║")
    print("║    ✓ Accessibility - to track mouse/keyboard                               ║")
    if not args.no_voice:
        print("║    ✓ Microphone - for voice narration                                      ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    print(f"  📍 Starting URL: {args.url}")
    print(f"  🎙️  Voice recording: {'✓ ENABLED' if not args.no_voice else '✗ Disabled'}")
    print(f"  🌐 Browser: {'✓ ENABLED' if not args.no_browser else '✗ Disabled'}")
    print()
    
    input("Press Enter to start recording (Ctrl+C to stop when done)...")
    print()
    print("🔴 RECORDING STARTED - Perform your workflow now...")
    
    try:
        recorder = SessionRecorder(
            output_dir=args.output_dir,
            use_browser=not args.no_browser,
            initial_url=args.url,
            enable_voice=not args.no_voice
        )
        
        session = recorder.start()
        
        # Print detailed next steps
        print("\n" + "═" * 70)
        print("  ✅ RECORDING COMPLETE!")
        print("═" * 70)
        print(f"\n  📁 Session saved: {session.session_id}")
        print(f"     Location: artifacts/sessions/{session.session_id}/")
        print(f"     Duration: {session.duration():.1f} seconds")
        print(f"     Events captured: {len(session.timeline)}")
        if session.voice_audio_path:
            print(f"     Voice audio: ✓ Recorded")
        
        print("\n" + "─" * 70)
        print("  📋 NEXT STEP: Compile your recording into a reusable workflow")
        print("─" * 70)
        print()
        print("  Copy and run this command:")
        print()
        print(f"  python -m src.cli.compile \\")
        print(f"      --session {session.session_id} \\")
        print(f"      --name <your_workflow_name> \\")
        print(f"      --goals")
        print()
        print("  Example:")
        print(f"  python -m src.cli.compile --session {session.session_id} --name restaurant_search --goals")
        print()
        print("─" * 70)
        print("  💡 The --goals flag creates a robust, goal-based workflow")
        print("  💡 Give it a descriptive name like 'restaurant_search' or 'job_apply'")
        print("═" * 70)
        print()
    
    except PermissionError as e:
        print(f"\n❌ Permission denied: {e}")
        print("Please grant the required permissions in System Preferences.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()