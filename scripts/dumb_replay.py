"""Phase 1: Dumb replayer using coordinate-based replay.

This is a simple replay script that uses pyautogui to replay
recorded clicks and keyboard events exactly as captured.

WARNING: This is very brittle and will break if:
- Window positions change
- Screen resolution changes  
- UI elements move

This is intentional for Phase 1 - it teaches us the problems
we need to solve in Phase 2+.
"""
import sys
from pathlib import Path
import time
import pyautogui

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.session_artifact import SessionArtifact
from src.utils.config import config


class DumbReplayer:
    """Simple coordinate-based replayer for Phase 1."""
    
    def __init__(self, session: SessionArtifact):
        self.session = session
        self.start_time = None
        
        # Safety settings
        pyautogui.PAUSE = 0.1  # Small pause between actions
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    
    def replay(self, speed_multiplier: float = 1.0):
        """
        Replay the session.
        
        Args:
            speed_multiplier: Speed up (>1.0) or slow down (<1.0) replay
        """
        print("=" * 50)
        print("DUMB REPLAY - Phase 1")
        print("=" * 50)
        print("This will replay clicks and typing at the same coordinates.")
        print("WARNING: Move mouse to top-left corner to abort!")
        print()
        print(f"Timeline events: {len(self.session.timeline)}")
        print(f"Duration: {self.session.duration():.1f}s")
        print(f"Speed: {speed_multiplier}x")
        print("=" * 50)
        
        input("Press Enter to start replay...")
        print()
        
        self.start_time = time.time()
        last_timestamp = 0.0
        
        for event in self.session.timeline:
            # Wait for correct timing
            target_time = event.timestamp / speed_multiplier
            current_time = time.time() - self.start_time
            wait_time = target_time - current_time
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Log event
            elapsed = time.time() - self.start_time
            print(f"[{elapsed:.1f}s] {event.active_app}: {event.window_title}")
            
            # Replay input events
            for input_event in event.input_events:
                if input_event.type == "mouse_click":
                    x, y = input_event.x, input_event.y
                    print(f"  → Click at ({x}, {y})")
                    pyautogui.click(x, y)
                
                elif input_event.type == "keyboard" and input_event.text:
                    text = input_event.text
                    print(f"  → Type: {text[:50]}...")
                    pyautogui.write(text, interval=0.05)
                
                elif input_event.type == "keyboard" and input_event.key == "return":
                    print(f"  → Press: Enter")
                    pyautogui.press('return')
            
            last_timestamp = event.timestamp
        
        print()
        print("=" * 50)
        print("Replay complete!")
        print("=" * 50)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dumb replay for Phase 1")
    parser.add_argument(
        "--session",
        required=True,
        help="Session ID to replay"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Load session
    session_dir = config.sessions_dir / args.session
    if not session_dir.exists():
        print(f"Error: Session not found: {session_dir}")
        sys.exit(1)
    
    try:
        session = SessionArtifact.load(session_dir)
    except Exception as e:
        print(f"Error loading session: {e}")
        sys.exit(1)
    
    # Replay
    replayer = DumbReplayer(session)
    replayer.replay(speed_multiplier=args.speed)


if __name__ == "__main__":
    main()