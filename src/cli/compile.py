#!/usr/bin/env python3
"""CLI for compiling sessions into workflow recipes."""
import argparse
import sys
import os
import json
import re
from pathlib import Path

# Add src to path if running as script
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(src_path))

from src.models.session_artifact import SessionArtifact
from src.compiler.workflow_compiler import WorkflowCompiler
from src.utils.config import config


def main():
    parser = argparse.ArgumentParser(
        description="Compile session recording into reusable workflow recipe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile session into recipe (recommended: LLM + Gemini)
  python -m src.cli.compile --session session_abc123 --name restaurant_search
  
  # With explicit LLM and Gemini flags
  python -m src.cli.compile --session session_abc123 --name restaurant_search --llm --gemini
  
  # Without Gemini (basic extraction)
  python -m src.cli.compile --session session_abc123 --name restaurant_search --no-gemini
  
  # Heuristics only (no LLM/Gemini)
  python -m src.cli.compile --session session_abc123 --name restaurant_search --no-llm --no-gemini
  
  # Specify output location
  python -m src.cli.compile --session session_abc123 --name my_workflow --output my_recipe.json
        """
    )
    
    parser.add_argument(
        "--session",
        required=True,
        help="Session ID to compile (from recording)"
    )
    
    parser.add_argument(
        "--name",
        required=True,
        help="Name for the workflow"
    )
    
    parser.add_argument(
        "--description",
        help="Optional workflow description"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: artifacts/recipes/<name>.json)"
    )
    
    parser.add_argument(
        "--llm",
        action="store_true",
        default=True,
        help="Use LLM (GPT-4o) for intent classification and parameter detection (default: enabled)"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM, use heuristics only"
    )
    
    parser.add_argument(
        "--gemini",
        action="store_true",
        default=True,
        help="Use Gemini for visual extraction analysis (default: enabled)"
    )
    
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Disable Gemini visual enrichment"
    )
    
    parser.add_argument(
        "--goals",
        action="store_true",
        help="Compile to goal-based workflow (new robust format)"
    )
    
    args = parser.parse_args()
    
    # Find session directory
    session_dir = config.sessions_dir / args.session
    
    if not session_dir.exists():
        # Try to find by prefix
        matches = list(config.sessions_dir.glob(f"{args.session}*"))
        if len(matches) == 1:
            session_dir = matches[0]
        elif len(matches) > 1:
            print(f"❌ Multiple sessions match '{args.session}':")
            for m in matches:
                print(f"   {m.name}")
            sys.exit(1)
        else:
            print(f"❌ Session not found: {args.session}")
            print(f"   Looking in: {config.sessions_dir}")
            
            # List available sessions
            sessions = list(config.sessions_dir.glob("session_*"))
            if sessions:
                print("\nAvailable sessions:")
                for s in sessions[:10]:
                    print(f"   {s.name}")
                if len(sessions) > 10:
                    print(f"   ... and {len(sessions) - 10} more")
            
            sys.exit(1)
    
    # Load session
    print(f"\nLoading session: {session_dir.name}")
    
    try:
        session = SessionArtifact.load(session_dir)
    except Exception as e:
        print(f"❌ Error loading session: {e}")
        sys.exit(1)
    
    print(f"  Duration: {session.duration():.1f}s")
    print(f"  Events: {len(session.timeline)}")
    print(f"  Copy events: {len(session.get_copy_events())}")
    if session.voice_audio_path:
        print(f"  Voice audio: {session.voice_audio_path}")
    if session.voice_transcription:
        print(f"  Voice (pre-transcribed): {len(session.voice_transcription.text)} chars")
    
    # Determine LLM usage
    use_llm = args.llm and not args.no_llm
    use_gemini = args.gemini and not args.no_gemini
    
    # Check API keys
    if use_llm and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  LLM enabled but OPENAI_API_KEY not set")
        print("   Set the environment variable or use --no-llm")
        print("   Falling back to heuristics...")
        use_llm = False
    
    if use_gemini and not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠️  Gemini enabled but GOOGLE_API_KEY not set")
        print("   Set the environment variable or use --no-gemini")
        print("   Extraction schemas will be basic...")
        use_gemini = False
    
    # Print config status
    print(f"\n📋 Compilation Settings:")
    print(f"   GPT-4o (text analysis): {'✓ Enabled' if use_llm else '✗ Disabled'}")
    print(f"   Gemini (visual analysis): {'✓ Enabled' if use_gemini else '✗ Disabled'}")
    
    # Create compiler
    compiler = WorkflowCompiler(use_llm=use_llm, use_gemini=use_gemini)
    
    # Compile
    print()
    
    try:
        if args.goals:
            # Compile to goal-based workflow (new format)
            print("   Mode: Goal-based workflow (new robust format)")
            workflow = compiler.compile_to_goals(
                session,
                workflow_name=args.name,
                description=args.description
            )
            
            # Save workflow
            if args.output:
                output_path = args.output
            else:
                output_path = config.recipes_dir / f"{args.name}_goals.json"
            
            workflow.save(output_path)
            
            # Print summary
            print(f"\n✅ Goal workflow saved to: {output_path}")
            
            print(f"\n{'═' * 60}")
            print(f"  📊 GOAL WORKFLOW SUMMARY")
            print(f"{'═' * 60}")
            print(f"  Goals: {len(workflow.steps)}")
            print(f"  Voice analyzed: {'✓' if workflow.voice_analyzed else '✗'}")
            print(f"  Gemini enriched: {'✓' if workflow.gemini_enriched else '✗'}")
            
            # Show goals with types
            print(f"\n  🎯 GOALS (execution order):")
            print(f"  {'─' * 56}")
            for i, goal in enumerate(workflow.steps):
                platform_icon = "🌐" if goal.platform == "browser" else "🖥️"
                print(f"  {i+1}. {platform_icon} [{goal.goal_type.value:8}] {goal.goal_description[:40]}")
            
            # Show parameters with where they're used
            if workflow.parameters:
                print(f"\n  📝 PARAMETERS:")
                print(f"  {'─' * 56}")
                for param_name, default_value in workflow.parameters.items():
                    # Find where it's used
                    used_in = []
                    pattern = f"{{{{\\s*{re.escape(param_name)}\\s*}}}}"
                    for i, step in enumerate(workflow.steps, 1):
                        if step.template and re.search(pattern, step.template):
                            used_in.append(f"Step {i}")
                        for strat in step.strategies:
                            if strat.input_value and re.search(pattern, strat.input_value):
                                if f"Step {i}" not in used_in:
                                    used_in.append(f"Step {i}")
                    
                    usage = f"→ Used in: {', '.join(used_in)}" if used_in else "⚠️  NOT USED"
                    print(f"  • {param_name}")
                    print(f"    Default: \"{default_value[:40]}{'...' if len(str(default_value)) > 40 else ''}\"")
                    print(f"    {usage}")
            else:
                print(f"\n  📝 PARAMETERS: (none detected)")
            
            # Show copy-pasteable replay command
            print(f"\n{'═' * 60}")
            print(f"  🚀 TO REPLAY THIS WORKFLOW:")
            print(f"{'═' * 60}")
            
            if workflow.parameters:
                # Template command
                param_template = {k: f"<your_{k}>" for k in workflow.parameters.keys()}
                print(f"\n  📋 With your values:")
                print(f"  python -m src.cli.replay \\")
                print(f"      --recipe {output_path} \\")
                print(f"      --goals \\")
                print(f"      --params '{json.dumps(param_template)}'")
                
                # Example with defaults
                print(f"\n  📋 Or test with recorded defaults:")
                print(f"  python -m src.cli.replay \\")
                print(f"      --recipe {output_path} \\")
                print(f"      --goals \\")
                print(f"      --params '{json.dumps(workflow.parameters)}'")
            else:
                print(f"\n  python -m src.cli.replay --recipe {output_path} --goals")
            
            print()
        
        else:
            # Traditional recipe compilation
            recipe = compiler.compile(
                session,
                workflow_name=args.name,
                description=args.description
            )
            
            # Save recipe
            if args.output:
                output_path = args.output
            else:
                output_path = config.recipes_dir / f"{args.name}.json"
            
            recipe.save(output_path)
            
            # Print summary and usage instructions
            print(f"\n✓ Recipe saved to: {output_path}")
            
            print(f"\n📊 Recipe Summary:")
            print(f"   Steps: {len(recipe.steps)}")
            print(f"   Parameters: {list(recipe.parameters.keys())}")
            print(f"   Extraction fields: {recipe.get_extraction_fields()}")
            print(f"   Voice analyzed: {'✓' if recipe.voice_analyzed else '✗'}")
            print(f"   Gemini enriched: {'✓' if recipe.gemini_enriched else '✗'}")
            
            print("\n🚀 To replay this workflow:")
            
            if recipe.parameters:
                param_example = ", ".join([
                    f'"{k}": "your_value"' 
                    for k in recipe.parameters.keys()
                ])
                print(f'  python -m src.cli.replay --recipe {output_path} \\')
                print(f"      --params '{{{param_example}}}'")
            else:
                print(f'  python -m src.cli.replay --recipe {output_path}')
            
            print("\n💡 TIP: Use --goals flag for the new goal-based format:")
            print(f'  python -m src.cli.compile --session {args.session} --name {args.name} --goals')
        
    except Exception as e:
        print(f"\n❌ Compilation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()