#!/usr/bin/env python3
"""CLI entry point for replaying workflows."""
import argparse
import sys
import json
from pathlib import Path

# Add src to path if running as script
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(src_path))

from src.models.workflow_recipe import WorkflowRecipe
from src.models.goal_step import GoalWorkflow
from src.executor.workflow_executor import WorkflowExecutor
from src.utils.config import config
import re


def _find_parameter_usage(workflow_or_recipe, param_name: str, is_goal: bool) -> list:
    """Find where a parameter is used in the workflow."""
    usages = []
    pattern = f"{{{{\\s*{re.escape(param_name)}\\s*}}}}"
    
    if is_goal:
        for i, step in enumerate(workflow_or_recipe.steps, 1):
            locations = []
            
            # Check goal description
            if step.goal_description and re.search(pattern, step.goal_description):
                locations.append("goal description")
            
            # Check template
            if step.template and re.search(pattern, step.template):
                locations.append("template")
            
            # Check strategies
            for strat in step.strategies:
                if strat.input_value and re.search(pattern, strat.input_value):
                    locations.append(f"strategy '{strat.name}'")
            
            # Check agent prompt
            if step.agent_goal_prompt and re.search(pattern, step.agent_goal_prompt):
                locations.append("agent fallback prompt")
            
            if locations:
                usages.append((i, step.goal_type.value, locations))
    else:
        for i, step in enumerate(workflow_or_recipe.steps, 1):
            locations = []
            
            # Check bindings
            for k, v in step.parameter_bindings.items():
                if isinstance(v, str) and re.search(pattern, v):
                    locations.append(f"binding '{k}'")
            
            if locations:
                usages.append((i, step.action_type, locations))
    
    return usages


def _print_parameter_info(workflow_or_recipe, is_goal: bool):
    """Print detailed parameter information with usage locations."""
    params = workflow_or_recipe.parameters
    
    if not params:
        print("  Parameters: (none)")
        return
    
    print("\n  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║  PARAMETERS                                                   ║")
    print("  ╠═══════════════════════════════════════════════════════════════╣")
    
    for param_name, default_value in params.items():
        # Find where this param is used
        usages = _find_parameter_usage(workflow_or_recipe, param_name, is_goal)
        
        print(f"  ║                                                               ║")
        print(f"  ║  📌 {param_name:<57} ║")
        print(f"  ║     Default: \"{default_value[:45]}{'...' if len(str(default_value)) > 45 else ''}\"")
        
        if usages:
            print(f"  ║     Used in:")
            for step_num, step_type, locs in usages:
                loc_str = ", ".join(locs)
                print(f"  ║       → Step {step_num} ({step_type}): {loc_str}")
        else:
            print(f"  ║     ⚠️  NOT USED in any step (may be metadata only)")
    
    print("  ║                                                               ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    
    # Generate example command
    recipe_path = getattr(workflow_or_recipe, '_source_path', 'your_recipe.json')
    example_params = {k: f"<your_{k}>" for k in params.keys()}
    
    print("\n  💡 EXAMPLE COMMAND:")
    print("  ─────────────────────────────────────────────────────────────────")
    print(f"  python -m src.cli.replay \\")
    print(f"      --recipe {recipe_path} \\")
    if is_goal:
        print(f"      --goals \\")
    print(f"      --params '{json.dumps(example_params)}'")
    print()
    
    # Show a more concrete example with defaults
    print("  📋 COPY-PASTE EXAMPLE (with defaults):")
    print("  ─────────────────────────────────────────────────────────────────")
    print(f"  python -m src.cli.replay \\")
    print(f"      --recipe {recipe_path} \\")
    if is_goal:
        print(f"      --goals \\")
    print(f"      --params '{json.dumps(params)}'")
    print()


def main():
    """Replay a workflow with new parameters."""
    parser = argparse.ArgumentParser(
        description="Replay a compiled workflow with new parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay with parameters as JSON string
  python -m src.cli.replay --recipe artifacts/recipes/restaurant_search.json \\
      --params '{"cuisine": "pizza", "location": "san jose"}'
  
  # Replay with parameters from file
  python -m src.cli.replay --recipe my_workflow.json --params-file params.json
  
  # Replay in headless mode (no visible browser)
  python -m src.cli.replay --recipe my_workflow.json --params '{}' --headless
  
  # Dry run (show steps without executing)
  python -m src.cli.replay --recipe my_workflow.json --dry-run
        """
    )
    
    parser.add_argument(
        "--recipe",
        required=True,
        type=Path,
        help="Path to workflow recipe JSON file"
    )
    
    parser.add_argument(
        "--params",
        type=str,
        help="Parameter values as JSON string"
    )
    
    parser.add_argument(
        "--params-file",
        type=Path,
        help="Path to JSON file containing parameters"
    )
    
    parser.add_argument(
        "--url",
        type=str,
        help="Override initial browser URL"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip step validation"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show steps without executing"
    )
    
    parser.add_argument(
        "--goals",
        action="store_true",
        help="Load as goal-based workflow (auto-detected if file ends with _goals.json)"
    )
    
    args = parser.parse_args()
    
    # Load recipe
    if not args.recipe.exists():
        # Try recipes directory
        recipe_path = config.recipes_dir / args.recipe.name
        if not recipe_path.exists():
            print(f"❌ Recipe not found: {args.recipe}")
            sys.exit(1)
        args.recipe = recipe_path
    
    print(f"\nLoading recipe: {args.recipe}")
    
    # Auto-detect goal workflow by filename
    is_goal_workflow = args.goals or "_goals.json" in str(args.recipe)
    
    try:
        if is_goal_workflow:
            workflow = GoalWorkflow.load(args.recipe)
            workflow._source_path = str(args.recipe)  # Store for example command
            print(f"  Type: Goal-based workflow ✨")
            print(f"  Name: {workflow.name}")
            print(f"  Goals: {len(workflow.steps)}")
            
            # Show detailed parameter info
            _print_parameter_info(workflow, is_goal=True)
        else:
            recipe = WorkflowRecipe.load(args.recipe)
            recipe._source_path = str(args.recipe)  # Store for example command
            print(f"  Type: Traditional recipe")
            print(f"  Name: {recipe.name}")
            print(f"  Steps: {len(recipe.steps)}")
            
            # Show detailed parameter info
            _print_parameter_info(recipe, is_goal=False)
    except Exception as e:
        print(f"❌ Error loading recipe: {e}")
        sys.exit(1)
    
    # Load parameters
    parameters = {}
    
    if args.params:
        try:
            parameters = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in --params: {e}")
            sys.exit(1)
    
    if args.params_file:
        try:
            with open(args.params_file) as f:
                file_params = json.load(f)
                parameters.update(file_params)
        except Exception as e:
            print(f"❌ Error reading params file: {e}")
            sys.exit(1)
    
    # Validate required parameters
    if is_goal_workflow:
        required = workflow.get_required_parameters()
        missing = [p for p in required if p not in parameters]
        
        if missing and not args.dry_run:
            print(f"\n❌ Missing required parameters: {missing}")
            print("\nRequired parameters:")
            for name in required:
                print(f"  • {name}")
                print(f"    Example: {workflow.parameters.get(name, 'N/A')}")
            sys.exit(1)
    else:
        required = recipe.get_required_parameters()
        missing = [p for p in required if p not in parameters]
        
        if missing and not args.dry_run:
            print(f"\n❌ Missing required parameters: {missing}")
            print("\nRequired parameters:")
            for name in required:
                param = recipe.parameters[name]
                print(f"  • {name}: {param.description or 'No description'}")
                print(f"    Example: {param.example_value}")
            sys.exit(1)
    
    # Dry run: just show steps
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Steps that would be executed:")
        print("=" * 60)
        
        if is_goal_workflow:
            # Show goals
            if parameters:
                display_workflow = workflow.substitute_parameters(parameters)
            else:
                display_workflow = workflow
            
            for i, goal in enumerate(display_workflow.steps):
                print(f"\n[Goal {i+1}] {goal.goal_type.value.upper()}")
                print(f"  Description: {goal.goal_description}")
                print(f"  Platform: {goal.platform}")
                print(f"  App: {goal.app_name}")
                print(f"  Strategies: {len(goal.strategies)}")
                if goal.strategies:
                    best = goal.get_best_strategy()
                    print(f"  Best strategy: {best.name} (priority {best.priority})")
                print(f"  Success criteria:")
                if goal.success_criteria.url_contains:
                    print(f"    - URL contains: {goal.success_criteria.url_contains}")
                if goal.success_criteria.url_changed:
                    print(f"    - URL must change")
                if goal.success_criteria.page_type:
                    print(f"    - Page type: {goal.success_criteria.page_type}")
                if goal.success_criteria.timeout_success:
                    print(f"    - Timeout success")
        else:
            # Show traditional steps
            if parameters:
                display_recipe = recipe.substitute_parameters(parameters)
            else:
                display_recipe = recipe
            
            for i, step in enumerate(display_recipe.steps):
                print(f"\n[Step {i+1}] {step.intent.upper()}")
                print(f"  Platform: {step.platform}")
                print(f"  App: {step.app_name}")
                print(f"  Action: {step.action_type}")
                
                if step.parameter_bindings:
                    print(f"  Value: {step.parameter_bindings.get('value', 'N/A')}")
                
                if step.element_reference:
                    print(f"  Target: {step.element_reference.get_description()}")
        
        print("\n" + "=" * 60)
        return
    
    # Confirm before execution
    print("\n" + "=" * 60)
    if is_goal_workflow:
        print("Ready to execute GOAL workflow ✨")
    else:
        print("Ready to execute workflow")
    print("=" * 60)
    
    if parameters:
        print("\nWith parameters:")
        for k, v in parameters.items():
            print(f"  • {k} = {v}")
    
    print()
    response = input("Press Enter to start (or 'q' to quit): ")
    
    if response.lower() == 'q':
        print("Cancelled.")
        sys.exit(0)
    
    # Execute
    try:
        executor = WorkflowExecutor(
            headless=args.headless,
            validate_steps=not args.no_validate
        )
        
        if is_goal_workflow:
            # Execute goal-based workflow
            result = executor.execute_goal_workflow(
                workflow=workflow,
                parameters=parameters,
                initial_url=args.url
            )
        else:
            # Execute traditional recipe
            result = executor.execute(
                recipe=recipe,
                parameters=parameters,
                initial_url=args.url
            )
        
        # Exit code based on success
        sys.exit(0 if result.success else 1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Execution error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()