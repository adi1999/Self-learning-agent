"""Main workflow executor - orchestrates replay with extracted data flow."""
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path

from src.models.workflow_recipe import WorkflowRecipe, WorkflowStep, FailurePolicy
from src.models.goal_step import GoalWorkflow
from src.executor.browser_executor import BrowserExecutor, StepResult
from src.executor.desktop_executor import DesktopExecutor, DesktopStepResult
from src.executor.goal_executor import GoalExecutor, WorkflowResult as GoalWorkflowResult
from src.executor.app_launcher import AppLauncher
from src.utils.logger import setup_logger, StepLogger
from src.utils.config import config


@dataclass
class ExecutionResult:
    """Result of workflow execution."""
    success: bool
    steps_executed: int = 0
    steps_failed: int = 0
    total_steps: int = 0
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0


class WorkflowExecutor:
    """
    Executes workflow recipes with parameter substitution and data flow.
    
    Key features:
    - Gemini-powered extraction
    - Extracted data flows to template filling
    - Fallback strategies for robust execution
    """
    
    def __init__(self, headless: bool = False, validate_steps: bool = True):
        """
        Initialize executor.
        
        Args:
            headless: Run browser in headless mode
            validate_steps: Validate step completion
        """
        self.headless = headless
        self.validate_steps = validate_steps
        self.logger = setup_logger("WorkflowExecutor")
        
        # Executors
        self.browser_executor: Optional[BrowserExecutor] = None
        self.desktop_executor: Optional[DesktopExecutor] = None
        self.app_launcher = AppLauncher()
        
        # State
        self._current_platform: Optional[str] = None
        
        # =====================================================================
        # EXTRACTED DATA STORE
        # =====================================================================
        # This is populated by extract steps and consumed by write steps
        self._extracted_data: Dict[str, Any] = {}
    
    def execute(
        self,
        recipe: WorkflowRecipe,
        parameters: Optional[Dict[str, Any]] = None,
        initial_url: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a workflow recipe.
        
        Args:
            recipe: Workflow recipe to execute
            parameters: Parameter values to substitute
            initial_url: Initial URL for browser
        
        Returns:
            ExecutionResult with status and extracted data
        """
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info(f"Executing Workflow: {recipe.name}")
        self.logger.info("=" * 60)
        
        # Reset extracted data
        self._extracted_data = {}
        
        # Validate and substitute parameters
        if parameters:
            self.logger.info(f"\nValidating {len(parameters)} parameters...")
            errors = recipe.validate_parameters(parameters)
            if errors:
                self.logger.error("Parameter validation failed:")
                for error in errors:
                    self.logger.error(f"  • {error}")
                return ExecutionResult(
                    success=False,
                    total_steps=len(recipe.steps),
                    errors=errors
                )
            
            recipe = recipe.substitute_parameters(parameters)
            
            self.logger.info("Parameters:")
            for key, value in parameters.items():
                self.logger.info(f"  • {key} = {value}")
        
        self.logger.info(f"\nSteps: {len(recipe.steps)}")
        
        if recipe.gemini_enriched:
            self.logger.info("Recipe includes Gemini-enriched extraction schemas")
        
        extraction_fields = recipe.get_extraction_fields()
        if extraction_fields:
            self.logger.info(f"Extraction fields: {extraction_fields}")
        
        self.logger.info("=" * 60)

        # Initialize result
        result = ExecutionResult(
            success=True,
            total_steps=len(recipe.steps)
        )
        
        try:
            # Initialize executors as needed
            self._initialize_executors(recipe, initial_url)
            
            # Execute steps
            for i, step in enumerate(recipe.steps):
                step_start = time.time()
                
                self.logger.info(f"\n[Step {i+1}/{len(recipe.steps)}] {step.description}")
                self.logger.info(f"  Intent: {step.intent} | Platform: {step.platform} | App: {step.app_name}")
                
                # =====================================================================
                # HANDLE WRITE STEPS WITH TEMPLATE + EXTRACTED DATA
                # =====================================================================
                if step.intent == "write" and step.template:
                    # Fill template with extracted data
                    filled_content = recipe.fill_template(step.template, self._extracted_data)
                    step.parameter_bindings["value"] = filled_content
                    self.logger.info(f"  Template filled with {len(self._extracted_data)} fields")
                    self.logger.debug(f"  Content: {filled_content[:100]}...")
                
                # Execute with retry
                step_result = self._execute_step_with_retry(step, recipe.failure_policy)
                
                step_duration = time.time() - step_start
                
                # Record result
                result.step_results.append({
                    "step_id": step.step_id,
                    "step_number": step.step_number,
                    "intent": step.intent,
                    "success": step_result.get("success", False),
                    "duration": step_duration,
                    "strategy": step_result.get("strategy"),
                    "error": step_result.get("error"),
                    "extracted_data": step_result.get("extracted_data", {})
                })
                
                if step_result.get("success"):
                    result.steps_executed += 1
                    self.logger.info(f"  ✓ Success ({step_duration:.2f}s)")
                    
                    # =====================================================================
                    # COLLECT EXTRACTED DATA FROM EXTRACT STEPS
                    # =====================================================================
                    if step_result.get("extracted_data"):
                        new_data = step_result["extracted_data"]
                        self._extracted_data.update(new_data)
                        self.logger.info(f"  ✓ Extracted {len(new_data)} fields (total: {len(self._extracted_data)})")
                else:
                    result.steps_failed += 1
                    error_msg = step_result.get("error", "Unknown error")
                    result.errors.append(f"Step {i+1}: {error_msg}")
                    self.logger.error(f"  ✗ Failed: {error_msg}")
                    
                    # Check failure policy
                    if recipe.failure_policy.on_failure in ["abort", "retry_then_abort"]:
                        self.logger.error("Aborting workflow due to failure")
                        result.success = False
                        break
        
        except KeyboardInterrupt:
            self.logger.warning("\nWorkflow interrupted by user")
            result.success = False
            result.errors.append("Interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Workflow execution error: {e}")
            result.success = False
            result.errors.append(str(e))
        
        finally:
            self._cleanup()
        
        # Finalize result
        result.extracted_data = self._extracted_data
        result.duration_seconds = time.time() - start_time
        
        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Execution Complete")
        self.logger.info("=" * 60)
        self.logger.info(f"Success: {result.success}")
        self.logger.info(f"Steps: {result.steps_executed}/{result.total_steps} executed")
        self.logger.info(f"Duration: {result.duration_seconds:.1f}s")
        
        if result.extracted_data:
            self.logger.info("\nExtracted Data:")
            for key, value in result.extracted_data.items():
                preview = str(value)[:50] + "..." if len(str(value)) > 50 else value
                self.logger.info(f"  • {key}: {preview}")
        
        if result.errors:
            self.logger.info(f"\nErrors ({len(result.errors)}):")
            for error in result.errors[:5]:
                self.logger.info(f"  • {error}")
        
        self.logger.info("=" * 60)
        
        return result
    
    def _initialize_executors(self, recipe: WorkflowRecipe, initial_url: Optional[str]):
        """Initialize executors based on workflow needs."""
        
        # Check if we need browser
        has_browser_steps = any(s.platform == "browser" for s in recipe.steps)
        
        if has_browser_steps:
            self.browser_executor = BrowserExecutor()
            url = initial_url or "https://www.google.com"
            self.browser_executor.launch(url=url, headless=self.headless)
        
        # Desktop executor is always available
        self.desktop_executor = DesktopExecutor()
    
    def _execute_step_with_retry(
        self,
        step: WorkflowStep,
        policy: FailurePolicy
    ) -> Dict[str, Any]:
        """Execute step with retry logic."""
        
        last_error = None
        
        for attempt in range(policy.retry_limit + 1):
            if attempt > 0:
                self.logger.info(f"  Retry attempt {attempt}/{policy.retry_limit}")
                time.sleep(policy.retry_delay_ms / 1000)
            
            try:
                result = self._execute_step(step, policy.use_gemini_fallback)
                
                if result.get("success"):
                    return result
                
                last_error = result.get("error")
            
            except Exception as e:
                last_error = str(e)
                self.logger.debug(f"  Attempt {attempt + 1} failed: {e}")
        
        return {"success": False, "error": last_error}
    
    def _execute_step(self, step: WorkflowStep, use_gemini_fallback: bool) -> Dict[str, Any]:
        """Execute a single step."""
        
        # Handle platform switch
        if step.platform != self._current_platform:
            self._switch_platform(step)
        
        # =====================================================================
        # HANDLE PASTE STEPS WITH TEMPLATE (fill from extracted data)
        # =====================================================================
        if step.shortcut == "paste" and step.template:
            # Fill template with extracted data
            filled_content = self._fill_paste_template(step.template)
            step.clipboard_content = filled_content
            self.logger.info(f"  Paste template filled: {step.template} → {filled_content[:50]}...")
        
        # Execute based on platform
        if step.platform == "browser":
            if not self.browser_executor:
                return {"success": False, "error": "Browser not initialized"}
            
            result = self.browser_executor.execute_step(step, use_gemini_fallback)
            
            return {
                "success": result.success,
                "error": result.error,
                "strategy": result.strategy_used,
                "extracted_data": result.extracted_data
            }
        
        else:  # desktop
            if not self.desktop_executor:
                return {"success": False, "error": "Desktop executor not initialized"}
            
            # For write steps with template, pass the filled value
            if step.intent == "write" and "value" in step.parameter_bindings:
                # Value already filled in execute() before calling this
                pass
            
            result = self.desktop_executor.execute_step(step)
            
            return {
                "success": result.success,
                "error": result.error,
                "strategy": result.strategy_used
            }
    
    def _switch_platform(self, step: WorkflowStep):
        """Handle switching between browser and desktop."""
        self.logger.debug(f"Switching platform: {self._current_platform} → {step.platform}")
        
        if step.platform == "desktop":
            # Ensure desktop app is active
            self.desktop_executor.ensure_app_active(step.app_name)
        
        self._current_platform = step.platform
    
    def _cleanup(self):
        """Clean up resources."""
        if self.browser_executor:
            self.browser_executor.close()
            self.browser_executor = None
        
        self.desktop_executor = None

    def _fill_paste_template(self, template: str) -> str:
        """Fill a paste template with extracted data."""
        import re
        
        result = template
        placeholders = re.findall(r'\{\{(\w+)\}\}', template)
        
        for field_name in placeholders:
            if field_name in self._extracted_data:
                value = str(self._extracted_data[field_name])
                result = result.replace(f"{{{{{field_name}}}}}", value)
            else:
                self.logger.warning(f"  Field '{field_name}' not found in extracted data")
                self.logger.warning(f"  Available: {list(self._extracted_data.keys())}")
        
        return result
    
    # =========================================================================
    # GOAL-BASED EXECUTION (New approach)
    # =========================================================================
    
    def execute_goal_workflow(
        self,
        workflow: GoalWorkflow,
        parameters: Optional[Dict[str, Any]] = None,
        initial_url: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a goal-based workflow.
        
        This uses the new GoalExecutor which tries multiple strategies
        and validates success by criteria, not just action completion.
        
        Args:
            workflow: GoalWorkflow to execute
            parameters: Parameter values to substitute
            initial_url: Initial URL for browser
        
        Returns:
            ExecutionResult with status and extracted data
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Executing GOAL Workflow: {workflow.name}")
        self.logger.info("=" * 60)
        
        # Check if we need browser
        has_browser_steps = any(s.platform == "browser" for s in workflow.steps)
        
        if has_browser_steps:
            # Initialize browser
            self.browser_executor = BrowserExecutor()
            url = initial_url or "https://www.google.com"
            self.browser_executor.launch(url=url, headless=self.headless)
        
        # Initialize desktop executor
        self.desktop_executor = DesktopExecutor()
        
        # Create GoalExecutor
        goal_executor = GoalExecutor(
            browser_page=self.browser_executor.page if self.browser_executor else None,
            desktop_executor=self.desktop_executor,
            app_launcher=self.app_launcher
        )
        
        try:
            # Execute the goal workflow
            goal_result = goal_executor.execute_workflow(workflow, parameters)
            
            # Convert to ExecutionResult
            result = ExecutionResult(
                success=goal_result.success,
                steps_executed=goal_result.steps_executed,
                steps_failed=goal_result.steps_failed,
                total_steps=goal_result.total_steps,
                extracted_data=goal_result.extracted_data,
                errors=goal_result.errors,
                duration_seconds=goal_result.duration_seconds
            )
            
            # Convert step results
            for i, goal_res in enumerate(goal_result.step_results):
                result.step_results.append({
                    "step_number": i + 1,
                    "success": goal_res.achieved,
                    "strategy": goal_res.strategy_used,
                    "error": goal_res.error,
                    "extracted_data": goal_res.extracted_data,
                    "duration": goal_res.duration_seconds
                })
            
            return result
        
        finally:
            self._cleanup()
    
    def execute_any(
        self,
        workflow: Union[WorkflowRecipe, GoalWorkflow],
        parameters: Optional[Dict[str, Any]] = None,
        initial_url: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute either a WorkflowRecipe or GoalWorkflow.
        
        Automatically detects the type and uses the appropriate executor.
        """
        if isinstance(workflow, GoalWorkflow):
            return self.execute_goal_workflow(workflow, parameters, initial_url)
        else:
            return self.execute(workflow, parameters, initial_url)