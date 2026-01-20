"""Audit logging for workflow executions.

Provides a persistent record of all workflow executions for:
- Debugging failed workflows
- Security auditing
- Performance analysis
- Compliance requirements
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
import threading

from src.utils.logger import setup_logger


@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: str
    workflow_id: str
    step_id: str
    step_number: int
    action_type: str
    goal_type: Optional[str] = None
    platform: Optional[str] = None
    app_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: str = "pending"  # pending, success, failed, blocked, skipped
    strategy_used: Optional[str] = None
    duration_ms: int = 0
    error: Optional[str] = None
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    safety_check: Optional[str] = None  # Result of safety check if applicable


@dataclass
class ExecutionSummary:
    """Summary of a workflow execution."""
    workflow_id: str
    workflow_name: str
    start_time: str
    end_time: Optional[str] = None
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    blocked_steps: int = 0
    skipped_steps: int = 0
    total_duration_ms: int = 0
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    final_status: str = "running"  # running, completed, failed, aborted


class AuditLog:
    """
    Thread-safe audit logger for workflow executions.
    
    Creates JSONL (JSON Lines) files for easy parsing and streaming.
    
    Usage:
        audit = AuditLog()
        audit.start_execution("my_workflow", "My Workflow", {"param": "value"})
        
        audit.log_step(AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            workflow_id="my_workflow",
            step_id="step_1",
            step_number=1,
            action_type="click",
            result="success"
        ))
        
        audit.end_execution(success=True)
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize audit logger.
        
        Args:
            log_dir: Directory to store audit logs. 
                    Defaults to artifacts/audit_logs/
        """
        self.logger = setup_logger("AuditLog")
        self.log_dir = log_dir or Path("artifacts/audit_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_log_path: Optional[Path] = None
        self._current_summary: Optional[ExecutionSummary] = None
        self._entries: List[AuditEntry] = []
        self._lock = threading.Lock()
        
        self.logger.info(f"AuditLog initialized: {self.log_dir}")
    
    def start_execution(
        self,
        workflow_id: str,
        workflow_name: str,
        parameters: Dict[str, Any] = None
    ) -> Path:
        """
        Start logging a new workflow execution.
        
        Args:
            workflow_id: Unique identifier for the workflow
            workflow_name: Human-readable name
            parameters: Parameters being used for this execution
        
        Returns:
            Path to the log file
        """
        with self._lock:
            timestamp = datetime.utcnow()
            filename = f"{workflow_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jsonl"
            self._current_log_path = self.log_dir / filename
            
            self._current_summary = ExecutionSummary(
                workflow_id=workflow_id,
                workflow_name=workflow_name,
                start_time=timestamp.isoformat(),
                parameters_used=parameters or {}
            )
            
            self._entries = []
            
            # Write header entry
            header = {
                "type": "execution_start",
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "start_time": timestamp.isoformat(),
                "parameters": parameters or {}
            }
            self._write_line(header)
            
            self.logger.info(f"Started audit log: {self._current_log_path}")
            return self._current_log_path
    
    def log_step(self, entry: AuditEntry):
        """
        Log a workflow step execution.
        
        Args:
            entry: AuditEntry with step details
        """
        with self._lock:
            if not self._current_log_path:
                self.logger.warning("No active execution - call start_execution first")
                return
            
            self._entries.append(entry)
            
            # Update summary counts
            if self._current_summary:
                self._current_summary.total_steps += 1
                if entry.result == "success":
                    self._current_summary.successful_steps += 1
                elif entry.result == "failed":
                    self._current_summary.failed_steps += 1
                elif entry.result == "blocked":
                    self._current_summary.blocked_steps += 1
                elif entry.result == "skipped":
                    self._current_summary.skipped_steps += 1
                
                self._current_summary.total_duration_ms += entry.duration_ms
                
                if entry.extracted_data:
                    self._current_summary.extracted_data.update(entry.extracted_data)
            
            # Write to file
            log_entry = {
                "type": "step",
                **asdict(entry)
            }
            self._write_line(log_entry)
    
    def log_safety_block(
        self,
        step_id: str,
        step_number: int,
        action_type: str,
        reason: str,
        blocked_content: str = ""
    ):
        """
        Log when a step is blocked by safety guard.
        
        Args:
            step_id: ID of the blocked step
            step_number: Number of the step
            action_type: Type of action that was blocked
            reason: Why it was blocked
            blocked_content: The content that triggered the block
        """
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            workflow_id=self._current_summary.workflow_id if self._current_summary else "unknown",
            step_id=step_id,
            step_number=step_number,
            action_type=action_type,
            result="blocked",
            error=reason,
            safety_check=f"BLOCKED: {blocked_content[:100]}" if blocked_content else "BLOCKED"
        )
        self.log_step(entry)
    
    def end_execution(
        self,
        success: bool,
        error: Optional[str] = None,
        extracted_data: Dict[str, Any] = None
    ):
        """
        End the current workflow execution logging.
        
        Args:
            success: Whether the workflow completed successfully
            error: Error message if failed
            extracted_data: Final extracted data from the workflow
        """
        with self._lock:
            if not self._current_log_path or not self._current_summary:
                self.logger.warning("No active execution to end")
                return
            
            end_time = datetime.utcnow()
            self._current_summary.end_time = end_time.isoformat()
            self._current_summary.final_status = "completed" if success else "failed"
            
            if extracted_data:
                self._current_summary.extracted_data.update(extracted_data)
            
            # Write footer entry
            footer = {
                "type": "execution_end",
                **asdict(self._current_summary),
                "error": error
            }
            self._write_line(footer)
            
            self.logger.info(
                f"Ended audit log: {self._current_summary.final_status} "
                f"({self._current_summary.successful_steps}/{self._current_summary.total_steps} steps)"
            )
            
            # Reset state
            self._current_log_path = None
            self._current_summary = None
            self._entries = []
    
    def _write_line(self, data: Dict[str, Any]):
        """Write a JSON line to the log file."""
        if not self._current_log_path:
            return
        
        try:
            with open(self._current_log_path, 'a') as f:
                f.write(json.dumps(data, default=str) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")
    
    def get_current_summary(self) -> Optional[ExecutionSummary]:
        """Get the current execution summary."""
        with self._lock:
            return self._current_summary
    
    def get_recent_logs(self, limit: int = 10) -> List[Path]:
        """
        Get paths to recent audit logs.
        
        Args:
            limit: Maximum number of logs to return
        
        Returns:
            List of paths sorted by modification time (newest first)
        """
        logs = sorted(
            self.log_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return logs[:limit]
    
    @staticmethod
    def load_log(log_path: Path) -> List[Dict[str, Any]]:
        """
        Load and parse an audit log file.
        
        Args:
            log_path: Path to the log file
        
        Returns:
            List of log entries
        """
        entries = []
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
    
    @contextmanager
    def execution_context(
        self,
        workflow_id: str,
        workflow_name: str,
        parameters: Dict[str, Any] = None
    ):
        """
        Context manager for workflow execution logging.
        
        Usage:
            with audit.execution_context("workflow_1", "My Workflow", params) as log_path:
                # Execute workflow
                audit.log_step(...)
        """
        log_path = self.start_execution(workflow_id, workflow_name, parameters)
        success = True
        error = None
        
        try:
            yield log_path
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            self.end_execution(success=success, error=error)


# Global instance
audit_log = AuditLog()
