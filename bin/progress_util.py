"""
Progress Tracking Utilities for Agno-based Agents

This module provides utilities for tracking progress of Agno-based financial analysis agents.
"""

from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
import time
import logging
from enum import Enum


class ProgressStatus(str, Enum):
    """Enum for progress status"""
    
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


def progress(message: str, step: int = None, total_steps: int = None):
    """
    Simple progress reporting function for agents
    
    Args:
        message: Progress message to log
        step: Current step number (optional)
        total_steps: Total steps in the process (optional)
        
    Returns:
        None, just logs the progress
    """
    logger = logging.getLogger("AgentProgress")
    
    if step is not None and total_steps is not None:
        percentage = (step / total_steps) * 100
        logger.info(f"[{percentage:.1f}%] {message}")
    else:
        logger.info(f"Progress: {message}")
    
    return None


class ProgressTracker:
    """Utility for tracking progress of agent execution"""
    
    def __init__(self, total_steps: int = 1, name: str = "analysis"):
        """
        Initialize the progress tracker
        
        Args:
            total_steps: Total number of steps in the analysis
            name: Name of the analysis process
        """
        self.name = name
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.end_time = None
        self.status = ProgressStatus.NOT_STARTED
        self.steps = {}
        self.logger = logging.getLogger(f"ProgressTracker:{name}")
    
    def start(self):
        """Start the progress tracking"""
        self.start_time = datetime.now()
        self.status = ProgressStatus.IN_PROGRESS
        self.logger.info(f"Started {self.name} with {self.total_steps} total steps")
        
        return self
    
    def update(self, 
             step: int = None, 
             status: ProgressStatus = ProgressStatus.IN_PROGRESS, 
             message: str = None,
             data: Dict[str, Any] = None):
        """
        Update the progress
        
        Args:
            step: Current step (if None, increment current_step)
            status: Status of the current step
            message: Message describing the current progress
            data: Additional data for this progress update
        """
        if self.start_time is None:
            self.start()
        
        if step is None:
            step = self.current_step + 1
        
        self.current_step = step
        
        # Store step information
        self.steps[step] = {
            "status": status,
            "message": message,
            "time": datetime.now(),
            "data": data or {}
        }
        
        # Log the update
        if message:
            self.logger.info(f"Step {step}/{self.total_steps} ({status}): {message}")
        else:
            self.logger.info(f"Step {step}/{self.total_steps} ({status})")
        
        return self
    
    def complete(self, message: str = None, data: Dict[str, Any] = None):
        """
        Mark the process as completed
        
        Args:
            message: Completion message
            data: Additional data for completion
        """
        self.end_time = datetime.now()
        self.status = ProgressStatus.COMPLETED
        
        if message:
            self.logger.info(f"Completed {self.name}: {message}")
        else:
            self.logger.info(f"Completed {self.name}")
        
        # Store completion information
        self.steps["completion"] = {
            "status": ProgressStatus.COMPLETED,
            "message": message,
            "time": self.end_time,
            "data": data or {}
        }
        
        return self
    
    def fail(self, message: str = None, data: Dict[str, Any] = None):
        """
        Mark the process as failed
        
        Args:
            message: Failure message
            data: Additional data for failure
        """
        self.end_time = datetime.now()
        self.status = ProgressStatus.FAILED
        
        if message:
            self.logger.error(f"Failed {self.name}: {message}")
        else:
            self.logger.error(f"Failed {self.name}")
        
        # Store failure information
        self.steps["failure"] = {
            "status": ProgressStatus.FAILED,
            "message": message,
            "time": self.end_time,
            "data": data or {}
        }
        
        return self
    
    def partial(self, message: str = None, data: Dict[str, Any] = None):
        """
        Mark the process as partially completed
        
        Args:
            message: Partial completion message
            data: Additional data for partial completion
        """
        self.end_time = datetime.now()
        self.status = ProgressStatus.PARTIAL
        
        if message:
            self.logger.warning(f"Partially completed {self.name}: {message}")
        else:
            self.logger.warning(f"Partially completed {self.name}")
        
        # Store partial completion information
        self.steps["partial"] = {
            "status": ProgressStatus.PARTIAL,
            "message": message,
            "time": self.end_time,
            "data": data or {}
        }
        
        return self
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status
        
        Returns:
            Status information
        """
        duration = None
        if self.start_time:
            if self.end_time:
                duration = (self.end_time - self.start_time).total_seconds()
            else:
                duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "name": self.name,
            "status": self.status,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percentage": (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": duration,
            "steps": self.steps
        }
