#!/usr/bin/env python3
"""
Progress tracking utilities for the Financial Agent System

This module provides utilities for tracking and reporting progress
of various operations in the system.
"""

from typing import Dict, List, Optional, Any
import time


class ProgressTracker:
    """Class for tracking progress of operations"""
    
    def __init__(self):
        """Initialize the progress tracker"""
        self.status = {}
        self.start_times = {}
        self.end_times = {}
    
    def update_status(self, component: str, operation: str, message: str):
        """
        Update the status of an operation
        
        Args:
            component: Component name (e.g., agent name, tool name)
            operation: Operation identifier
            message: Status message
        """
        key = f"{component}:{operation}"
        if key not in self.start_times:
            self.start_times[key] = time.time()
        
        self.status[key] = message
    
    def complete(self, component: str, operation: str, message: str = "Completed"):
        """
        Mark an operation as completed
        
        Args:
            component: Component name
            operation: Operation identifier
            message: Completion message
        """
        key = f"{component}:{operation}"
        self.status[key] = message
        self.end_times[key] = time.time()
    
    def get_status(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the current status of operations
        
        Args:
            component: Optional component name to filter by
            
        Returns:
            Dictionary of status information
        """
        if component:
            return {k: v for k, v in self.status.items() if k.startswith(f"{component}:")}
        return self.status
    
    def get_elapsed_time(self, component: str, operation: str) -> Optional[float]:
        """
        Get the elapsed time for an operation
        
        Args:
            component: Component name
            operation: Operation identifier
            
        Returns:
            Elapsed time in seconds, or None if operation not started
        """
        key = f"{component}:{operation}"
        if key not in self.start_times:
            return None
        
        end_time = self.end_times.get(key, time.time())
        return end_time - self.start_times[key]
    
    def reset(self):
        """Reset all progress tracking"""
        self.status = {}
        self.start_times = {}
        self.end_times = {}


# Create a global progress tracker instance
progress = ProgressTracker()
