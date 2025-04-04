"""
Team Coordinator for Financial Agent System

This module provides utilities for coordinating master and slave agents using Agno Teams
with the coordinate mode pattern.
"""

from typing import Dict, List, Optional, Union, Any, Callable, Type
import logging
from datetime import datetime
import agno
from agno.team import Team
from enum import Enum


class ProgressStatus(str, Enum):
    """Enum for progress status"""
    
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ProgressTracker:
    """Utility for tracking progress of agent execution"""
    
    def __init__(self, total_steps: int = 1, name: str = "analysis"):
        """
        Initialize a progress tracker
        
        Args:
            total_steps: Total number of steps in the process
            name: Name of the process being tracked
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.name = name
        self.start_time = None
        self.end_time = None
        self.status = ProgressStatus.NOT_STARTED
        self.step_times = {}
        self.step_statuses = {}
        self.messages = []
        self.logger = logging.getLogger(f"ProgressTracker:{name}")
        
    def start(self, message: str = None):
        """
        Start tracking progress
        
        Args:
            message: Optional message to log
        """
        self.start_time = datetime.now()
        self.status = ProgressStatus.IN_PROGRESS
        
        if message:
            self.add_message(message)
        
        self.logger.info(f"Started {self.name} with {self.total_steps} total steps")
        
    def step(self, message: str = None, step_name: str = None):
        """
        Record a step completion
        
        Args:
            message: Optional message to log
            step_name: Optional name for this step
        """
        if self.status != ProgressStatus.IN_PROGRESS:
            self.start()
            
        self.current_step += 1
        step_time = datetime.now()
        
        if step_name is None:
            step_name = f"Step {self.current_step}"
            
        self.step_times[step_name] = step_time
        self.step_statuses[step_name] = ProgressStatus.COMPLETED
        
        if message:
            self.add_message(f"[{step_name}] {message}")
            
        percentage = (self.current_step / self.total_steps) * 100
        self.logger.info(f"[{percentage:.1f}%] {message or step_name}")
        
    def complete(self, message: str = None):
        """
        Mark tracking as complete
        
        Args:
            message: Optional message to log
        """
        self.end_time = datetime.now()
        self.status = ProgressStatus.COMPLETED
        
        if message:
            self.add_message(message)
            
        duration = (self.end_time - self.start_time).total_seconds()
        self.logger.info(f"Completed {self.name} in {duration:.2f} seconds")
        
    def fail(self, message: str = None, error: Exception = None):
        """
        Mark tracking as failed
        
        Args:
            message: Optional message to log
            error: Optional exception that caused the failure
        """
        self.end_time = datetime.now()
        self.status = ProgressStatus.FAILED
        
        if error:
            error_message = f"Failed {self.name}: {str(error)}"
            self.add_message(error_message)
            self.logger.error(error_message)
        elif message:
            self.add_message(f"Failed: {message}")
            self.logger.error(f"Failed {self.name}: {message}")
            
    def add_message(self, message: str):
        """
        Add a message to the progress log
        
        Args:
            message: Message to add
        """
        timestamp = datetime.now()
        self.messages.append({
            "time": timestamp,
            "message": message
        })
        
    def get_progress(self) -> Dict[str, Any]:
        """
        Get the current progress state
        
        Returns:
            Dictionary with progress information
        """
        return {
            "name": self.name,
            "status": self.status,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "percentage": (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None,
            "messages": self.messages
        }


class Tool:
    """Wrapper for Agno Tool class"""
    
    def __init__(self, name: str, func: Callable, description: str):
        """
        Initialize an Agno tool
        
        Args:
            name: Name of the tool
            func: Function to execute when the tool is called
            description: Description of what the tool does
        """
        self.name = name
        self.func = func
        self.description = description
        
    def to_agno_tool(self):
        """Convert to native Agno tool format"""
        return agno.Tool(
            name=self.name,
            func=self.func,
            description=self.description
        )


class Agent:
    """Wrapper for Agno Agent class"""
    
    def __init__(self, name: str = None, role: str = None, model_name: str = "gpt-4", model_provider: str = "openai", instructions: List[str] = None):
        """
        Initialize an Agno agent
        
        Args:
            name: Name of the agent
            role: Role description for the agent
            model_name: Name of the LLM model to use
            model_provider: Provider of the LLM
            instructions: List of instructions for the agent
        """
        self.name = name
        self.role = role
        self.model_name = model_name
        self.model_provider = model_provider
        self.instructions = instructions or []
        self.tools = []
        
        # Create the Agno agent
        self.agent = self._create_agent()
        
    def _create_agent(self):
        """Create the native Agno agent"""
        model = self._get_model()
        
        agent_kwargs = {
            "model": model,
        }
        
        if self.name:
            agent_kwargs["name"] = self.name
            
        if self.role:
            agent_kwargs["role"] = self.role
            
        if self.instructions:
            agent_kwargs["instructions"] = self.instructions
            
        return agno.Agent(**agent_kwargs)
    
    def _get_model(self):
        """Get the appropriate LLM model based on provider and model name"""
        if self.model_provider.lower() == "openai":
            from agno.models.openai import OpenAIChat
            return OpenAIChat(model=self.model_name)
        elif self.model_provider.lower() == "anthropic":
            from agno.models.anthropic import Claude
            return Claude(model=self.model_name)
        elif self.model_provider.lower() == "gemini":
            from agno.models.google import Gemini
            return Gemini(model=self.model_name)
        elif self.model_provider.lower() == "groq":
            from agno.models.groq import Groq
            return Groq(model=self.model_name)
        elif self.model_provider.lower() == "lmstudio":
            # Configure LMStudio with the local server URL
            from agno.models.lmstudio import LMStudio
            return LMStudio(
                model=self.model_name,
                api_base="http://localhost:1234/v1",
                api_key="lm-studio"
            )
        else:
            # Default to OpenAI if provider not recognized
            from agno.models.openai import OpenAIChat
            return OpenAIChat(model=self.model_name)
        
    def add_tool(self, tool: Tool):
        """
        Add a tool to the agent
        
        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
        self.agent.add_tool(tool.to_agno_tool())
        
    def add_tools(self, tools: List[Tool]):
        """
        Add multiple tools to the agent
        
        Args:
            tools: List of tools to add
        """
        for tool in tools:
            self.add_tool(tool)
            
    def run(self, prompt: str, **kwargs):
        """
        Run the agent with the given prompt
        
        Args:
            prompt: Input prompt for the agent
            kwargs: Additional keyword arguments for Agno Agent run
            
        Returns:
            Agent response
        """
        return self.agent.run(prompt, **kwargs)


class TeamCoordinator:
    """
    Coordinator for master-slave teams using Agno's coordinate mode
    
    This class facilitates the creation and coordination of teams with
    a master agent (coordinator) and slave agents (specialized analysts).
    """
    
    def __init__(self):
        """Initialize the team coordinator"""
        self.teams = {}
        self.logger = logging.getLogger("TeamCoordinator")
        
    def create_team(self, team_name: str, team_description: str = None, mode: str = "coordinate") -> Team:
        """
        Create a new team with coordinate mode
        
        Args:
            team_name: Name of the team
            team_description: Description of the team's purpose
            mode: Team mode (default: coordinate)
            
        Returns:
            Newly created team
        """
        self.logger.info(f"TeamCoordinator.create_team called with name={team_name}, mode={mode}")
        
        try:
            # Create the team using Agno Team
            # Note: In Agno, members must be provided at team creation time
            # We'll create an empty list and add members later
            team_kwargs = {
                "name": team_name,
                "description": team_description,
                "mode": mode,
                "members": []
            }
            
            self.logger.info(f"Creating team with kwargs: {team_kwargs}")
            
            # Debug: Print Team class signature
            import inspect
            team_init_params = inspect.signature(Team.__init__).parameters
            self.logger.info(f"Team.__init__ parameters: {list(team_init_params.keys())}")
            
            # Create the team
            self.logger.info("Attempting to create Team instance")
            team = Team(**team_kwargs)
            self.logger.info(f"Team created successfully: {team}")
        except Exception as e:
            self.logger.error(f"Error creating team: {str(e)}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        self.teams[team_name] = team
        self.logger.info(f"Created team {team_name} with {mode} mode")
        
        return team
    
    def add_master_agent(self, team: Team, agent: Agent, instructions: List[str] = None):
        """
        Add a master agent (coordinator) to the team
        
        Args:
            team: Team to add the master agent to
            agent: Agent instance
            instructions: List of instructions for the master agent
        """
        # Add instructions if provided
        if instructions:
            team.instructions = instructions
            
        # The master agent should be added as the first member of the team
        team.members.append(agent.agent)
            
        self.logger.info(f"Added master agent {agent.name} to team {team.name}")
        
    def add_slave_agent(self, team: Team, agent: Agent):
        """
        Add a slave agent to the team
        
        Args:
            team: Team to add the slave agent to
            agent: Agent instance
        """
        team.members.append(agent.agent)
        self.logger.info(f"Added slave agent {agent.name} to team {team.name}")
        
    def run_team(self, team: Team, prompt: str, **kwargs):
        """
        Run the team with the given prompt
        
        Args:
            team: Team to run
            prompt: Input prompt for the team
            kwargs: Additional keyword arguments for Agno Team run
            
        Returns:
            Team response
        """
        return team.run(prompt, **kwargs)
