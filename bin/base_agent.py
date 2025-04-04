"""
Base Agno Agent for Financial Analysis

This module defines the base class for Agno-based financial analysis agents.
"""

from typing import Dict, List, Optional, Union, Any, Callable
import sys
import os
from pydantic import BaseModel
from datetime import datetime
import logging

# Import custom Agno wrapper classes
from bin.agno_wrapper import Agent, Tool


class AgnoFinancialAgent:
    """Base class for Agno-based financial analysis agents"""
    
    def __init__(self, model_name: str = "gpt-4", model_provider: str = "openai"):
        """
        Initialize the Agno financial agent
        
        Args:
            model_name: Name of the LLM model to use
            model_provider: Provider of the LLM (e.g., openai, anthropic)
        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.tools = []
        self.agent = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize the agent with the specified model
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the Agno agent with the specified model and tools"""
        
        # Create Agno agent with the specified model
        self.agent = Agent(
            model_name=self.model_name,
            model_provider=self.model_provider
        )
        
        # Add any existing tools
        if self.tools:
            self.agent.add_tools(self.tools)
    
    def add_tool(self, tool: Tool):
        """
        Add a tool to the agent
        
        Args:
            tool: Agno Tool to add to the agent
        """
        self.tools.append(tool)
        
        # Reinitialize the agent with the updated tools
        if self.agent:
            self.agent.add_tool(tool)
    
    def add_tools(self, tools: List[Tool]):
        """
        Add multiple tools to the agent
        
        Args:
            tools: List of Agno Tools to add to the agent
        """
        self.tools.extend(tools)
        
        # Reinitialize the agent with the updated tools
        if self.agent:
            self.agent.add_tools(tools)
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with the given state
        
        Args:
            state: Current state of the analysis, including data and metadata
            
        Returns:
            Updated state with analysis results
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the run method")
    
    def _prepare_agent_prompt(self, state: Dict[str, Any]) -> str:
        """
        Prepare the prompt for the agent based on the current state
        
        Args:
            state: Current state of the analysis
            
        Returns:
            Formatted prompt for the agent
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the _prepare_agent_prompt method")
    
    def _parse_agent_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the agent's response into a structured format
        
        Args:
            response: Raw response from the agent
            
        Returns:
            Structured response data
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the _parse_agent_response method")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about this agent
        
        Returns:
            Dictionary with agent information
        """
        return {
            "class_name": self.__class__.__name__,
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "tools_count": len(self.tools),
            "tool_names": [tool.name for tool in self.tools]
        }


class MasterAgent(AgnoFinancialAgent):
    """
    Base class for master agents (orchestrators) that coordinate slave agents
    
    Master agents are responsible for:
    1. Task delegation to appropriate slave agents
    2. Progress monitoring
    3. Result aggregation
    4. Final report generation
    """
    
    def __init__(self, model_name: str = "gpt-4", model_provider: str = "openai"):
        """
        Initialize the master agent
        
        Args:
            model_name: Name of the LLM model to use
            model_provider: Provider of the LLM
        """
        super().__init__(model_name, model_provider)
        self.slave_agents = {}
    
    def add_slave_agent(self, agent_id: str, agent: AgnoFinancialAgent, role: str = None):
        """
        Add a slave agent to be coordinated by this master agent
        
        Args:
            agent_id: Unique identifier for the agent
            agent: Agent instance
            role: Role description for this agent
        """
        self.slave_agents[agent_id] = {
            "instance": agent,
            "role": role or "Unspecified role"
        }
        self.logger.info(f"Added slave agent {agent_id} with role {role}")
    
    def remove_slave_agent(self, agent_id: str):
        """
        Remove a slave agent from this master agent
        
        Args:
            agent_id: ID of the agent to remove
        """
        if agent_id in self.slave_agents:
            self.slave_agents.pop(agent_id)
            self.logger.info(f"Removed slave agent {agent_id}")
    
    def delegate_task(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegate a task to a specific slave agent
        
        Args:
            agent_id: ID of the agent to delegate to
            task: Task definition and data
            
        Returns:
            Results from the slave agent
        """
        if agent_id not in self.slave_agents:
            raise ValueError(f"Slave agent {agent_id} not found")
        
        agent = self.slave_agents[agent_id]["instance"]
        self.logger.info(f"Delegating task to slave agent {agent_id}")
        
        return agent.run(task)
    
    def aggregate_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple slave agents
        
        Args:
            results: Dictionary mapping agent IDs to their results
            
        Returns:
            Aggregated results
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the aggregate_results method")
    
    def generate_final_report(self, aggregated_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a final report based on aggregated results
        
        Args:
            aggregated_results: Results aggregated from slave agents
            
        Returns:
            Final report
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the generate_final_report method")
    
    def get_slave_agent_ids(self) -> List[str]:
        """
        Get IDs of all slave agents
        
        Returns:
            List of slave agent IDs
        """
        return list(self.slave_agents.keys())
    
    def get_slave_agent_info(self, agent_id: str = None) -> Dict[str, Any]:
        """
        Get information about slave agents
        
        Args:
            agent_id: Optional ID to get info for a specific agent
            
        Returns:
            Dictionary with agent information
        """
        if agent_id:
            if agent_id not in self.slave_agents:
                raise ValueError(f"Slave agent {agent_id} not found")
            
            agent = self.slave_agents[agent_id]["instance"]
            role = self.slave_agents[agent_id]["role"]
            
            return {
                "agent_id": agent_id,
                "role": role,
                "agent_info": agent.get_agent_info()
            }
        else:
            return {
                agent_id: {
                    "role": info["role"],
                    "agent_info": info["instance"].get_agent_info()
                }
                for agent_id, info in self.slave_agents.items()
            }


class SlaveAgent(AgnoFinancialAgent):
    """
    Base class for slave agents (specialized analysts)
    
    Slave agents are responsible for:
    1. Performing specialized analysis
    2. Generating investment signals with confidence scores
    3. Providing detailed reasoning for recommendations
    """
    
    def __init__(self, 
               model_name: str = "gpt-4", 
               model_provider: str = "openai",
               analysis_type: str = "general"):
        """
        Initialize the slave agent
        
        Args:
            model_name: Name of the LLM model to use
            model_provider: Provider of the LLM
            analysis_type: Type of analysis this agent specializes in
        """
        super().__init__(model_name, model_provider)
        self.analysis_type = analysis_type
    
    def perform_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform specialized analysis on the given data
        
        Args:
            data: Data to analyze
            
        Returns:
            Analysis results
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the perform_analysis method")
    
    def generate_signal(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an investment signal based on analysis results
        
        Args:
            analysis_results: Results from perform_analysis
            
        Returns:
            Investment signal with confidence score and reasoning
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the generate_signal method")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about this agent
        
        Returns:
            Dictionary with agent information
        """
        info = super().get_agent_info()
        info["analysis_type"] = self.analysis_type
        return info
