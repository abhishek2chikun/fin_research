"""
Custom Agno Framework Wrapper

This module provides custom wrappers for Agno framework components to ensure
compatibility with the Financial Agent System.
"""

from typing import Dict, List, Optional, Union, Any, Callable
import agno
import logging

# Set up logging
logger = logging.getLogger("agno_wrapper")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
    
    def __init__(self, 
                 model_name: str = "gpt-4", 
                 model_provider: str = "openai",
                 name: str = None,
                 role: str = None,
                 description: str = None,
                 instructions: List[str] = None,
                 model = None,
                 add_datetime_to_instructions: bool = False,
                 **kwargs):
        logger.debug(f"Initializing Agent with params: model_name={model_name}, provider={model_provider}, name={name}, model={model}")
        """
        Initialize an Agno agent
        
        Args:
            model_name: Name of the LLM model to use
            model_provider: Provider of the LLM
            name: Name of the agent
            role: Role of the agent
            description: Description of the agent
            instructions: List of instructions for the agent
            model: Model object (if provided directly)
            add_datetime_to_instructions: Whether to add datetime to instructions
            **kwargs: Additional keyword arguments for Agno Agent
        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.name = name
        self.role = role
        self.description = description
        self.instructions = instructions or []
        self.tools = []
        
        # Create native Agno agent with all parameters
        agent_kwargs = {}
        
        # Handle model parameter (either direct model object or model_name/provider)
        if model is not None:
            logger.debug(f"Using provided model object: {type(model)}")
            agent_kwargs['model'] = model
        else:
            logger.debug(f"Using model_name={model_name} and provider={model_provider}")
            agent_kwargs['model'] = model_name
            agent_kwargs['provider'] = model_provider
            
        # Add other parameters if provided
        if name:
            agent_kwargs['name'] = name
        if role:
            agent_kwargs['role'] = role
        if description:
            agent_kwargs['description'] = description
        if instructions:
            agent_kwargs['instructions'] = instructions
        if add_datetime_to_instructions:
            agent_kwargs['add_datetime_to_instructions'] = add_datetime_to_instructions
            
        # Add any other kwargs
        agent_kwargs.update(kwargs)
        
        # Create the agent
        logger.debug(f"Creating Agno Agent with kwargs: {agent_kwargs}")
        try:
            self.agent = agno.Agent(**agent_kwargs)
            logger.debug("Successfully created Agno Agent")
        except Exception as e:
            logger.error(f"Error creating Agno Agent: {e}")
            raise
    
    def add_tool(self, tool: Tool):
        """Add a tool to the agent"""
        self.tools.append(tool)
        self.agent.add_tool(tool.to_agno_tool())
    
    def add_tools(self, tools: List[Tool]):
        """Add multiple tools to the agent"""
        self.tools.extend(tools)
        for tool in tools:
            self.agent.add_tool(tool.to_agno_tool())
    
    def run(self, prompt: str, **kwargs) -> str:
        """Run the agent with the given prompt"""
        return self.agent.run(prompt, **kwargs)


class Team:
    """Wrapper for Agno Team functionality"""
    
    def __init__(self, 
                 name: str, 
                 description: str = None,
                 mode: str = "coordinate",
                 model = None,  # Keep parameter for compatibility but don't use it
                 members: List[Agent] = None,
                 instructions: List[str] = None,
                 add_datetime_to_instructions: bool = False,
                 send_team_context_to_members: bool = False,
                 show_members_responses: bool = False,
                 markdown: bool = False,
                 **kwargs):
        logger.debug(f"Initializing Team with params: name={name}, mode={mode}, members={len(members) if members else 0}")
        if model is not None:
            logger.warning("Model parameter provided but will be ignored to avoid errors")
        """
        Initialize an Agno Team
        
        Args:
            name: Name of the team
            description: Description of the team's purpose
            mode: Team mode (coordinate, collaborate, route)
            model: Model object for the team
            members: List of team members
            instructions: List of instructions for the team
            add_datetime_to_instructions: Whether to add datetime to instructions
            send_team_context_to_members: Whether to send team context to members
            show_members_responses: Whether to show member responses
            markdown: Whether to use markdown formatting
            **kwargs: Additional keyword arguments for Agno Team
        """
        self.name = name
        self.description = description
        self.mode = mode
        self.agents = {}
        self.members = members or []
        self.instructions = instructions or []
        
        # Create team kwargs
        team_kwargs = {
            'name': name,
            'mode': mode
        }
        
        # Add optional parameters if provided
        if description:
            team_kwargs['description'] = description
        # IMPORTANT: DO NOT add model parameter - it's not supported in the current Agno version
        if instructions:
            team_kwargs['instructions'] = instructions
        if add_datetime_to_instructions:
            team_kwargs['add_datetime_to_instructions'] = add_datetime_to_instructions
        if send_team_context_to_members:
            team_kwargs['send_team_context_to_members'] = send_team_context_to_members
        if show_members_responses:
            team_kwargs['show_members_responses'] = show_members_responses
        if markdown:
            team_kwargs['markdown'] = markdown
            
        # Add any other kwargs, but ensure model is not included
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'model'}
        team_kwargs.update(filtered_kwargs)
        
        logger.debug(f"Creating Agno Team with kwargs: {team_kwargs}")
        
        # Create the team
        try:
            self.team = agno.Team(**team_kwargs)
            logger.debug("Successfully created Agno Team")
        except Exception as e:
            logger.error(f"Error creating Team: {e}")
            raise
        
        # Add members if provided
        if members:
            for member in members:
                self.add_member(member)
    
    def add_agent(self, agent_id: str, agent: Agent, role: str = None):
        """
        Add an agent to the team
        
        Args:
            agent_id: Unique identifier for the agent
            agent: Agent instance to add
            role: Role description for the agent in the team
        """
        self.agents[agent_id] = agent
        self.team.add_agent(agent_id, agent.agent, role=role)
        
    def add_member(self, agent: Agent):
        """
        Add a member to the team (simplified method that matches Agno examples)
        
        Args:
            agent: Agent instance to add as a member
        """
        # Generate a unique ID based on agent name or a counter
        agent_id = agent.name if agent.name else f"agent_{len(self.agents)}"
        self.add_agent(agent_id, agent, role=agent.role)
    
    def run(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Run the team with the given prompt
        
        Args:
            prompt: Input prompt for the team
            kwargs: Additional keyword arguments for Agno Team run
            
        Returns:
            Results from all agents in the team
        """
        return self.team.run(prompt, **kwargs)


class Workflow:
    """Wrapper for Agno Workflow functionality"""
    
    def __init__(self, name: str):
        """
        Initialize an Agno Workflow
        
        Args:
            name: Name of the workflow
        """
        self.name = name
        self.workflow = agno.Workflow(name=name)
        self.steps = []
    
    def add_step(self, step_name: str, agent_or_team: Union[Agent, Team], prompt_template: str):
        """
        Add a step to the workflow
        
        Args:
            step_name: Name of the step
            agent_or_team: Agent or Team to execute in this step
            prompt_template: Template for the prompt to send to the agent/team
        """
        if isinstance(agent_or_team, Agent):
            native_agent = agent_or_team.agent
            self.workflow.add_step(step_name, native_agent, prompt_template)
        elif isinstance(agent_or_team, Team):
            native_team = agent_or_team.team
            self.workflow.add_step(step_name, native_team, prompt_template)
        
        self.steps.append({
            "name": step_name,
            "executor": agent_or_team,
            "prompt_template": prompt_template
        })
    
    def run(self, initial_state: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Run the workflow with the given initial state
        
        Args:
            initial_state: Initial state for the workflow
            kwargs: Additional keyword arguments for Agno Workflow run
            
        Returns:
            Final state after all steps are executed
        """
        if initial_state is None:
            initial_state = {}
        
        return self.workflow.run(initial_state, **kwargs)
