"""
LLM Providers for Financial Agent System

This module provides utilities for initializing and accessing various LLM providers,
including OpenAI, Anthropic, DeepSeek, Gemini, Groq, and LMStudio.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union

# Set up logging
logger = logging.getLogger("llm.providers")

# Import Agno
import agno

# Set up provider mappings
PROVIDER_CONFIGS = {
    "openai": {"name": "openai"},
    "anthropic": {"name": "anthropic"},
    "deepseek": {"name": "deepseek"},
    "gemini": {"name": "gemini"},
    "groq": {"name": "groq"},
    "lmstudio": {"name": "lmstudio"},
}

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-opus",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-pro",
    "groq": "llama-3-70b-8192",
    "lmstudio": "sufe-aiflm-lab_fin-r1",  # Financial model
}

# Initialize clients based on available API keys and settings
INITIALIZED_CLIENTS = {}


def initialize_providers():
    """
    Initialize all available LLM providers based on environment variables.
    """
    # Pre-populate all providers as available for UI display purposes
    for provider in PROVIDER_CONFIGS.keys():
        INITIALIZED_CLIENTS[provider] = False
    
    # Initialize OpenAI if API key is available
    if os.environ.get("OPENAI_API_KEY"):
        try:
            INITIALIZED_CLIENTS["openai"] = True
            logger.info("Initialized OpenAI client")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    # Initialize Anthropic if API key is available
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            INITIALIZED_CLIENTS["anthropic"] = True
            logger.info("Initialized Anthropic client")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")

    # Initialize DeepSeek if API key is available
    if os.environ.get("DEEPSEEK_API_KEY"):
        try:
            INITIALIZED_CLIENTS["deepseek"] = True
            logger.info("Initialized DeepSeek client")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}")

    # Initialize Gemini if API key is available
    if os.environ.get("GEMINI_API_KEY"):
        try:
            INITIALIZED_CLIENTS["gemini"] = True
            logger.info("Initialized Gemini client")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    # Initialize Groq if API key is available
    if os.environ.get("GROQ_API_KEY"):
        try:
            INITIALIZED_CLIENTS["groq"] = True
            logger.info("Initialized Groq client")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")

    # Initialize LMStudio (no API key needed, uses local endpoint)
    try:
        INITIALIZED_CLIENTS["lmstudio"] = True
        logger.info("Initialized LMStudio client")
    except Exception as e:
        logger.error(f"Failed to initialize LMStudio client: {e}")


def get_available_providers() -> List[str]:
    """
    Get list of available LLM providers.
    
    Returns:
        List of provider names
    """
    # Return all configured providers, not just initialized ones
    return list(PROVIDER_CONFIGS.keys())


def get_default_provider() -> str:
    """
    Get the default LLM provider based on available providers.
    
    Returns:
        Default provider name
    """
    # Preference order: OpenAI > Anthropic > LMStudio > others
    if "openai" in INITIALIZED_CLIENTS:
        return "openai"
    elif "anthropic" in INITIALIZED_CLIENTS:
        return "anthropic"
    elif "lmstudio" in INITIALIZED_CLIENTS:
        return "lmstudio"
    
    # Fall back to first available provider
    available = get_available_providers()
    if available:
        return available[0]
    
    # If no providers are available, default to OpenAI
    return "openai"


def get_available_models(provider: str) -> List[str]:
    """
    Get list of available models for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        List of model names
    """
    if provider == "openai":
        return ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    elif provider == "anthropic":
        return ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
    elif provider == "deepseek":
        return ["deepseek-chat"]
    elif provider == "gemini":
        return ["gemini-pro", "gemini-pro-1.5"]
    elif provider == "groq":
        return ["llama-3-70b-8192", "llama-3-8b-8192", "mixtral-8x7b-32768"]
    elif provider == "lmstudio":
        return ["local-model", "sufe-aiflm-lab_fin-r1"]
    
    return []


def get_default_model(provider: str) -> str:
    """
    Get the default model for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        Default model name
    """
    return DEFAULT_MODELS.get(provider, "")


def get_llm_provider_model(provider: str, model: str) -> Any:
    """
    Get an Agno model configuration for the specified provider and model.
    
    Args:
        provider: Provider name
        model: Model name
        
    Returns:
        Agno model configuration object
    """
    # Ensure provider is valid
    if provider not in PROVIDER_CONFIGS:
        logger.warning(f"Unknown provider {provider}, falling back to OpenAI")
        provider = "openai"
        model = DEFAULT_MODELS["openai"]
    
    # Create a proper Agno model configuration based on provider
    try:
        # Import the appropriate model classes based on provider
        if provider == "openai":
            from agno.models.openai import OpenAIChat
            return OpenAIChat(id=model)
            
        elif provider == "anthropic":
            from agno.models.anthropic import AnthropicChat
            return AnthropicChat(id=model)
            
        elif provider == "deepseek":
            from agno.models.deepseek import DeepSeekChat
            return DeepSeekChat(id=model)
            
        elif provider == "gemini":
            from agno.models.google import GoogleChat
            return GoogleChat(id=model)
            
        elif provider == "groq":
            from agno.models.groq import GroqChat
            return GroqChat(id=model)
            
        elif provider == "lmstudio":
            # Use the dedicated LMStudio class
            from agno.models.lmstudio import LMStudio
            return LMStudio(
                id=model,
                base_url="http://127.0.0.1:1234/v1"  # Default LMStudio port
            )
            
        else:
            # Fallback to generic Model
            logger.warning(f"Using generic model for provider {provider}")
            from agno import Model
            return Model(provider=provider, name=model)
            
    except Exception as e:
        logger.error(f"Failed to create {provider} model {model}: {e}")
        # Fall back to OpenAI
        from agno.models.openai import OpenAIChat
        return OpenAIChat(id=DEFAULT_MODELS["openai"])


from enum import Enum

class LLMProvider(Enum):
    """Enum class for LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    GROQ = "groq"
    LMSTUDIO = "lmstudio"


class LLMManager:
    """Manager for LLM providers and models"""
    
    def __init__(self):
        """Initialize LLM manager"""
        self.providers = get_available_providers()
        self.default_provider = get_default_provider()
        
        # Initialize available_models dictionary
        self.available_models = {}
        
        # Populate available_models dictionary
        for provider in LLMProvider:
            if provider.value in self.providers:
                self.available_models[provider] = get_available_models(provider)
        
        # Model display names (for UI)
        self.model_display_names = {
            "gpt-4o": "GPT-4o",
            "gpt-4-turbo": "GPT-4 Turbo",
            "gpt-4": "GPT-4",
            "gpt-3.5-turbo": "GPT-3.5 Turbo",
            "claude-3-opus": "Claude 3 Opus",
            "claude-3-sonnet": "Claude 3 Sonnet",
            "claude-3-haiku": "Claude 3 Haiku",
            "deepseek-chat": "DeepSeek Chat",
            "gemini-pro": "Gemini Pro",
            "gemini-pro-1.5": "Gemini Pro 1.5",
            "llama-3-70b-8192": "Llama 3 70B",
            "llama-3-8b-8192": "Llama 3 8B",
            "mixtral-8x7b-32768": "Mixtral 8x7B",
            "sufe-aiflm-lab_fin-r1": "Fin R1"
        }
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get available LLM providers as Enum objects"""
        # Convert string provider names to LLMProvider enum objects
        string_providers = get_available_providers()
        enum_providers = []
        
        # Get all enum values
        all_providers = list(LLMProvider)
        
        # Filter to only available providers
        for provider in all_providers:
            if provider.value in string_providers:
                enum_providers.append(provider)
                
        return enum_providers
    
    def get_default_provider(self) -> LLMProvider:
        """Get default LLM provider as an Enum object"""
        default_str = get_default_provider()
        
        # Convert string to Enum object
        for provider in LLMProvider:
            if provider.value == default_str:
                return provider
                
        # Fallback to LMSTUDIO
        return LLMProvider.LMSTUDIO
    
    def get_available_models(self, provider: Union[str, LLMProvider]) -> List[str]:
        """Get available models for a provider"""
        # Convert enum to string value if needed
        if isinstance(provider, LLMProvider):
            provider = provider.value
        return get_available_models(provider)
    
    def get_default_model(self, provider: Union[str, LLMProvider]) -> str:
        """Get default model for a provider"""
        # Convert enum to string value if needed
        if isinstance(provider, LLMProvider):
            provider = provider.value
        return get_default_model(provider)
    
    def get_provider_model(self, provider: Union[str, LLMProvider], model: str) -> Any:
        """Get LLM provider model configuration"""
        # Convert enum to string value if needed
        if isinstance(provider, LLMProvider):
            provider = provider.value
        return get_llm_provider_model(provider, model)
    
    def get_model_display_name(self, model: str) -> str:
        """Get a user-friendly display name for a model"""
        return self.model_display_names.get(model, model)


# Create a singleton instance of LLMManager
llm_manager = LLMManager()

# Initialize providers when module is imported
initialize_providers()
