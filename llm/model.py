import os
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from enum import Enum
from pydantic import BaseModel
from typing import Tuple, Dict, Any, List, Optional
import requests
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult


class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""
    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GEMINI = "Gemini"
    GROQ = "Groq"
    OPENAI = "OpenAI"
    LMSTUDIO = "LMStudio"   



class LLMModel(BaseModel):
    """Represents an LLM model configuration"""
    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)
    
    def has_json_mode(self) -> bool:
        """Check if the model supports JSON mode"""
        return not self.is_deepseek() and not self.is_gemini()
    
    def is_deepseek(self) -> bool:
        """Check if the model is a DeepSeek model"""
        return self.model_name.startswith("deepseek")
    
    def is_gemini(self) -> bool:
        """Check if the model is a Gemini model"""
        return self.model_name.startswith("gemini")


# Define available models
AVAILABLE_MODELS = [
    LLMModel(
        display_name="[anthropic] claude-3.5-haiku",
        model_name="claude-3-5-haiku-latest",
        provider=ModelProvider.ANTHROPIC
    ),
    LLMModel(
        display_name="[anthropic] claude-3.5-sonnet",
        model_name="claude-3-5-sonnet-latest",
        provider=ModelProvider.ANTHROPIC
    ),
    LLMModel(
        display_name="[anthropic] claude-3.7-sonnet",
        model_name="claude-3-7-sonnet-latest",
        provider=ModelProvider.ANTHROPIC
    ),
    LLMModel(
        display_name="[deepseek] deepseek-r1",
        model_name="deepseek-reasoner",
        provider=ModelProvider.DEEPSEEK
    ),
    LLMModel(
        display_name="[deepseek] deepseek-v3",
        model_name="deepseek-chat",
        provider=ModelProvider.DEEPSEEK
    ),
    LLMModel(
        display_name="[gemini] gemini-2.0-flash",
        model_name="gemini-2.0-flash",
        provider=ModelProvider.GEMINI
    ),
    LLMModel(
        display_name="[gemini] gemini-2.5-pro",
        model_name="gemini-2.5-pro-exp-03-25",
        provider=ModelProvider.GEMINI
    ),
    LLMModel(
        display_name="[groq] llama-3.3 70b",
        model_name="llama-3.3-70b-versatile",
        provider=ModelProvider.GROQ
    ),
    LLMModel(
        display_name="[openai] gpt-4.5",
        model_name="gpt-4.5-preview",
        provider=ModelProvider.OPENAI
    ),
    LLMModel(
        display_name="[openai] gpt-4o",
        model_name="gpt-4o",
        provider=ModelProvider.OPENAI
    ),
    LLMModel(
        display_name="[openai] o1",
        model_name="o1",
        provider=ModelProvider.OPENAI
    ),
    LLMModel(
        display_name="[openai] o3-mini",
        model_name="o3-mini",
        provider=ModelProvider.OPENAI
    ),
    LLMModel(
        display_name="[lmstudio] fin-r1-mlx",
        model_name="fin-r1-mlx",
        provider=ModelProvider.LMSTUDIO
    ),
    LLMModel(
        display_name="[lmstudio] sufe-aiflm-lab_fin-r1",
        model_name="sufe-aiflm-lab_fin-r1",
        provider=ModelProvider.LMSTUDIO
    ),
    LLMModel(
        display_name="[lmstudio] meta-llama-3.1-8b-instruct",
        model_name="meta-llama-3.1-8b-instruct",
        provider=ModelProvider.LMSTUDIO
    ),
]

# Create LLM_ORDER in the format expected by the UI
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

def get_model_info(model_name: str) -> LLMModel | None:
    """Get model information by model_name"""
    return next((model for model in AVAILABLE_MODELS if model.model_name == model_name), None)

class LMStudioChatModel(BaseChatModel):
    """LMStudio chat model wrapper for OpenAI-compatible API."""
    
    model_name: str
    base_url: str = "http://127.0.0.1:1234"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    
    def _get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models from LMStudio"""
        try:
            response = requests.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            print(f"Error fetching LMStudio models: {e}")
            return []
    
    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, str]:
        """Convert a LangChain message to a dict for the LMStudio API"""
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
        
        return {"role": role, "content": message.content}
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response from LMStudio"""
        message_dicts = [self._convert_message_to_dict(message) for message in messages]
        
        data = {
            "model": self.model_name,
            "messages": message_dicts,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
            
        if stop is not None:
            data["stop"] = stop
            
        # Add any additional kwargs
        for k, v in kwargs.items():
            data[k] = v
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=data,
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the response content
            assistant_message = response_data.get("choices", [{}])[0].get("message", {})
            content = assistant_message.get("content", "")
            
            generation = ChatGeneration(
                message=AIMessage(content=content),
                generation_info={
                    "finish_reason": response_data.get("choices", [{}])[0].get("finish_reason"),
                    "model": response_data.get("model"),
                }
            )
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            raise ValueError(f"Error calling LMStudio API: {e}")
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "lmstudio"


def get_model(model_name: str, model_provider: ModelProvider):
    """Get the appropriate LLM model based on provider and model name"""
    if model_provider == ModelProvider.ANTHROPIC:
        return ChatAnthropic(model=model_name)
    elif model_provider == ModelProvider.DEEPSEEK:
        return ChatDeepSeek(model=model_name)
    elif model_provider == ModelProvider.GEMINI:
        return ChatGoogleGenerativeAI(model=model_name)
    elif model_provider == ModelProvider.GROQ:
        return ChatGroq(model=model_name)
    elif model_provider == ModelProvider.OPENAI:
        if model_name.startswith("gpt"):  # Chat models
            return ChatOpenAI(model=model_name)
        else:  # Completion models
            return OpenAI(model=model_name)
    elif model_provider == ModelProvider.LMSTUDIO:
        return LMStudioChatModel(model_name=model_name)
    
    raise ValueError(f"Unsupported model provider: {model_provider}")