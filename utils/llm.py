"""Helper functions for LLM"""

import json
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from utils.progress import progress
from llm.models import ModelProvider

T = TypeVar('T', bound=BaseModel)

def extract_fields_from_reasoning_model(content: str, field_names: list) -> dict:
    """
    Extract fields from a reasoning model's response that might not be in perfect JSON format.
    Designed for models like O1 or fin-r1-mlx that provide detailed reasoning.
    
    Args:
        content: The model's response text
        field_names: List of field names to extract from the response
        
    Returns:
        Dictionary with extracted field values
    """
    import re
    result = {}
    
    # Common patterns for different field types
    patterns = {
        "signal": r'(?:signal|recommendation|decision)\s*[:\-=]\s*["\']*([\w]+)["\']*',
        "confidence": r'(?:confidence|certainty|probability)\s*[:\-=]\s*["\']*([\d\.]+)["\']*',
        "reasoning": r'(?:reasoning|rationale|analysis)\s*[:\-=]\s*["\']*([^"\n]+(?:\n[^"\n]+)*)',
    }
    
    # For any field not covered by the patterns above
    generic_pattern = r'{}\s*[:\-=]\s*["\']*([^"\n]+(?:\n[^"\n]+)*)'  
    
    for field in field_names:
        # Use specific pattern if available, otherwise use generic
        pattern = patterns.get(field.lower(), generic_pattern.format(field))
        match = re.search(pattern, content, re.IGNORECASE)
        
        if match:
            value = match.group(1).strip()
            
            # Convert to appropriate type based on field name
            if field.lower() == "confidence":
                # Handle percentage or decimal
                if "%" in value:
                    value = float(value.replace("%", "")) / 100
                else:
                    value = float(value)
            elif field.lower() == "signal" and value.lower() in ["bullish", "bearish", "neutral"]:
                value = value.lower()
                
            result[field] = value
    
    return result

def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory = None
) -> T:
    """
    Makes an LLM call with retry logic, handling different model providers.
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure
        
    Returns:
        An instance of the specified Pydantic model
    """
    from llm.models import get_model, get_model_info, ModelProvider
    
    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)
    
    # Modify the prompt for LMStudio reasoning models
    if model_provider == ModelProvider.LMSTUDIO:
        # For reasoning models like fin-r1-mlx, add explicit instructions to format as JSON
        if isinstance(prompt, str):
            # If prompt is a string, add the instruction
            prompt = prompt + "\n\nPlease format your response as a valid JSON object with no additional text before or after."
        # Don't use structured output for LMStudio, handle it manually
        pass
    # For models that support JSON mode, use structured output
    elif not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )
    
    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)
            
            # Handle LMStudio specifically
            if model_provider == ModelProvider.LMSTUDIO:
                if hasattr(result, 'content'):
                    # Direct content from AIMessage
                    content = result.content
                elif hasattr(result, 'generations') and result.generations:
                    # Content from ChatResult
                    content = result.generations[0].message.content
                else:
                    # Try to handle dictionary-like response
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # For reasoning models, try to extract structured data
                try:
                    # First try to extract JSON directly
                    parsed_result = extract_json_from_lmstudio_response(content)
                    if parsed_result:
                        return pydantic_model(**parsed_result)
                    
                    # If that fails, try a more flexible approach for reasoning models
                    # Extract key fields based on the expected model structure
                    if hasattr(pydantic_model, 'model_fields'):
                        field_names = list(pydantic_model.model_fields.keys())
                        extracted_data = extract_fields_from_reasoning_model(content, field_names)
                        if extracted_data and len(extracted_data) > 0:
                            return pydantic_model(**extracted_data)
                    
                    # If we still can't extract data, raise an error
                    raise ValueError(f"Could not parse output from LMStudio reasoning model: {content[:200]}...")
                except Exception as e:
                    print(f"Error extracting data from LMStudio response: {e}")
                    raise ValueError(f"Failed to extract structured data from LMStudio response: {str(e)}")
            
            # For models without JSON mode support, extract JSON manually
            elif model_info and not model_info.has_json_mode():
                # Handle other models like DeepSeek
                parsed_result = extract_json_from_deepseek_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
                else:
                    raise ValueError(f"Could not parse JSON from response: {result.content[:200]}...")
            else:
                # For models with JSON mode, the result should already be structured
                return result
                
        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")
            
            print(f"Error in LLM call (attempt {attempt+1}/{max_retries}): {str(e)}")
            
            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)

def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None
    
    return model_class(**default_values)

def extract_json_from_deepseek_response(content: str) -> Optional[dict]:
    """Extracts JSON from Deepseek's markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from Deepseek response: {e}")
    return None

def extract_json_from_lmstudio_response(content: str) -> Optional[dict]:
    """Extracts JSON from LMStudio's response, handling control characters."""
    def clean_json_text(text):
        """Clean JSON text by handling control characters and newlines in strings."""
        # Replace literal newlines in strings with \n
        # First, try a simple approach to fix common issues
        # Replace control characters with their escaped versions
        for i in range(32):
            if i not in (9, 10, 13):  # tab, newline, carriage return
                text = text.replace(chr(i), f"\\u{i:04x}")
        
        # Handle newlines in strings more carefully
        # This is a simplified approach and may not work for all cases
        in_string = False
        in_escape = False
        result = []
        
        for char in text:
            if in_escape:
                result.append(char)
                in_escape = False
                continue
                
            if char == '\\' and in_string:
                in_escape = True
                result.append(char)
                continue
                
            if char == '"':
                in_string = not in_string
                
            if char in ('\n', '\r') and in_string:
                result.append('\\n')  # Replace with escaped newline
            else:
                result.append(char)
                
        return ''.join(result)
    
    try:
        # First try to parse directly as JSON
        return json.loads(content)
    except json.JSONDecodeError:
        # If that fails, try to extract and clean JSON from markdown code blocks
        try:
            # Check for JSON in code blocks
            json_start = content.find("```json")
            if json_start != -1:
                json_text = content[json_start + 7:]  # Skip past ```json
                json_end = json_text.find("```")
                if json_end != -1:
                    json_text = json_text[:json_end].strip()
                    cleaned_json = clean_json_text(json_text)
                    try:
                        return json.loads(cleaned_json)
                    except json.JSONDecodeError as e:
                        print(f"Still couldn't parse JSON after cleaning: {e}")
            
            # Check for code blocks without language specifier
            json_start = content.find("```")
            if json_start != -1:
                json_text = content[json_start + 3:]  # Skip past ```
                json_end = json_text.find("```")
                if json_end != -1:
                    json_text = json_text[:json_end].strip()
                    cleaned_json = clean_json_text(json_text)
                    try:
                        return json.loads(cleaned_json)
                    except json.JSONDecodeError:
                        pass  # Not valid JSON, continue searching
            
            # Look for JSON-like structures with curly braces
            brace_start = content.find("{")
            if brace_start != -1:
                # Try to find the matching closing brace
                # This is a simplified approach and might not work for all cases
                stack = 0
                for i in range(brace_start, len(content)):
                    if content[i] == "{":
                        stack += 1
                    elif content[i] == "}":
                        stack -= 1
                        if stack == 0:
                            # Found a complete JSON object
                            json_text = content[brace_start:i+1]
                            cleaned_json = clean_json_text(json_text)
                            try:
                                return json.loads(cleaned_json)
                            except json.JSONDecodeError as e:
                                print(f"Failed to parse JSON object: {e}")
                                # Try a more aggressive approach - use a JSON repair library if available
                                try:
                                    # Fallback to a simple regex-based approach to extract valid fields
                                    import re
                                    result = {}
                                    # Extract key-value pairs with regex
                                    pattern = r'"([^"]+)"\s*:\s*("[^"]*"|\d+\.?\d*|true|false|null|\{[^\}]*\}|\[[^\]]*\])'
                                    matches = re.findall(pattern, cleaned_json)
                                    for key, value in matches:
                                        try:
                                            # Try to parse the value
                                            if value.startswith('"') and value.endswith('"'):
                                                # String value
                                                result[key] = value[1:-1]
                                            elif value.lower() == 'true':
                                                result[key] = True
                                            elif value.lower() == 'false':
                                                result[key] = False
                                            elif value.lower() == 'null':
                                                result[key] = None
                                            elif value.replace('.', '', 1).isdigit():
                                                # Numeric value
                                                if '.' in value:
                                                    result[key] = float(value)
                                                else:
                                                    result[key] = int(value)
                                        except Exception:
                                            # If we can't parse the value, use it as a string
                                            result[key] = str(value)
                                    
                                    if result:
                                        return result
                                except Exception as regex_err:
                                    print(f"Regex extraction failed: {regex_err}")
            
            # Last resort - try to manually construct a valid JSON
            try:
                # Look for signal, confidence, and reasoning fields
                signal_match = re.search(r'"signal"\s*:\s*"([^"]+)"', content)
                confidence_match = re.search(r'"confidence"\s*:\s*(\d+\.?\d*)', content)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)', content)
                
                if signal_match or confidence_match or reasoning_match:
                    result = {}
                    if signal_match:
                        result["signal"] = signal_match.group(1)
                    if confidence_match:
                        result["confidence"] = float(confidence_match.group(1))
                    if reasoning_match:
                        # Extract reasoning and clean it
                        reasoning = reasoning_match.group(1)
                        # Truncate at first occurrence of a double quote not preceded by a backslash
                        end_idx = 0
                        for i in range(len(reasoning)):
                            if reasoning[i] == '"' and (i == 0 or reasoning[i-1] != '\\'):
                                end_idx = i
                                break
                        if end_idx > 0:
                            reasoning = reasoning[:end_idx]
                        result["reasoning"] = reasoning
                    
                    if result:
                        return result
            except Exception as e:
                print(f"Manual JSON extraction failed: {e}")
            
            print(f"Warning: Could not extract JSON from LMStudio response: {content[:200]}...")
            return None
        except Exception as e:
            print(f"Error extracting JSON from LMStudio response: {e}")
            return None
