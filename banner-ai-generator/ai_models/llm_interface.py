"""
Large Language Model Interface

Unified interface for interacting with various LLM providers
including OpenAI GPT, Anthropic Claude, and local models.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from enum import Enum
import json
from structlog import get_logger

logger = get_logger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    AZURE = "azure"


class LLMInterface:
    """
    Unified interface for Large Language Models
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Provider configurations
        self.providers_config = {
            LLMProvider.OPENAI: config.get("openai", {}),
            LLMProvider.ANTHROPIC: config.get("anthropic", {}),
            LLMProvider.LOCAL: config.get("local", {}),
            LLMProvider.AZURE: config.get("azure", {})
        }
        
        # Default provider
        self.default_provider = LLMProvider(config.get("default_provider", "openai"))
        
        # Model mappings
        self.model_mappings = {
            # OpenAI models
            "gpt-4": LLMProvider.OPENAI,
            "gpt-4-turbo": LLMProvider.OPENAI,
            "gpt-3.5-turbo": LLMProvider.OPENAI,
            
            # Anthropic models
            "claude-3-opus": LLMProvider.ANTHROPIC,
            "claude-3-sonnet": LLMProvider.ANTHROPIC,
            "claude-3-haiku": LLMProvider.ANTHROPIC,
            
            # Local models
            "llama-2": LLMProvider.LOCAL,
            "mistral": LLMProvider.LOCAL
        }
        
        # Initialize clients
        self._clients = {}
        asyncio.create_task(self._initialize_clients())
    
    async def generate_text(self, 
                           prompt: str,
                           model: Optional[str] = None,
                           max_tokens: int = 1000,
                           temperature: float = 0.7,
                           system_prompt: Optional[str] = None,
                           **kwargs) -> str:
        """
        Generate text using specified LLM
        
        Args:
            prompt: User prompt
            model: Model name (optional, uses default if not specified)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: System prompt (optional)
            **kwargs: Additional model-specific parameters
        
        Returns:
            Generated text
        """
        try:
            provider, model_name = self._resolve_model(model)
            
            logger.info(f"Generating text with {provider.value}:{model_name}")
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Generate based on provider
            if provider == LLMProvider.OPENAI:
                result = await self._generate_openai(
                    messages, model_name, max_tokens, temperature, **kwargs
                )
            elif provider == LLMProvider.ANTHROPIC:
                result = await self._generate_anthropic(
                    messages, model_name, max_tokens, temperature, **kwargs
                )
            elif provider == LLMProvider.LOCAL:
                result = await self._generate_local(
                    messages, model_name, max_tokens, temperature, **kwargs
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            logger.info(f"Generated {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    async def generate_text_stream(self, 
                                  prompt: str,
                                  model: Optional[str] = None,
                                  max_tokens: int = 1000,
                                  temperature: float = 0.7,
                                  system_prompt: Optional[str] = None,
                                  **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming response
        
        Args:
            prompt: User prompt
            model: Model name
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            system_prompt: System prompt
            **kwargs: Additional parameters
        
        Yields:
            Text chunks as they are generated
        """
        try:
            provider, model_name = self._resolve_model(model)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            if provider == LLMProvider.OPENAI:
                async for chunk in self._stream_openai(
                    messages, model_name, max_tokens, temperature, **kwargs
                ):
                    yield chunk
            elif provider == LLMProvider.ANTHROPIC:
                async for chunk in self._stream_anthropic(
                    messages, model_name, max_tokens, temperature, **kwargs
                ):
                    yield chunk
            else:
                # Fallback to non-streaming
                result = await self.generate_text(
                    prompt, model, max_tokens, temperature, system_prompt, **kwargs
                )
                yield result
                
        except Exception as e:
            logger.error(f"Error streaming text: {e}")
            raise
    
    async def generate_structured_output(self, 
                                       prompt: str,
                                       schema: Dict[str, Any],
                                       model: Optional[str] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate structured JSON output
        
        Args:
            prompt: Prompt requesting structured data
            schema: JSON schema for expected output
            model: Model name
            **kwargs: Additional parameters
        
        Returns:
            Parsed JSON response
        """
        try:
            # Add schema instructions to prompt
            schema_prompt = f"""
            {prompt}
            
            Please respond with valid JSON that matches this schema:
            {json.dumps(schema, indent=2)}
            
            Respond with only the JSON, no additional text.
            """
            
            response = await self.generate_text(
                schema_prompt, 
                model=model,
                temperature=0.1,  # Lower temperature for structured output
                **kwargs
            )
            
            # Extract and parse JSON
            try:
                # Try to find JSON in response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    json_str = response.strip()
                
                result = json.loads(json_str)
                
                # Basic validation against schema
                if self._validate_against_schema(result, schema):
                    return result
                else:
                    logger.warning("Generated JSON doesn't match schema, attempting correction")
                    return await self._correct_json_structure(result, schema)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                # Attempt to fix common JSON issues
                return await self._fix_json_response(response, schema)
                
        except Exception as e:
            logger.error(f"Error generating structured output: {e}")
            raise
    
    async def analyze_text(self, 
                          text: str,
                          analysis_type: str = "sentiment",
                          model: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze text for various properties
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (sentiment, emotion, topics, etc.)
            model: Model name
        
        Returns:
            Analysis results
        """
        try:
            analysis_prompts = {
                "sentiment": f"""
                Analyze the sentiment of this text: "{text}"
                
                Provide a JSON response with:
                - sentiment: "positive", "negative", or "neutral"
                - confidence: float between 0 and 1
                - reasoning: brief explanation
                """,
                
                "emotion": f"""
                Analyze the emotional tone of this text: "{text}"
                
                Provide a JSON response with:
                - primary_emotion: main emotion detected
                - emotions: array of emotions with confidence scores
                - intensity: float between 0 and 1
                """,
                
                "topics": f"""
                Extract the main topics from this text: "{text}"
                
                Provide a JSON response with:
                - topics: array of topic strings
                - keywords: array of important keywords
                - categories: array of relevant categories
                """,
                
                "style": f"""
                Analyze the writing style of this text: "{text}"
                
                Provide a JSON response with:
                - tone: formal/informal/conversational/etc
                - complexity: simple/moderate/complex
                - audience: target audience description
                - characteristics: array of style characteristics
                """
            }
            
            if analysis_type not in analysis_prompts:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            prompt = analysis_prompts[analysis_type]
            
            schema = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            result = await self.generate_structured_output(prompt, schema, model)
            result["analysis_type"] = analysis_type
            result["analyzed_text"] = text
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            raise
    
    async def _generate_openai(self, 
                              messages: List[Dict[str, str]], 
                              model: str,
                              max_tokens: int, 
                              temperature: float,
                              **kwargs) -> str:
        """Generate text using OpenAI API"""
        try:
            client = self._clients.get(LLMProvider.OPENAI)
            if not client:
                raise ValueError("OpenAI client not initialized")
            
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error with OpenAI generation: {e}")
            raise
    
    async def _generate_anthropic(self, 
                                 messages: List[Dict[str, str]], 
                                 model: str,
                                 max_tokens: int, 
                                 temperature: float,
                                 **kwargs) -> str:
        """Generate text using Anthropic API"""
        try:
            client = self._clients.get(LLMProvider.ANTHROPIC)
            if not client:
                raise ValueError("Anthropic client not initialized")
            
            # Convert messages format for Anthropic
            system_msg = None
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_messages.append(msg)
            
            # Anthropic API call
            response = await client.messages.create(
                model=model,
                system=system_msg,
                messages=user_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error with Anthropic generation: {e}")
            raise
    
    async def _generate_local(self, 
                             messages: List[Dict[str, str]], 
                             model: str,
                             max_tokens: int, 
                             temperature: float,
                             **kwargs) -> str:
        """Generate text using local model"""
        try:
            # This would integrate with local model serving
            # For now, return placeholder
            logger.warning("Local model generation not implemented")
            return "Local model response placeholder"
            
        except Exception as e:
            logger.error(f"Error with local generation: {e}")
            raise
    
    async def _stream_openai(self, 
                            messages: List[Dict[str, str]], 
                            model: str,
                            max_tokens: int, 
                            temperature: float,
                            **kwargs) -> AsyncGenerator[str, None]:
        """Stream text from OpenAI"""
        try:
            client = self._clients.get(LLMProvider.OPENAI)
            if not client:
                raise ValueError("OpenAI client not initialized")
            
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error streaming from OpenAI: {e}")
            raise
    
    async def _stream_anthropic(self, 
                               messages: List[Dict[str, str]], 
                               model: str,
                               max_tokens: int, 
                               temperature: float,
                               **kwargs) -> AsyncGenerator[str, None]:
        """Stream text from Anthropic"""
        try:
            # Anthropic streaming implementation would go here
            logger.warning("Anthropic streaming not implemented")
            
            # Fallback to regular generation
            result = await self._generate_anthropic(messages, model, max_tokens, temperature, **kwargs)
            yield result
            
        except Exception as e:
            logger.error(f"Error streaming from Anthropic: {e}")
            raise
    
    def _resolve_model(self, model: Optional[str]) -> tuple[LLMProvider, str]:
        """Resolve model name to provider and model"""
        if not model:
            # Use default
            provider = self.default_provider
            model_name = self.providers_config[provider].get("default_model", "gpt-3.5-turbo")
        else:
            # Find provider for model
            provider = self.model_mappings.get(model)
            if not provider:
                # Assume it's for default provider
                provider = self.default_provider
            model_name = model
        
        return provider, model_name
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Basic JSON schema validation"""
        try:
            # Very basic validation - in production, use jsonschema library
            required_fields = schema.get("required", [])
            
            for field in required_fields:
                if field not in data:
                    return False
            
            return True
            
        except Exception:
            return False
    
    async def _correct_json_structure(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to correct JSON structure"""
        try:
            # Add missing required fields with default values
            required_fields = schema.get("required", [])
            properties = schema.get("properties", {})
            
            for field in required_fields:
                if field not in data:
                    # Add default value based on type
                    field_schema = properties.get(field, {})
                    field_type = field_schema.get("type", "string")
                    
                    if field_type == "string":
                        data[field] = ""
                    elif field_type == "number":
                        data[field] = 0
                    elif field_type == "array":
                        data[field] = []
                    elif field_type == "object":
                        data[field] = {}
                    elif field_type == "boolean":
                        data[field] = False
            
            return data
            
        except Exception as e:
            logger.error(f"Error correcting JSON structure: {e}")
            return data
    
    async def _fix_json_response(self, response: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to fix malformed JSON"""
        try:
            # Common fixes
            fixed_response = response.strip()
            
            # Remove markdown code blocks
            if fixed_response.startswith("```"):
                lines = fixed_response.split('\n')
                fixed_response = '\n'.join(lines[1:-1])
            
            # Try parsing again
            try:
                return json.loads(fixed_response)
            except json.JSONDecodeError:
                # Return minimal valid structure
                return {"error": "Failed to parse JSON", "raw_response": response}
                
        except Exception as e:
            logger.error(f"Error fixing JSON response: {e}")
            return {"error": str(e), "raw_response": response}
    
    async def _initialize_clients(self):
        """Initialize LLM clients"""
        try:
            # Initialize OpenAI client
            openai_config = self.providers_config[LLMProvider.OPENAI]
            if openai_config.get("enabled", False):
                import openai
                self._clients[LLMProvider.OPENAI] = openai.AsyncOpenAI(
                    api_key=openai_config.get("api_key"),
                    base_url=openai_config.get("base_url")
                )
                logger.info("OpenAI client initialized")
            
            # Initialize Anthropic client
            anthropic_config = self.providers_config[LLMProvider.ANTHROPIC]
            if anthropic_config.get("enabled", False):
                try:
                    import anthropic
                    self._clients[LLMProvider.ANTHROPIC] = anthropic.AsyncAnthropic(
                        api_key=anthropic_config.get("api_key")
                    )
                    logger.info("Anthropic client initialized")
                except ImportError:
                    logger.warning("Anthropic library not available")
            
            # Initialize local client
            local_config = self.providers_config[LLMProvider.LOCAL]
            if local_config.get("enabled", False):
                # Local model initialization would go here
                logger.info("Local model configuration found")
            
            logger.info("LLM interface initialized")
            
        except Exception as e:
            logger.error(f"Error initializing LLM clients: {e}")
    
    async def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models by provider"""
        models = {}
        
        for provider, config in self.providers_config.items():
            if config.get("enabled", False):
                if provider == LLMProvider.OPENAI:
                    models[provider.value] = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
                elif provider == LLMProvider.ANTHROPIC:
                    models[provider.value] = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
                elif provider == LLMProvider.LOCAL:
                    models[provider.value] = config.get("available_models", [])
        
        return models
    
    async def estimate_cost(self, 
                           prompt: str, 
                           model: Optional[str] = None,
                           max_tokens: int = 1000) -> Dict[str, Any]:
        """Estimate cost for text generation"""
        try:
            provider, model_name = self._resolve_model(model)
            
            # Token estimation (very rough)
            input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            output_tokens = max_tokens
            
            # Cost estimates (per 1K tokens, approximate)
            cost_per_1k = {
                ("openai", "gpt-4"): {"input": 0.03, "output": 0.06},
                ("openai", "gpt-3.5-turbo"): {"input": 0.0015, "output": 0.002},
                ("anthropic", "claude-3-opus"): {"input": 0.015, "output": 0.075},
                ("anthropic", "claude-3-sonnet"): {"input": 0.003, "output": 0.015},
            }
            
            rates = cost_per_1k.get((provider.value, model_name), {"input": 0.001, "output": 0.002})
            
            estimated_cost = (
                (input_tokens / 1000) * rates["input"] + 
                (output_tokens / 1000) * rates["output"]
            )
            
            return {
                "provider": provider.value,
                "model": model_name,
                "estimated_input_tokens": int(input_tokens),
                "estimated_output_tokens": output_tokens,
                "estimated_cost_usd": round(estimated_cost, 4)
            }
            
        except Exception as e:
            logger.error(f"Error estimating cost: {e}")
            return {"error": str(e)}
