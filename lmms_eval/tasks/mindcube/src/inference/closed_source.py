"""
Closed Source Model Inference Engine with OpenAI Compatible API support.
"""

import base64
import json
from typing import List, Dict, Any, Optional
from PIL import Image
import io
from openai import OpenAI
from openai import AzureOpenAI
from .base import BaseInferenceEngine
from volcenginesdkarkruntime import Ark
import os


class ClosedSourceInferenceEngine(BaseInferenceEngine):
    """OpenAI Compatible API inference engine for closed source models."""
    
    def __init__(self, model_type: str, api_key: Optional[str] = None, 
                 base_url: str = "https://ark.cn-beijing.volces.com/api/v3", 
                 max_tokens: int = 2048, temperature: float = 0.7, **kwargs):
        """
        Initialize closed source inference engine with OpenAI compatible API.
        
        Args:
            model_type: Type of model ('gpt-4o', 'gpt-4-vision-preview', etc.)
            api_key: API key for the service
            base_url: Base URL for the API endpoint
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional configuration
        """
        super().__init__("", model_type, **kwargs)
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # # Initialize OpenAI client with custom base URL
        # self.client = AzureOpenAI(
        #     api_key=self.api_key,
        #     base_url=self.base_url,
        #     api_version="2023-07-01-preview",
        #     max_retries=kwargs.get("max_retries", 5),
        #     timeout=kwargs.get("timeout", 120.0)
        # )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=kwargs.get("max_retries", 5),
            timeout=kwargs.get("timeout", 120.0)
        )
        
    def load_model(self) -> None:
        """Load model (validate API connection)."""
        if not self.api_key:
            print("Warning: No API key provided")
            return
            
        print(f"Initializing {self.model_type} API connection to {self.base_url}")
        
        # Test API connection with a simple request
        try:
            response = self.client.chat.completions.create(
                model=self.model_type,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
            #     thinking={
            #     "type": "disabled" # 不使用深度思考能力,
            #     # "type": "enabled" # 使用深度思考能力
            #     # "type": "auto" # 模型自行判断是否使用深度思考能力
            # },
            )
            
            print(f"Successfully connected to {self.model_type} API")
                
        except Exception as e:
            print(f"API connection test failed: {str(e)}")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                max_size = 1024
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_str}"
                
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
            return ""
        
    def process_input(self, prompt: str, image_paths: List[str], **kwargs) -> Any:
        """
        Process input for OpenAI compatible API call.
        
        Args:
            prompt: Text prompt
            image_paths: List of image file paths
            **kwargs: Additional processing parameters
            
        Returns:
            Processed input ready for API call
        """
        # Prepare the message content
        content = []
        
        # Add text prompt
        if prompt:
            content.append({
                "type": "text",
                "text": prompt
            })
        
        # Add images
        for image_path in image_paths:
            base64_image = self.encode_image_to_base64(image_path)
            if base64_image:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image,
                        "detail": kwargs.get("image_detail", "high")
                    }
                })
        
        # Prepare the API payload
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Override parameters if provided in kwargs
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        
        payload = {
            "model": self.model_type,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add any additional parameters
        for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
            if key in kwargs:
                payload[key] = kwargs[key]
        
        return payload
        
    def generate_response(self, processed_input: Any, **kwargs) -> str:
        """
        Generate response via OpenAI compatible API.
        
        Args:
            processed_input: Payload for API call
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        try:
            # Extract parameters from processed_input
            model = processed_input["model"]
            messages = processed_input["messages"]
            max_tokens = processed_input.get("max_tokens", self.max_tokens)
            temperature = processed_input.get("temperature", self.temperature)
            
            # Prepare additional parameters
            chat_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # "thinking": {
            #     "type": "disabled"  # 不使用深度思考能力,
            #     # "type": "enabled"  # 使用深度思考能力
            #     # "type": "auto"  # 模型自行判断是否使用深度思考能力
            # },
            }
            
            # Add optional parameters if present
            for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
                if key in processed_input:
                    chat_params[key] = processed_input[key]
            
            # Make API request using OpenAI client
            response = self.client.chat.completions.create(**chat_params)
            
            # Extract the generated text
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.message and choice.message.content:
                    return choice.message.content.strip()
                else:
                    return "Error: Invalid response format"
            else:
                return "Error: No choices in response"
                
        except Exception as e:
            # OpenAI client handles retries and various error types automatically
            return f"Error: {str(e)}"
    
    @staticmethod
    def list_supported_models() -> List[str]:
        """List supported OpenAI compatible models."""
        return [
            'gpt-4o',
            'gpt-4o-mini', 
            'gpt-4-vision-preview',
            'gpt-4-turbo',
            'claude-3-5-sonnet-20241022',
            'claude-3-opus-20240229',
            'claude-3-haiku-20240307',
            "gemini-2.5-pro",
            'gemini-1.5-pro',
            'gemini-1.5-flash',
            'Qwen/Qwen2.5-VL-72B-Instruct',
            'doubao-1.5-vision-pro-250328',
            "doubao-seed-1-6-250615",
        ]
    
    def get_usage_info(self, response_data: Any) -> Dict[str, Any]:
        """
        Extract usage information from OpenAI API response.
        
        Args:
            response_data: OpenAI response object
            
        Returns:
            Usage information dictionary
        """
        usage_info = {}
        if hasattr(response_data, 'usage') and response_data.usage:
            usage = response_data.usage
            usage_info = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0)
            }
        return usage_info
    
    def generate_response_with_usage(self, processed_input: Any, **kwargs) -> tuple[str, Dict[str, Any]]:
        """
        Generate response and return usage information.
        
        Args:
            processed_input: Payload for API call
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (response_text, usage_info)
        """
        try:
            # Extract parameters from processed_input
            model = processed_input["model"]
            messages = processed_input["messages"]
            max_tokens = processed_input.get("max_tokens", self.max_tokens)
            temperature = processed_input.get("temperature", self.temperature)
            
            # Prepare additional parameters
            chat_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add optional parameters if present
            for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
                if key in processed_input:
                    chat_params[key] = processed_input[key]
            
            # Make API request using OpenAI client
            response = self.client.chat.completions.create(**chat_params)
            
            # Extract usage information
            usage_info = self.get_usage_info(response)
            
            # Extract the generated text
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.message and choice.message.content:
                    return choice.message.content.strip(), usage_info
                else:
                    return "Error: Invalid response format", {}
            else:
                return "Error: No choices in response", {}
                
        except Exception as e:
            return f"Error: {str(e)}", {} 