"""
Unified LLM client abstraction for SynthMaxxer
Supports multiple API providers: Anthropic, OpenAI, Grok, Gemini, OpenRouter, DeepSeek
"""
import json
import requests
from typing import Optional, List, Dict, Any


class LLMClient:
    """Unified client for multiple LLM API providers"""
    
    def __init__(
        self,
        api_key: str,
        api_type: str = "OpenAI Chat Completions",
        endpoint: Optional[str] = None,
        timeout: int = 300
    ):
        self.api_key = api_key
        self.api_type = api_type
        self.timeout = timeout
        
        # Set default endpoints based on API type
        if endpoint:
            self.endpoint = endpoint
        else:
            self.endpoint = self._get_default_endpoint()
    
    def _get_default_endpoint(self) -> str:
        """Get default endpoint for the API type"""
        defaults = {
            "Anthropic Claude": "https://api.anthropic.com/v1/messages",
            "OpenAI Official": "https://api.openai.com/v1/chat/completions",
            "OpenAI Chat Completions": "https://api.openai.com/v1/chat/completions",
            "OpenAI Text Completions": "https://api.openai.com/v1/completions",
            "Grok (xAI)": "https://api.x.ai/v1/chat/completions",
            "Gemini (Google)": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            "OpenRouter": "https://openrouter.ai/api/v1/chat/completions",
            "DeepSeek": "https://api.deepseek.com/v1/chat/completions",
        }
        return defaults.get(self.api_type, "https://api.openai.com/v1/chat/completions")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for the API request"""
        if self.api_type == "Anthropic Claude":
            return {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
        elif self.api_type == "Gemini (Google)":
            return {
                "Content-Type": "application/json"
            }
        elif self.api_type == "OpenRouter":
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/ShareGPT-Formaxxing",
                "X-Title": "SynthMaxxer"
            }
        else:
            # OpenAI-compatible APIs
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
    
    def chat_completion(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """
        Send a chat completion request and return the response text.
        
        Args:
            model: Model name to use
            system_prompt: System message/instructions
            user_message: User's message/prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            The assistant's response text
        """
        headers = self._get_headers()
        
        if self.api_type == "Anthropic Claude":
            return self._anthropic_completion(model, system_prompt, user_message, temperature, max_tokens, headers)
        elif self.api_type == "Gemini (Google)":
            return self._gemini_completion(model, system_prompt, user_message, temperature, max_tokens, headers)
        elif self.api_type == "OpenAI Text Completions":
            return self._text_completion(model, system_prompt, user_message, temperature, max_tokens, headers)
        else:
            # OpenAI-compatible chat completions (OpenAI, Grok, OpenRouter, DeepSeek)
            return self._openai_chat_completion(model, system_prompt, user_message, temperature, max_tokens, headers)
    
    def _anthropic_completion(
        self, model: str, system_prompt: str, user_message: str,
        temperature: float, max_tokens: int, headers: Dict[str, str]
    ) -> str:
        """Handle Anthropic Claude API"""
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_message}
            ]
        }
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract text from Anthropic response
        content = result.get("content", [])
        if content and isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    return block.get("text", "")
        return ""
    
    def _gemini_completion(
        self, model: str, system_prompt: str, user_message: str,
        temperature: float, max_tokens: int, headers: Dict[str, str]
    ) -> str:
        """Handle Google Gemini API"""
        # Update endpoint with model name and API key
        endpoint = self.endpoint
        if "{model}" in endpoint or "gemini-" not in endpoint:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        if "?key=" not in endpoint:
            endpoint = f"{endpoint}?key={self.api_key}"
        
        # Build contents with system instruction
        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System instructions: {system_prompt}"}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "I understand and will follow these instructions."}]
            })
        
        contents.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })
        
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        response = requests.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract text from Gemini response
        candidates = result.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return ""
    
    def _openai_chat_completion(
        self, model: str, system_prompt: str, user_message: str,
        temperature: float, max_tokens: int, headers: Dict[str, str]
    ) -> str:
        """Handle OpenAI-compatible chat completion APIs"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract text from OpenAI response
        choices = result.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")
        return ""
    
    def _text_completion(
        self, model: str, system_prompt: str, user_message: str,
        temperature: float, max_tokens: int, headers: Dict[str, str]
    ) -> str:
        """Handle OpenAI text completion API (legacy)"""
        prompt = ""
        if system_prompt:
            prompt = f"System: {system_prompt}\n\n"
        prompt += f"User: {user_message}\n\nAssistant:"
        
        data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract text from completion response
        choices = result.get("choices", [])
        if choices:
            return choices[0].get("text", "")
        return ""


def create_client(
    api_key: str,
    api_type: str = "OpenAI Chat Completions",
    endpoint: Optional[str] = None,
    timeout: int = 300
) -> LLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        api_key: API key for the provider
        api_type: One of the supported API types
        endpoint: Custom endpoint URL (optional)
        timeout: Request timeout in seconds
        
    Returns:
        LLMClient instance
    """
    return LLMClient(api_key, api_type, endpoint, timeout)
