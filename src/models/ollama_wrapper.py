"""Wrapper for Ollama models."""

import ollama
from typing import Dict, Any, Optional
from loguru import logger
import time


class OllamaModel:
    """Wrapper for Ollama models with timing and error handling."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = ollama.Client()
        
        # Verify model is available
        try:
            models = self.client.list()
            available = [m['name'] for m in models['models']]
            if model_name not in available:
                logger.warning(f"Model {model_name} not found in: {available}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """Generate response from Ollama model."""
        
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )
            
            elapsed_time = time.time() - start_time
            
            return {
                "response": response['message']['content'],
                "model": self.model_name,
                "time_seconds": elapsed_time,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error generating response with {self.model_name}: {e}")
            
            return {
                "response": "",
                "model": self.model_name,
                "time_seconds": elapsed_time,
                "success": False,
                "error": str(e)
            }


if __name__ == "__main__":
    # Test
    model = OllamaModel("llama3.2:3b")
    result = model.generate("What is 2+2?")
    print(result)