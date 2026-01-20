"""LLM client wrapper for OpenAI API calls."""
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from src.utils.logger import setup_logger
from src.utils.config import config

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class LLMClient:
    """Wrapper for OpenAI API calls with retry and error handling."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.openai_api_key
        self.model = model or config.llm_model
        self.logger = setup_logger("LLMClient")
        
        if not OPENAI_AVAILABLE:
            self.logger.warning(
                "OpenAI package not installed. Install with: pip install openai\n"
                "LLM features will be disabled."
            )
            self.client = None
        elif not self.api_key:
            self.logger.warning("No OpenAI API key found. LLM features will be disabled.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            self.logger.info(f"OpenAI client initialized (model: {self.model})")
    
    @property
    def is_available(self) -> bool:
        """Check if LLM client is available."""
        return self.client is not None
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        json_response: bool = False
    ) -> Optional[str]:
        """
        Get completion from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Response randomness (0.0 = deterministic)
            max_tokens: Maximum response length
            json_response: Request JSON format
        
        Returns:
            Response text or None if failed
        """
        if not self.client:
            return None
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if json_response:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            
            return response.choices[0].message.content
        
        except Exception as e:
            self.logger.error(f"LLM completion failed: {e}")
            return None
    
    def complete_with_images(
        self,
        prompt: str,
        image_paths: List[Path],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Get completion with image inputs (vision).
        
        Args:
            prompt: User prompt
            image_paths: Paths to images
            system_prompt: Optional system prompt
            temperature: Response randomness
            max_tokens: Maximum response length
        
        Returns:
            Response text or None if failed
        """
        if not self.client:
            return None
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Build content with images
        content = [{"type": "text", "text": prompt}]
        
        for image_path in image_paths:
            if image_path.exists():
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                })
        
        messages.append({"role": "user", "content": content})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            self.logger.error(f"Vision completion failed: {e}")
            return None
    
    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """
        Get JSON completion from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Response randomness
        
        Returns:
            Parsed JSON dict or None if failed
        """
        response = self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_response=True
        )
        
        if not response:
            return None
        
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            return json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Response was: {response}")
            return None
    
    def transcribe_audio(
        self,
        audio_path: Path,
        language: str = "en"
    ) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio file using OpenAI's transcription API.
        
        Supports both whisper-1 and gpt-4o-transcribe models.
        
        Args:
            audio_path: Path to audio file
            language: Language code
        
        Returns:
            Transcription result with text and segments
        """
        if not self.client:
            return None
        
        model = config.whisper_model
        self.logger.info(f"Transcribing with model: {model}")
        
        try:
            with open(audio_path, "rb") as f:
                # Different API call based on model
                if model == "gpt-4o-transcribe":
                    # gpt-4o-transcribe API
                    response = self.client.audio.transcriptions.create(
                        model=model,
                        file=f,
                    )
                    
                    # gpt-4o-transcribe returns simpler response
                    return {
                        "text": response.text if hasattr(response, 'text') else str(response),
                        "segments": [],  # May not have segments
                        "language": language,
                        "duration": None
                    }
                else:
                    # whisper-1 API with verbose response
                    response = self.client.audio.transcriptions.create(
                        model=model,
                        file=f,
                        language=language,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"]
                    )
                    
                    return {
                        "text": response.text,
                        "segments": [
                            {
                                "start": s.start,
                                "end": s.end,
                                "text": s.text
                            }
                            for s in (response.segments or [])
                        ],
                        "language": response.language,
                        "duration": response.duration
                    }
        
        except Exception as e:
            self.logger.error(f"Audio transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return None


# Global LLM client instance
llm_client = LLMClient()