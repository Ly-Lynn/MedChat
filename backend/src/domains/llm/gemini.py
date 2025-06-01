import google.generativeai as genai
from src.domains.llm.base import BaseLLM
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.api_key = settings.GEMINI_API_KEY
    
    def _connect(self):
        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(self.model_name)

    def generate_response(self, prompt: str, _type: str = "doc") -> str:
        self.ensure_connection()
        system_prompt = self.get_system_prompt(type=_type)
        try:
            response = self._client.generate_content(
                system_prompt + prompt
            )
            response_text = self.clean_response(response.text)
            return response_text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise e
    
