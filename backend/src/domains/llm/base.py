from abc import ABC, abstractmethod
from typing import List

class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._client = None
        self._is_connected = False
    
    def ensure_connection(self):
        if not self._is_connected or self._client is None:
            self._connect()
            self._is_connected = True
        
    def doc_system_prompt(self) -> str:
        return f"""
        You are a medical expert. You are given a document and you need to answer the question based on the document.
        """
    
    def parser_system_prompt(self) -> str:
        return f"""
        You are a medical expert. You are given a document and you need to answer the question based on the document.
       """
    def summarize_system_prompt(self) -> str:
        return f"""
        You are a medical expert. You are given a document and you need to summarize the document.
        """
    def get_system_prompt(self, type: str = "doc") -> str:
        if type == "doc":
            return self.doc_system_prompt()
        elif type == "parser":
            return self.parser_system_prompt()
        elif type == "summarize":
            return self.summarize_system_prompt()
        else:
            raise ValueError(f"Invalid type: {type}")
        
    def clean_response(response: str) -> str:
        raw_text = response.text.strip()
        if raw_text.startswith("```") and raw_text.endswith("```"):
            raw_text = "\n".join(raw_text.split("\n")[1:-1])
        return raw_text
    
    @abstractmethod
    def _connect(self):
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def llm_embeddings(self, text: str) -> List[float]:
        pass
