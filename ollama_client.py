import requests
import json
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, host: str, model: str, embedding_model: str):
        self.host = host
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = f"{host}/api"

    def generate(self, prompt: str, system: str = None) -> str:
        """Genera una risposta usando il modello di chat."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if system:
                payload["system"] = system

            response = requests.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status()
            
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Errore nella generazione con Ollama: {str(e)}")
            raise

    def get_embeddings(self, text: str) -> List[float]:
        """Ottiene gli embedding per un testo usando il modello di embedding."""
        try:
            payload = {
                "model": self.embedding_model,
                "prompt": text
            }
            
            response = requests.post(f"{self.base_url}/embeddings", json=payload)
            response.raise_for_status()
            
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Errore nell'ottenimento degli embedding: {str(e)}")
            raise

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Ottiene gli embedding per una lista di testi."""
        return [self.get_embeddings(text) for text in texts]

    def health_check(self) -> bool:
        """Verifica che il server Ollama sia in esecuzione."""
        try:
            response = requests.get(f"{self.base_url}/version")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Errore nel controllo dello stato di Ollama: {str(e)}")
            return False 