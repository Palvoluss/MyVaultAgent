import os
import hashlib
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import logging
from ollama_client import OllamaClient
import numpy as np
from scipy.special import softmax
import json

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    def __init__(self, vault_path: str):
        """Inizializza il gestore degli embedding."""
        self.vault_path = vault_path
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chromadb")
        
        # Inizializza il client Ollama
        self.ollama = OllamaClient(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "deepseek-r1"),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
        )
        
        # Verifica che Ollama sia in esecuzione
        if not self.ollama.health_check():
            raise RuntimeError("Ollama non è in esecuzione. Assicurati che il server sia attivo.")
        
        # Inizializza ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_dir,
            anonymized_telemetry=False
        ))
        
        # Crea o ottieni la collezione
        self.collection = self.client.get_or_create_collection(
            name="markdown_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Carica o crea il file di stato dell'indicizzazione
        self.index_state_file = os.path.join(self.persist_dir, "index_state.json")
        self.index_state = self._load_index_state()
        
        logger.info("EmbeddingsManager inizializzato con successo")

    def _load_index_state(self) -> Dict[str, Dict[str, Any]]:
        """Carica lo stato dell'indicizzazione da file."""
        try:
            if os.path.exists(self.index_state_file):
                with open(self.index_state_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Errore nel caricamento dello stato dell'indicizzazione: {str(e)}")
            return {}

    def _save_index_state(self):
        """Salva lo stato dell'indicizzazione su file."""
        try:
            # Assicurati che la directory esista
            os.makedirs(self.persist_dir, exist_ok=True)
            
            with open(self.index_state_file, 'w') as f:
                json.dump(self.index_state, f)
        except Exception as e:
            logger.error(f"Errore nel salvataggio dello stato dell'indicizzazione: {str(e)}")

    def _get_file_hash(self, file_path: str) -> str:
        """Calcola l'hash MD5 del contenuto del file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Ottiene le informazioni di un file."""
        try:
            stat = os.stat(file_path)
            return {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "hash": self._get_file_hash(file_path)
            }
        except Exception as e:
            logger.error(f"Errore nell'ottenimento delle informazioni del file {file_path}: {str(e)}")
            return None

    def is_file_indexed(self, file_path: str) -> bool:
        """Verifica se un file è già stato indicizzato e non è stato modificato."""
        try:
            # Ottieni le informazioni attuali del file
            current_info = self._get_file_info(file_path)
            if not current_info:
                return False

            # Controlla se il file è nello stato dell'indicizzazione
            if file_path in self.index_state:
                stored_info = self.index_state[file_path]
                # Verifica che hash, dimensione e data di modifica corrispondano
                return (stored_info["hash"] == current_info["hash"] and
                        stored_info["size"] == current_info["size"] and
                        stored_info["mtime"] == current_info["mtime"])
            
            return False
        except Exception as e:
            logger.error(f"Errore nel controllo dell'indicizzazione del file {file_path}: {str(e)}")
            return False

    def add_or_update_document(self, file_path: str, chunks: List[str]) -> None:
        """Aggiunge o aggiorna un documento nella collezione."""
        try:
            # Ottieni le informazioni del file
            file_info = self._get_file_info(file_path)
            if not file_info:
                raise ValueError(f"Impossibile ottenere le informazioni del file {file_path}")
            
            # Verifica se il file è già indicizzato e non è stato modificato
            if self.is_file_indexed(file_path):
                logger.info(f"Il file {file_path} è già indicizzato e non è stato modificato")
                return
            
            # Rimuovi le vecchie versioni del documento
            self.collection.delete(
                where={"file_path": file_path}
            )
            
            # Genera gli embedding per i chunk
            embeddings = self.ollama.get_embeddings_batch(chunks)
            
            # Prepara i metadati
            metadatas = []
            for i in range(len(chunks)):
                metadata = {
                    "file_path": file_path,
                    "file_hash": file_info["hash"],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                metadatas.append(metadata)
            
            # Aggiungi i chunk alla collezione
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=[f"{file_path}_{i}" for i in range(len(chunks))]
            )
            
            # Aggiorna lo stato dell'indicizzazione
            self.index_state[file_path] = file_info
            self._save_index_state()
            
            logger.info(f"Documento {file_path} indicizzato con successo")
        except Exception as e:
            logger.error(f"Errore nell'indicizzazione del documento {file_path}: {str(e)}")
            raise

    def _calculate_relevance(self, distances: List[float]) -> List[float]:
        """Calcola la rilevanza normalizzata usando softmax sulle distanze."""
        # Converti le distanze in array numpy
        distances = np.array(distances)
        
        # Normalizza le distanze (più piccole = più rilevanti)
        # Usa softmax per ottenere una distribuzione di probabilità
        # Moltiplica per -1 perché vogliamo che le distanze più piccole abbiano probabilità più alte
        relevance = softmax(-distances)
        
        # Normalizza tra 0 e 1
        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min())
        
        return relevance.tolist()

    def find_similar_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Trova i documenti più simili alla query."""
        try:
            # Genera l'embedding per la query
            query_embedding = self.ollama.get_embeddings(query)
            
            # Cerca i documenti più simili
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Calcola la rilevanza normalizzata
            relevance_scores = self._calculate_relevance(results["distances"][0])
            
            # Formatta i risultati
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "relevance": relevance_scores[i]
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Errore nella ricerca di documenti simili: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, int]:
        """Ottiene le statistiche della collezione."""
        try:
            # Ottieni tutti i metadati
            results = self.collection.get(include=["metadatas"])
            metadatas = results["metadatas"]
            
            # Calcola le statistiche
            unique_files = len(set(m["file_path"] for m in metadatas))
            total_chunks = len(metadatas)
            
            return {
                "unique_files": unique_files,
                "total_chunks": total_chunks
            }
        except Exception as e:
            logger.error(f"Errore nel recupero delle statistiche: {str(e)}")
            raise 