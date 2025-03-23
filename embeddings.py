import chromadb
from chromadb.config import Settings
import os
import hashlib
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import utils

class SentenceTransformerEmbedding:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input)
        return embeddings.tolist()

class EmbeddingsManager:
    def __init__(self, persist_directory: str):
        """Initialize the embeddings manager with a persistence directory."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="obsidian_notes",
            embedding_function=SentenceTransformerEmbedding()
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file content."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def is_file_indexed(self, file_path: str) -> bool:
        """Check if a file is already indexed and unchanged."""
        try:
            current_hash = self._get_file_hash(file_path)
            doc_id = utils.get_file_id(file_path)
            
            # Try to get the first chunk of the document
            results = self.collection.get(
                ids=[f"{doc_id}_0"],
                include=["metadatas"]
            )
            
            if results and results['metadatas']:
                stored_hash = results['metadatas'][0].get('file_hash')
                return stored_hash == current_hash
            
            return False
        except:
            return False

    def add_or_update_document(self, file_path: str) -> None:
        """Add or update a document in the vector database."""
        # Check if file needs to be reprocessed
        if self.is_file_indexed(file_path):
            print(f"📎 File già indicizzato e non modificato: {os.path.basename(file_path)}")
            return

        content = utils.read_markdown_file(file_path)
        text = utils.markdown_to_text(content)
        metadata = utils.extract_metadata(content)
        doc_id = utils.get_file_id(file_path)
        file_hash = self._get_file_hash(file_path)
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Remove existing document if it exists
        try:
            self.collection.delete(ids=[f"{doc_id}_{i}" for i in range(len(chunks))])
        except:
            pass
        
        # Add new chunks
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "file_path": file_path,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_hash": file_hash  # Add file hash to track changes
            }
            
            self.collection.add(
                ids=[f"{doc_id}_{i}"],
                documents=[chunk],
                metadatas=[chunk_metadata]
            )

    def find_similar_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Find similar documents to the query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return [
            {
                "document": doc,
                "metadata": metadata,
                "distance": distance
            }
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            all_metadata = self.collection.get(
                include=["metadatas"]
            )["metadatas"]
            
            unique_files = set()
            for metadata in all_metadata:
                if metadata and "file_path" in metadata:
                    unique_files.add(metadata["file_path"])
            
            return {
                "total_chunks": len(all_metadata),
                "unique_files": len(unique_files),
                "files": list(unique_files)
            }
        except:
            return {
                "total_chunks": 0,
                "unique_files": 0,
                "files": []
            }

    def delete_document(self, file_path: str) -> None:
        """Delete a document from the vector database."""
        doc_id = utils.get_file_id(file_path)
        try:
            self.collection.delete(ids=[doc_id])
        except:
            pass 