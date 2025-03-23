from typing import List, Dict
from llama_cpp import Llama
from embeddings import EmbeddingsManager
import os

class NoteAnalyzer:
    def __init__(self, embeddings_manager: EmbeddingsManager):
        """Initialize the note analyzer."""
        self.embeddings_manager = embeddings_manager
        # Initialize Llama model
        model_path = os.getenv("LLAMA_MODEL_PATH", "models/llama-2-7b-chat.gguf")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window
            n_threads=os.cpu_count(),  # Use all CPU cores
        )

    def analyze_connections(self, new_note_path: str) -> List[Dict]:
        """Analyze connections between a new note and existing notes."""
        # Read the new note's content
        with open(new_note_path, 'r', encoding='utf-8') as f:
            new_note_content = f.read()

        # Extract key concepts from the new note
        key_concepts = self._extract_key_concepts(new_note_content)
        
        # Find similar documents based on the entire content
        similar_docs = self.embeddings_manager.find_similar_documents(new_note_content, n_results=10)
        
        # Analyze connections using Llama
        connections = []
        for doc in similar_docs:
            if doc["metadata"]["file_path"] != new_note_path:  # Avoid self-references
                connection = self._analyze_connection(new_note_content, doc["document"], key_concepts)
                if connection:
                    connections.append({
                        "file_path": doc["metadata"]["file_path"],
                        "relationship": connection["relationship"],
                        "practical_applications": connection["practical_applications"],
                        "key_concepts": connection["shared_concepts"],
                        "similarity_score": 1 - doc["distance"]
                    })

        return connections

    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from the content using Llama."""
        prompt = f"""Analyze this note and extract the key practical concepts, tools, methods, or strategies mentioned:

Content:
{content[:1500]}...

List the key concepts, one per line, focusing on practical and actionable items."""

        response = self.llm(
            prompt,
            max_tokens=200,
            temperature=0.3,
            stop=["Content:", "\n\n"]
        )
        
        concepts = [concept.strip() for concept in response['choices'][0]['text'].split('\n') if concept.strip()]
        return concepts

    def _analyze_connection(self, source_content: str, target_content: str, key_concepts: List[str]) -> Dict:
        """Use Llama to analyze the connection between two notes with focus on practical applications."""
        prompt = f"""Analyze the relationship between these two notes, focusing on practical knowledge and applications.

Note 1 Key Concepts:
{chr(10).join(key_concepts)}

Note 1:
{source_content[:800]}...

Note 2:
{target_content[:800]}...

Provide a detailed analysis in the following format:

1. Relationship: [Explain how these notes are connected conceptually]
2. Shared Concepts: [List the specific concepts, tools, or methods that appear in both notes]
3. Practical Applications: [Describe how the knowledge from both notes could be combined or applied practically]"""

        response = self.llm(
            prompt,
            max_tokens=500,
            temperature=0.7,
            stop=["Note 1:", "Note 2:"]
        )

        text = response['choices'][0]['text']
        
        # Parse the response into sections
        sections = text.split('\n')
        relationship = ""
        shared_concepts = []
        practical_applications = ""
        
        current_section = ""
        for line in sections:
            if line.startswith("1. Relationship:"):
                current_section = "relationship"
                relationship = line.replace("1. Relationship:", "").strip()
            elif line.startswith("2. Shared Concepts:"):
                current_section = "concepts"
                shared_concepts = []
            elif line.startswith("3. Practical Applications:"):
                current_section = "applications"
                practical_applications = line.replace("3. Practical Applications:", "").strip()
            elif line.strip():
                if current_section == "relationship":
                    relationship += " " + line.strip()
                elif current_section == "concepts":
                    shared_concepts.append(line.strip())
                elif current_section == "applications":
                    practical_applications += " " + line.strip()

        return {
            "relationship": relationship,
            "shared_concepts": shared_concepts,
            "practical_applications": practical_applications
        }

    def suggest_backlinks(self, note_path: str) -> List[Dict]:
        """Suggest backlinks for a note based on content analysis."""
        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read()

        similar_docs = self.embeddings_manager.find_similar_documents(content)
        
        backlinks = []
        for doc in similar_docs:
            if doc["metadata"]["file_path"] != note_path:  # Avoid self-references
                backlinks.append({
                    "file_path": doc["metadata"]["file_path"],
                    "relevance_score": 1 - doc["distance"],
                    "excerpt": doc["document"][:200] + "..."  # Preview of the related content
                })

        return backlinks 