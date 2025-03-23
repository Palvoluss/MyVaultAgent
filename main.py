import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
from embeddings import EmbeddingsManager
from analyzer import NoteAnalyzer
import utils
from typing import List, Dict
from threading import Timer, Thread
from api import start_api

# Load environment variables
load_dotenv()

class DebounceTimer:
    def __init__(self, timeout, callback):
        self.timeout = timeout
        self.callback = callback
        self.timer = None

    def __call__(self, *args, **kwargs):
        if self.timer:
            self.timer.cancel()
        self.timer = Timer(self.timeout, self.callback, args, kwargs)
        self.timer.start()

class ObsidianVaultHandler(FileSystemEventHandler):
    def __init__(self, embeddings_manager: EmbeddingsManager, analyzer: NoteAnalyzer, debounce_seconds: float):
        self.embeddings_manager = embeddings_manager
        self.analyzer = analyzer
        self.processing_files = set()
        self.debounce_seconds = debounce_seconds
        self.debounced_process = {}

    def on_modified(self, event):
        if not event.is_directory and utils.is_markdown_file(event.src_path):
            if event.src_path not in self.debounced_process:
                self.debounced_process[event.src_path] = DebounceTimer(
                    self.debounce_seconds,
                    self._process_file
                )
            self.debounced_process[event.src_path](event.src_path)

    def on_created(self, event):
        if not event.is_directory and utils.is_markdown_file(event.src_path):
            if event.src_path not in self.debounced_process:
                self.debounced_process[event.src_path] = DebounceTimer(
                    self.debounce_seconds,
                    self._process_file
                )
            self.debounced_process[event.src_path](event.src_path)

    def _process_file(self, file_path: str):
        """Process a new or modified file."""
        # Avoid processing the same file multiple times
        if file_path in self.processing_files:
            return
        
        try:
            self.processing_files.add(file_path)
            print(f"\n⏳ Analisi del file: {os.path.basename(file_path)}")
            
            # Update the vector database
            self.embeddings_manager.add_or_update_document(file_path)
            
            # Analyze connections
            connections = self.analyzer.analyze_connections(file_path)
            
            # Print the analysis results
            self._print_analysis_results(file_path, connections)
            
        finally:
            self.processing_files.remove(file_path)
            if file_path in self.debounced_process:
                del self.debounced_process[file_path]

    def _print_analysis_results(self, file_path: str, connections: List[Dict]):
        """Print analysis results in a structured format."""
        print(f"\n📝 Analisi per: {os.path.basename(file_path)}")
        print("=" * 80)
        
        if not connections:
            print("❌ Nessuna connessione rilevante trovata con altre note.")
            print("=" * 80)
            return
        
        print(f"🔍 Trovate {len(connections)} connessioni rilevanti:\n")
        
        for i, conn in enumerate(connections, 1):
            relevance = conn['similarity_score'] * 100
            print(f"\n🔗 Connessione #{i} - {os.path.basename(conn['file_path'])} (Rilevanza: {relevance:.1f}%)")
            print("-" * 80)
            
            # Concetti condivisi
            print("📌 Concetti Chiave Condivisi:")
            for concept in conn['key_concepts']:
                print(f"  • {concept}")
            
            # Relazione
            print("\n🤝 Relazione:")
            print(f"  {conn['relationship']}")
            
            # Applicazioni pratiche
            print("\n💡 Applicazioni Pratiche:")
            print(f"  {conn['practical_applications']}")
            
            print("-" * 80)

def index_entire_vault(vault_path: str, embeddings_manager: EmbeddingsManager):
    """Index all markdown files in the vault."""
    print("\n🔄 Indicizzazione completa del vault...")
    total_files = 0
    processed_files = 0

    # Get all markdown files
    for root, _, files in os.walk(vault_path):
        for file in files:
            if utils.is_markdown_file(file):
                total_files += 1

    if total_files == 0:
        print("❌ Nessun file markdown trovato nel vault.")
        return

    print(f"📚 Trovati {total_files} file markdown da processare.")

    for root, _, files in os.walk(vault_path):
        for file in files:
            if utils.is_markdown_file(file):
                file_path = os.path.join(root, file)
                try:
                    print(f"\r⏳ Processando {processed_files + 1}/{total_files}: {file}", end="")
                    embeddings_manager.add_or_update_document(file_path)
                    processed_files += 1
                except Exception as e:
                    print(f"\n❌ Errore nel processare {file}: {str(e)}")

    print(f"\n✅ Indicizzazione completata! Processati {processed_files}/{total_files} file.")

def main():
    # Get configuration from environment variables
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    llama_model_path = os.getenv("LLAMA_MODEL_PATH")
    debounce_seconds = float(os.getenv("DEBOUNCE_SECONDS", "5.0"))
    should_index_all = os.getenv("INDEX_ALL_ON_START", "true").lower() == "true"
    start_web_viewer = os.getenv("START_WEB_VIEWER", "true").lower() == "true"
    
    if not vault_path:
        print("Error: Please set OBSIDIAN_VAULT_PATH in .env file")
        return
    
    if not llama_model_path or not os.path.exists(llama_model_path):
        print("Warning: LLAMA_MODEL_PATH not set or model not found. Using default path.")
    
    # Create data directory for vector database
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize components
    embeddings_manager = EmbeddingsManager(data_dir)
    analyzer = NoteAnalyzer(embeddings_manager)
    
    # Index entire vault if requested
    if should_index_all:
        index_entire_vault(vault_path, embeddings_manager)
    
    # Start web viewer in a separate thread if requested
    if start_web_viewer:
        web_thread = Thread(target=start_api, daemon=True)
        web_thread.start()
        print(f"\n🌐 Visualizzatore web avviato su http://localhost:8000")
    
    # Set up file system observer
    event_handler = ObsidianVaultHandler(embeddings_manager, analyzer, debounce_seconds)
    observer = Observer()
    observer.schedule(event_handler, vault_path, recursive=True)
    
    print(f"\n🚀 Monitoraggio del vault Obsidian avviato: {vault_path}")
    print(f"⏱️  Debounce impostato a {debounce_seconds} secondi")
    print("⌨️  Premi Ctrl+C per terminare")
    
    # Start monitoring
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main() 