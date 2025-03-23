import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
from embeddings import EmbeddingsManager
import utils
from threading import Thread
from api import start_api
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
import threading

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configurazione
VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")
if not VAULT_PATH:
    raise ValueError("OBSIDIAN_VAULT_PATH non è stato impostato nel file .env")

DEBOUNCE_TIME = int(os.getenv("DEBOUNCE_TIME", "2"))
INDEX_ALL_ON_START = os.getenv("INDEX_ALL_ON_START", "true").lower() == "true"
START_WEB_VIEWER = os.getenv("START_WEB_VIEWER", "true").lower() == "true"

# Carica i percorsi da escludere
EXCLUDED_PATHS = os.getenv("EXCLUDED_PATHS", "").split(",")
EXCLUDED_PATHS = [path.strip() for path in EXCLUDED_PATHS if path.strip()]
logger.info(f"Percorsi esclusi dall'indicizzazione: {EXCLUDED_PATHS}")

# Inizializza il text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

def should_exclude_path(path: str) -> bool:
    """Verifica se un percorso deve essere escluso dall'indicizzazione."""
    # Converti il percorso in relativo rispetto al vault
    rel_path = os.path.relpath(path, VAULT_PATH)
    # Verifica se il percorso inizia con uno dei percorsi esclusi
    return any(rel_path.startswith(excluded) for excluded in EXCLUDED_PATHS)

class MarkdownHandler(FileSystemEventHandler):
    def __init__(self, embeddings_manager):
        self.embeddings_manager = embeddings_manager
        self.last_modified = {}
        self.lock = threading.Lock()

    def on_modified(self, event):
        if event.is_directory:
            return
        
        if not event.src_path.endswith('.md'):
            return
            
        if should_exclude_path(event.src_path):
            logger.debug(f"File escluso dall'indicizzazione: {event.src_path}")
            return
            
        with self.lock:
            current_time = time.time()
            last_time = self.last_modified.get(event.src_path, 0)
            
            if current_time - last_time < DEBOUNCE_TIME:
                return
                
            self.last_modified[event.src_path] = current_time
            
            try:
                logger.info(f"📝 File modificato: {os.path.basename(event.src_path)}")
                content = utils.read_markdown_file(event.src_path)
                text = utils.markdown_to_text(content)
                chunks = text_splitter.split_text(text)
                self.embeddings_manager.add_or_update_document(event.src_path, chunks)
            except Exception as e:
                logger.error(f"Errore nell'elaborazione del file {event.src_path}: {str(e)}")

    def on_created(self, event):
        if event.is_directory:
            return
            
        if not event.src_path.endswith('.md'):
            return
            
        if should_exclude_path(event.src_path):
            logger.debug(f"File escluso dall'indicizzazione: {event.src_path}")
            return
            
        try:
            logger.info(f"✨ Nuovo file rilevato: {os.path.basename(event.src_path)}")
            content = utils.read_markdown_file(event.src_path)
            text = utils.markdown_to_text(content)
            chunks = text_splitter.split_text(text)
            self.embeddings_manager.add_or_update_document(event.src_path, chunks)
        except Exception as e:
            logger.error(f"Errore nell'elaborazione del nuovo file {event.src_path}: {str(e)}")

    def on_deleted(self, event):
        if event.is_directory:
            return
            
        if not event.src_path.endswith('.md'):
            return
            
        if should_exclude_path(event.src_path):
            logger.debug(f"File escluso dall'indicizzazione: {event.src_path}")
            return
            
        try:
            logger.info(f"🗑️ File eliminato: {os.path.basename(event.src_path)}")
            self.embeddings_manager.collection.delete(
                where={"file_path": event.src_path}
            )
        except Exception as e:
            logger.error(f"Errore nell'eliminazione del file {event.src_path}: {str(e)}")

def index_all_files():
    """Indicizza tutti i file markdown nella directory."""
    try:
        total_files = 0
        processed_files = 0
        excluded_files = 0
        
        # Conta il numero totale di file
        for root, _, files in os.walk(VAULT_PATH):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    if should_exclude_path(file_path):
                        excluded_files += 1
                    else:
                        total_files += 1
        
        if total_files == 0:
            logger.warning("❌ Nessun file markdown trovato nel vault.")
            return
            
        logger.info(f"📚 Trovati {total_files} file markdown da processare (esclusi {excluded_files} file).")
        
        # Processa i file
        for root, _, files in os.walk(VAULT_PATH):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    if should_exclude_path(file_path):
                        continue
                        
                    try:
                        processed_files += 1
                        logger.info(f"⏳ Processando {processed_files}/{total_files}: {file}")
                        content = utils.read_markdown_file(file_path)
                        text = utils.markdown_to_text(content)
                        chunks = text_splitter.split_text(text)
                        embeddings_manager.add_or_update_document(file_path, chunks)
                    except Exception as e:
                        logger.error(f"Errore nell'indicizzazione del file {file_path}: {str(e)}")
        
        logger.info(f"✅ Indicizzazione completata! Processati {processed_files}/{total_files} file.")
    except Exception as e:
        logger.error(f"Errore durante l'indicizzazione dei file: {str(e)}")

def main():
    """Funzione principale."""
    try:
        # Inizializza il gestore degli embedding
        global embeddings_manager
        embeddings_manager = EmbeddingsManager(VAULT_PATH)
        
        # Indicizza tutti i file se richiesto
        if INDEX_ALL_ON_START:
            logger.info("🔄 Avvio indicizzazione completa...")
            index_all_files()
        
        # Configura l'observer per il file system
        event_handler = MarkdownHandler(embeddings_manager)
        observer = Observer()
        observer.schedule(event_handler, VAULT_PATH, recursive=True)
        
        # Avvia l'observer
        observer.start()
        logger.info(f"👀 Monitoraggio attivo sul vault: {VAULT_PATH}")
        
        # Avvia il web viewer se richiesto
        if START_WEB_VIEWER:
            logger.info("🌐 Avvio web viewer...")
            api_thread = threading.Thread(target=start_api)
            api_thread.daemon = True
            api_thread.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logger.info("👋 Arresto del programma...")
        
        observer.join()
        
    except Exception as e:
        logger.error(f"Errore critico: {str(e)}")
        raise

if __name__ == "__main__":
    main() 