from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from embeddings import EmbeddingsManager
import os
from typing import List, Dict
from pydantic import BaseModel
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Second Brain Vector DB Viewer")

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inizializzazione del database
data_dir = os.path.join(os.path.dirname(__file__), "data")
embeddings_manager = EmbeddingsManager(data_dir)

class SearchRequest(BaseModel):
    query: str
    n_results: int = 5

@app.get("/visualizer", response_class=HTMLResponse)
async def get_visualizer():
    """Pagina di visualizzazione 3D degli embedding."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Second Brain Vector DB - Visualizzatore 3D</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .plot-container { height: 800px; }
            .controls { margin: 20px 0; }
            input[type="text"] { width: 70%; padding: 10px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
            .error { color: red; margin: 10px 0; }
            .loading { display: none; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Visualizzatore 3D degli Embedding</h1>
            
            <div class="controls">
                <input type="text" id="search-input" placeholder="Inserisci una query per evidenziare i documenti più simili...">
                <button onclick="searchAndHighlight()">Cerca</button>
            </div>

            <div id="error" class="error"></div>
            <div id="loading" class="loading">Caricamento...</div>
            <div id="plot" class="plot-container"></div>
        </div>

        <script>
            let currentPlot = null;

            async function loadPlot() {
                try {
                    document.getElementById('loading').style.display = 'block';
                    document.getElementById('error').textContent = '';
                    
                    const response = await fetch('/plot3d');
                    if (!response.ok) {
                        throw new Error('Errore nel caricamento del plot');
                    }
                    
                    const plotData = await response.json();
                    if (!plotData.data || plotData.data.length === 0) {
                        throw new Error('Nessun dato disponibile per il plot');
                    }
                    
                    Plotly.newPlot('plot', plotData.data, plotData.layout);
                    currentPlot = plotData;
                } catch (error) {
                    document.getElementById('error').textContent = error.message;
                    console.error('Errore:', error);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }

            async function searchAndHighlight() {
                try {
                    const query = document.getElementById('search-input').value;
                    if (!query) {
                        document.getElementById('error').textContent = 'Inserisci una query di ricerca';
                        return;
                    }

                    document.getElementById('loading').style.display = 'block';
                    document.getElementById('error').textContent = '';
                    
                    const response = await fetch('/plot3d?query=' + encodeURIComponent(query));
                    if (!response.ok) {
                        throw new Error('Errore nella ricerca');
                    }
                    
                    const plotData = await response.json();
                    if (!plotData.data || plotData.data.length === 0) {
                        throw new Error('Nessun risultato trovato');
                    }
                    
                    Plotly.newPlot('plot', plotData.data, plotData.layout);
                    currentPlot = plotData;
                } catch (error) {
                    document.getElementById('error').textContent = error.message;
                    console.error('Errore:', error);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }

            // Gestione dell'invio con il tasto Enter
            document.getElementById('search-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    searchAndHighlight();
                }
            });

            // Carica il plot iniziale
            loadPlot();
        </script>
    </body>
    </html>
    """

@app.get("/plot3d")
async def get_plot3d(query: str = None):
    """Genera il plot 3D degli embedding."""
    try:
        # Ottieni tutti gli embedding dal database
        collection_data = embeddings_manager.collection.get(include=["embeddings", "documents", "metadatas"])
        
        # Verifica che tutti i dati necessari siano presenti
        if not collection_data or "embeddings" not in collection_data or not collection_data["embeddings"]:
            logger.warning("Nessun dato trovato nel database")
            return {
                "data": [],
                "layout": {
                    "title": "Nessun dato disponibile",
                    "annotations": [{
                        "text": "Nessun embedding trovato nel database",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 20}
                    }]
                }
            }

        # Converti in array numpy e verifica le dimensioni
        embeddings = np.array(collection_data["embeddings"])
        n_samples = len(embeddings)
        
        if n_samples == 0:
            logger.warning("Array embeddings vuoto")
            return {"data": [], "layout": {}}

        # Calcola la perplexity appropriata
        perplexity = min(30, max(5, n_samples // 10))
        
        try:
            # Riduci le dimensioni usando t-SNE con gestione errori
            tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
            reduced_vectors = tsne.fit_transform(embeddings)
        except Exception as tsne_error:
            logger.error(f"Errore durante la riduzione dimensionale: {str(tsne_error)}")
            return {"data": [], "layout": {"title": "Errore nella riduzione dimensionale"}}

        # Verifica che i vettori ridotti siano validi
        if reduced_vectors.shape[0] != n_samples or reduced_vectors.shape[1] != 3:
            logger.error(f"Dimensioni non valide dei vettori ridotti: {reduced_vectors.shape}")
            return {"data": [], "layout": {"title": "Errore nelle dimensioni dei vettori"}}

        # Crea il plot base
        try:
            base_trace = go.Scatter3d(
                x=reduced_vectors[:, 0].tolist(),
                y=reduced_vectors[:, 1].tolist(),
                z=reduced_vectors[:, 2].tolist(),
                mode='markers',
                marker=dict(
                    size=5,
                    color='lightgrey',
                    opacity=0.7
                ),
                text=[
                    f"File: {m.get('file_path', 'N/A')}<br>"
                    f"Chunk: {i+1}/{n_samples}"
                    for i, m in enumerate(collection_data["metadatas"])
                ],
                hoverinfo='text',
                name='Tutti i documenti'
            )
        except Exception as trace_error:
            logger.error(f"Errore nella creazione del trace base: {str(trace_error)}")
            return {"data": [], "layout": {"title": "Errore nella creazione del grafico"}}

        data = [base_trace]
        
        # Se c'è una query, evidenzia i documenti più simili
        if query and query.strip():
            logger.info(f"Ricerca per query: {query}")
            try:
                results = embeddings_manager.find_similar_documents(query, n_results=5)
                if not results:
                    logger.warning("Nessun risultato trovato per la query")
                    return {
                        "data": data,
                        "layout": go.Layout(
                            title='Visualizzazione 3D degli Embedding (nessun risultato trovato)',
                            scene=dict(
                                xaxis_title='Dimensione 1',
                                yaxis_title='Dimensione 2',
                                zaxis_title='Dimensione 3'
                            ),
                            margin=dict(l=0, r=0, b=0, t=30),
                            showlegend=True
                        )
                    }

                # Crea un dizionario per mappare i metadati agli indici
                metadata_to_index = {}
                for i, metadata in enumerate(collection_data["metadatas"]):
                    key = (
                        str(metadata.get('file_path', '')),
                        str(metadata.get('chunk_index', -1))
                    )
                    metadata_to_index[key] = i

                # Trova gli indici corrispondenti
                similar_indices = []
                for result in results:
                    if not result or 'metadata' not in result:
                        continue
                    key = (
                        str(result['metadata'].get('file_path', '')),
                        str(result['metadata'].get('chunk_index', -1))
                    )
                    if key in metadata_to_index:
                        idx = metadata_to_index[key]
                        if 0 <= idx < n_samples:
                            similar_indices.append(idx)

                if similar_indices:
                    # Crea il trace per i punti evidenziati
                    highlighted_trace = go.Scatter3d(
                        x=[reduced_vectors[i, 0] for i in similar_indices],
                        y=[reduced_vectors[i, 1] for i in similar_indices],
                        z=[reduced_vectors[i, 2] for i in similar_indices],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color='red',
                            opacity=1
                        ),
                        text=[
                            f"File: {collection_data['metadatas'][i].get('file_path', 'N/A')}<br>"
                            f"Rilevanza: {(1 - results[j]['distance']) * 100:.1f}%<br>"
                            f"Contenuto: {collection_data['documents'][i][:100]}..."
                            for j, i in enumerate(similar_indices)
                            if i < len(collection_data['documents'])
                        ],
                        hoverinfo='text',
                        name='Documenti simili'
                    )
                    data.append(highlighted_trace)
                else:
                    logger.warning("Nessun indice valido trovato per i documenti simili")

            except Exception as search_error:
                logger.error(f"Errore durante la ricerca: {str(search_error)}")
                # Continua con il plot base se la ricerca fallisce

        layout = go.Layout(
            title='Visualizzazione 3D degli Embedding',
            scene=dict(
                xaxis_title='Dimensione 1',
                yaxis_title='Dimensione 2',
                zaxis_title='Dimensione 3'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=True
        )

        return {"data": data, "layout": layout}
    except Exception as e:
        logger.error(f"Errore durante la generazione del plot 3D: {str(e)}")
        return {
            "data": [],
            "layout": {
                "title": "Errore nella generazione del plot",
                "annotations": [{
                    "text": str(e),
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 14}
                }]
            }
        }

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Pagina principale con interfaccia utente."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Second Brain Vector DB Viewer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .stats { background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .search-box { margin: 20px 0; }
            input[type="text"] { width: 70%; padding: 10px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
            .results { margin-top: 20px; }
            .result-item { background: #fff; padding: 15px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }
            .similarity { color: #28a745; }
            .nav { margin-bottom: 20px; }
            .nav a { 
                padding: 10px 20px; 
                background: #007bff; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
                margin-right: 10px; 
            }
            .error { color: red; margin: 10px 0; }
            .loading { display: none; margin: 10px 0; }
        </style>
        <script>
            async function loadStats() {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    document.getElementById('stats').innerHTML = `
                        <h3>Statistiche Database</h3>
                        <p>File Unici: ${stats.unique_files}</p>
                        <p>Chunks Totali: ${stats.total_chunks}</p>
                    `;
                } catch (error) {
                    console.error('Errore nel caricamento delle statistiche:', error);
                }
            }

            async function search() {
                try {
                    const query = document.getElementById('search-input').value;
                    if (!query) {
                        document.getElementById('error').textContent = 'Inserisci una query di ricerca';
                        return;
                    }

                    document.getElementById('loading').style.display = 'block';
                    document.getElementById('error').textContent = '';
                    document.getElementById('results').innerHTML = '';

                    const response = await fetch('/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query, n_results: 5 })
                    });

                    if (!response.ok) {
                        throw new Error('Errore nella ricerca');
                    }

                    const results = await response.json();
                    
                    if (results.length === 0) {
                        document.getElementById('results').innerHTML = '<p>Nessun risultato trovato</p>';
                        return;
                    }

                    const resultsHtml = results.map(result => `
                        <div class="result-item">
                            <h4>${result.metadata.file_path}</h4>
                            <p class="similarity">Rilevanza: ${(result.similarity * 100).toFixed(1)}%</p>
                            <p><strong>Contenuto:</strong> ${result.document}</p>
                        </div>
                    `).join('');
                    
                    document.getElementById('results').innerHTML = resultsHtml;
                } catch (error) {
                    document.getElementById('error').textContent = error.message;
                    console.error('Errore:', error);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }

            // Gestione dell'invio con il tasto Enter
            document.addEventListener('DOMContentLoaded', function() {
                document.getElementById('search-input').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        search();
                    }
                });
                
                // Carica le statistiche all'avvio
                loadStats();
            });
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Second Brain Vector DB Viewer</h1>
            
            <div class="nav">
                <a href="/">Ricerca</a>
                <a href="/visualizer">Visualizzatore 3D</a>
            </div>
            
            <div id="stats" class="stats">
                Caricamento statistiche...
            </div>

            <div class="search-box">
                <input type="text" id="search-input" placeholder="Cerca nel database...">
                <button onclick="search()">Cerca</button>
            </div>

            <div id="error" class="error"></div>
            <div id="loading" class="loading">Caricamento...</div>
            <div id="results" class="results"></div>
        </div>
    </body>
    </html>
    """

@app.get("/stats")
async def get_stats():
    """Ottieni statistiche sul database."""
    try:
        return embeddings_manager.get_collection_stats()
    except Exception as e:
        logger.error(f"Errore nel recupero delle statistiche: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(request: SearchRequest):
    """Cerca nel database vettoriale."""
    try:
        logger.info(f"Ricerca per query: {request.query}")
        results = embeddings_manager.find_similar_documents(request.query, request.n_results)
        return [
            {
                "document": result["document"],
                "metadata": result["metadata"],
                "similarity": 1 - result["distance"]
            }
            for result in results
        ]
    except Exception as e:
        logger.error(f"Errore durante la ricerca: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start_api():
    """Avvia il server API."""
    uvicorn.run(app, host="0.0.0.0", port=8000) 