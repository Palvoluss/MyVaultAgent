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
        results = embeddings_manager.collection.get(
            include=["embeddings", "metadatas", "documents"]
        )
        
        if not results["embeddings"]:
            return {"error": "Nessun embedding trovato nel database"}
            
        # Converti gli embedding in numpy array
        embeddings = np.array(results["embeddings"])
        
        # Riduci la dimensionalità a 3D usando t-SNE
        tsne = TSNE(n_components=3, random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings)
        
        # Prepara i dati per il plot
        plot_data = []
        
        # Colori per i punti
        colors = ['blue'] * len(embeddings_3d)
        
        # Se c'è una query, cerca i documenti simili
        if query:
            try:
                similar_docs = embeddings_manager.find_similar_documents(query, n_results=5)
                similar_paths = [doc["metadata"]["file_path"] for doc in similar_docs]
                
                # Evidenzia i documenti simili in rosso
                for i, metadata in enumerate(results["metadatas"]):
                    if metadata["file_path"] in similar_paths:
                        colors[i] = 'red'
            except Exception as e:
                logger.error(f"Errore nella ricerca di documenti simili: {str(e)}")
        
        # Crea il plot
        fig = go.Figure(data=[go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                opacity=0.8
            ),
            text=[f"File: {metadata['file_path']}<br>Chunk: {metadata['chunk_index'] + 1}/{metadata['total_chunks']}" 
                  for metadata in results["metadatas"]],
            hoverinfo='text'
        )])
        
        # Aggiorna il layout
        fig.update_layout(
            title='Visualizzazione 3D degli Embedding',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        # Converti il plot in HTML
        plot_html = fig.to_html(full_html=False)
        
        return {
            "plot": plot_html,
            "stats": {
                "total_points": len(embeddings_3d),
                "query": query if query else None
            }
        }
        
    except Exception as e:
        logger.error(f"Errore nella generazione del plot 3D: {str(e)}")
        return {"error": f"Errore nella generazione del plot: {str(e)}"}

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
                "similarity": result["relevance"]
            }
            for result in results
        ]
    except Exception as e:
        logger.error(f"Errore durante la ricerca: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start_api():
    """Avvia il server API."""
    uvicorn.run(app, host="0.0.0.0", port=8000) 