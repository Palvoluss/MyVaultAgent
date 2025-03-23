# Second Brain AI Agent

An intelligent agent that automatically analyzes and connects notes in your Obsidian vault using AI and vector embeddings.

## Features

- 🔍 Monitors new notes added to your Obsidian vault
- 🧠 Uses AI to analyze note content and find semantic connections
- 🔗 Suggests relevant connections between notes
- 📚 Maintains a vector database of your knowledge
- 🤖 Provides AI-powered insights about note relationships

## Setup

1. Clone this repository
2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the agent:
   ```bash
   python main.py
   ```

## How it Works

1. The agent monitors your Obsidian vault for new or modified notes
2. When a change is detected, the note's content is processed and embedded into a vector database
3. The agent uses AI to analyze the content and find semantic connections with existing notes
4. Relevant connections are suggested with explanations of why notes are related

## Project Structure

- `main.py`: Entry point and file monitoring
- `embeddings.py`: Vector database management
- `analyzer.py`: AI-powered note analysis
- `utils.py`: Utility functions 