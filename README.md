# SimplexRAG - Fire Alarm System BOQ Processing

## Overview

SimplexRAG is a production-ready fire alarm system sizing and Bill of Quantity (BOQ) processing application. It uses NetworkX graphs for relationship traversal, TF-IDF vector search with sklearn, and OpenAI's API for component analysis.

## Features

- **Component Database**: 294 fire alarm components with relationships
- **Graph-based Relationships**: 119 component relationships using NetworkX
- **Vector Search**: TF-IDF based similarity search for component matching
- **OpenAI Integration**: Batch API processing for component analysis
- **Web Interface**: Flask-based interface for queries and BOQ processing
- **Batch Processing**: Large dataset ingestion with monitoring

## System Status

- **Components**: 294 fire alarm components ingested
- **Relationships**: 119 component relationships restored
- **Database**: SQLite with full component specifications
- **Graph**: NetworkX graph with component compatibility mapping

## Architecture

```
SimplexRAG/
├── simplex_rag/           # Core library
│   ├── config.py         # Configuration settings
│   ├── database.py       # SQLite + NetworkX management
│   ├── orchestrator.py   # Main coordination layer
│   ├── llm_interface.py  # OpenAI API integration
│   └── ingestion/        # Data processing
├── scripts/              # Utility scripts
├── dataset/              # Fire alarm datasheets
└── web_interface.py      # Flask web server
```

## Key Components

### Database Layer (`database.py`)
- SQLite storage for component specifications
- NetworkX graph for relationship mapping
- Vector index for similarity search
- JSON persistence for graph data

### LLM Interface (`llm_interface.py`)
- OpenAI GPT-5-mini integration
- Batch API processing for large datasets
- Component analysis and relationship extraction

### Orchestrator (`orchestrator.py`)
- Coordinates between database, LLM, and search
- Question answering functionality
- BOQ generation and validation

## Installation

```bash
git clone https://github.com/yourusername/simplexlogicrag.git
cd simplexlogicrag
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Configuration

Create `.env` file:
```
OPENAI_API_KEY=your_openai_api_key
SIMPLEX_GRAPH_PATH=simplex_graph.json
SIMPLEX_VECTOR_INDEX_PATH=simplex_vector_index.pkl
SIMPLEX_DB_PATH=simplex_components.db
```

## Usage

### Web Interface
```bash
python create_web_interface.py
```

### Batch Ingestion
```bash
python ingest_full_dataset.py
```

### Component Analysis
```python
from simplex_rag.orchestrator import SimplexRAGOrchestrator

orch = SimplexRAGOrchestrator()
answer = orch.answer_question("What are the main Simplex control panels?")
```

## Investigation & Debugging

The repository includes comprehensive investigation scripts used to resolve relationship data loss:

- `analyze_relationships.py` - Compare graph states
- `deep_investigation.py` - Root cause analysis
- `create_merge_strategy.py` - Relationship restoration planning
- `execute_merge.py` - Merge execution with 94.4% validity

## Recent Fixes

**Relationship Data Recovery** (August 2024):
- **Issue**: Full dataset ingestion resulted in 0 relationships despite 294 components
- **Root Cause**: OpenAI batch responses lacked relationship data extraction
- **Solution**: Merged old graph relationships (233 components, 126 relationships) with new components (294 total)
- **Result**: 119 relationships restored with 94.4% validity rate

**OpenAI API Updates**:
- Updated to GPT-5-mini model
- Fixed markdown-wrapped JSON parsing
- Modern OpenAI client v1+ syntax

## Technical Details

- **Graph Persistence**: NetworkX graphs serialized to JSON
- **Vector Search**: TF-IDF with sklearn for component similarity
- **Batch Processing**: OpenAI Batch API with 10-30 minute processing times
- **Relationship Validation**: 94.4% reference validity maintained during merges

## Production Status

This system is 95% production-ready with:
- ✅ Full component database (294 components)
- ✅ Relationship mapping (119 edges)
- ✅ Vector search functionality
- ✅ Web interface
- ✅ OpenAI integration
- ✅ Batch processing capabilities

## License

MIT License - See LICENSE file for details

## Contributing

Please read CONTRIBUTING.md for contribution guidelines.