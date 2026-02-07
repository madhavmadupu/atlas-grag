# Atlas-GRAG 🌐

**Mapping Unseen Global Supply Chain Risks via Multi-Hop Graph Reasoning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

Atlas-GRAG is a **Graph Retrieval Augmented Generation** system designed to analyze complex supply chain risks that standard RAG systems fail to identify. By combining Knowledge Graphs with Vector Search, Atlas-GRAG can perform multi-hop reasoning to uncover hidden dependencies and cascading risks.

### The Problem Standard RAG Can't Solve

Traditional RAG systems excel at finding semantically similar content, but they fail when answers require **connecting multiple pieces of information**:

> "How will the labor strike in Singapore specifically impact GlobalTech's ability to compete with EuroComputing?"

A standard RAG system might return documents about:
- "Singapore port strikes"
- "GlobalTech company profile"

But it **cannot reason** that:
1. The strike is at the **Port of Singapore**
2. Which affects **TechFlow Inc.**'s facility in Singapore
3. Which manufactures **FlowChips**
4. Which are essential components for **GlobalTech**'s VisionPro Max laptops
5. While **EuroComputing** sources from Germany (unaffected)

**Atlas-GRAG connects these dots automatically.**

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   User Query    │────▶│  Entity         │
└─────────────────┘     │  Extraction     │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
     │  ChromaDB      │ │   Neo4j        │ │   LLM Chain    │
     │  Vector Search │ │   Graph Query  │ │   Reasoning    │
     └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
             │                  │                  │
             └──────────────────┼──────────────────┘
                                ▼
                    ┌─────────────────────┐
                    │  Synthesized Answer │
                    │  with Graph Paths   │
                    └─────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Ollama (llama3) | Local inference, entity extraction |
| **Graph DB** | Neo4j | Knowledge graph storage & Cypher queries |
| **Vector DB** | ChromaDB | Semantic search & document retrieval |
| **Framework** | LangChain | Orchestration & chain management |
| **Frontend** | Streamlit | Interactive web dashboard |

**100% Local. No API Keys Required.**

## 📊 Knowledge Graph Schema

```cypher
// Node Types
(:Company {name, industry, location})
(:Product {name, category})
(:Location {name, type, country})
(:LogisticsNode {name, type, capacity})
(:RiskEvent {name, type, severity, date})

// Relationships
(Company)-[:MANUFACTURES]->(Product)
(Company)-[:DEPENDS_ON]->(Company)
(Product)-[:STORED_IN]->(Location)
(RiskEvent)-[:AFFECTS]->(Location)
(RiskEvent)-[:AFFECTS]->(Company)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Neo4j Desktop (or Docker)
- Ollama with llama3 model

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/atlas-grag.git
cd atlas-grag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your Neo4j credentials
```

### Running Atlas-GRAG

```bash
# Start the application
python main.py

# Or run the Streamlit dashboard
streamlit run src/app/main.py
```

## 📁 Project Structure

```
Atlas-GRAG/
├── data/               # Raw text files & sample data
├── src/
│   ├── ingestion/      # PDF parsing & Graph extraction
│   ├── database/       # Neo4j & ChromaDB managers
│   ├── retriever/      # Hybrid Search logic
│   ├── llm/            # LLM chains & prompts
│   └── app/            # Streamlit UI
├── schema/             # Cypher constraint scripts
├── tests/              # Test suite
├── .env.example        # Environment template
├── requirements.txt    # Dependencies
└── main.py             # Entry point
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Built with ❤️ for understanding complex supply chain dynamics**
