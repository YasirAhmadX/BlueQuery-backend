# BlueQuery 

Backend logic and codes for the **BlueQuery SIH2026** project — an intelligent oceanographic data assistant powered by AI agents.

## 📋 Overview

BlueQuery is a sophisticated system designed to process oceanographic queries about ARGO float data using a multi-agent CrewAI framework. The backend provides:

- **Agentic Query Processing**: A three-tier agent system (Prompt Guard, Query Processor, Output Formatter)
- **Voice Input Support**: Record and transcribe audio queries using Gemini
- **Database Integration**: MCP-based SQLite database access for ARGO float datasets
- **FastAPI REST Endpoints**: Production-ready API for frontend integration
- **Data Visualization**: Support for chart generation and data summarization
- **Safety Guardrails**: Built-in prompt validation and unsafe input detection

## 🏗️ Architecture

### Core Components

#### 1. **Agent-Based Pipeline** (`dev_notebooks/devsan.py`)
   - **Prompt Guard Agent**: Validates user queries for safety and relevance
   - **Query Processor Agent**: Interprets queries and fetches/analyzes ARGO data
   - **Output Formatter Agent**: Formats responses as clean, Markdown-structured output

#### 2. **Voice Input System** (`dev_notebooks/devsan-2.py`)
   - Records audio from microphone
   - Transcribes using Google Gemini API
   - Feeds transcript into the agent pipeline

#### 3. **MCP Integration** (`dev_notebooks/devsan-mcp.py`)
   - Model Context Protocol (MCP) adapter for SQLite database
   - Executes SQL queries against ARGO float database
   - Supports schema-aware data retrieval

#### 4. **Visualization Module** (`dev_notebooks/devsan-viz.py`)
   - Generates charts using `@antv/mcp-server-chart`
   - Visualizes oceanographic data (salinity profiles, trajectories, etc.)

#### 5. **FastAPI Server** (`main.py`, `main2.py`, `main3.py`, `main4.py`, `main5.py`)
   - REST API endpoint: `POST /query`
   - Persistent MCP connection (main2.py)
   - Request caching and rate limiting
   - Configurable database paths and timeouts

## 📁 Project Structure

```
BlueQuery-backend/
├── dev_notebooks/
│   ├── devsan.py              # Base agent pipeline (no MCP)
│   ├── devsan-2.py            # Voice input + agent pipeline
│   ├── devsan-mcp.py          # MCP + SQLite integration
│   ├── devsan-mcp-rag.py      # RAG variant with MCP
│   ├── devsan-viz.py          # Visualization variant
│   └── local_devsan.py        # Local LLM variant (Ollama)
├── main.py                    # FastAPI endpoint (basic)
├── main2.py                   # FastAPI with persistent MCP
├── main3.py                   # FastAPI variant (no formatter)
├── main4.py                   # FastAPI with schema details
├── main5.py                   # FastAPI with logging suppression
├── tests/
│   ├── fapi-test.py          # FastAPI test
│   ├── gemini_voice_test.py   # Gemini transcription test
│   ├── voiceInputTest.py      # Whisper transcription test
│   └── file.md                # Sample output
├── ARGO_DATA/                 # ARGO dataset directory
├── database/                  # SQLite database files
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## 🗄️ Database Schema

The ARGO database includes the following tables:

### `meta_rel` — Metadata
- Platform information (number, type, family, maker)
- Project and PI details
- Sensor and parameter specifications
- Deployment and launch information

### `traj_rel` — Trajectory
- Position data (latitude, longitude, JULD timestamp)
- Positioning system and data state indicators
- Quality control flags and accuracy metrics
- Cycle numbers and data modes

### `prof_rel` — Profiles
- Profile measurements (pressure, temperature, salinity)
- Quality control and adjusted values
- Platform type and cycle information

### `tech_rel` — Technical Parameters
- Technical metadata and versions
- Handbook information
- Parameter specifications and updates

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Node.js](https://nodejs.org/) (for MCP server: `@executeautomation/database-server`)
- [Google Gemini API Key](https://aistudio.google.com/) (for voice/transcription)
- SQLite database file with ARGO data

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YasirAhmadX/BlueQuery-backend.git
   cd BlueQuery-backend
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # Or use: pip install -e .
   ```

4. **Set environment variables**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   export ARGO_DB_PATH="/path/to/argo_database.db"
   ```

### Running the Application

#### Option 1: Interactive CLI
```bash
python dev_notebooks/devsan.py
# or with voice input:
python dev_notebooks/devsan-2.py
```

#### Option 2: FastAPI Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Option 3: MCP-Enhanced Server
```bash
uvicorn main2:app --reload --host 0.0.0.0 --port 8000
```

### API Usage

**Request:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me salinity profiles near the equator in March 2023"}'
```

**Response:**
```json
{
  "query": "Show me salinity profiles near the equator in March 2023",
  "result": "## Salinity Profiles Analysis\n\n### Geographic Focus\n- **Region**: Equatorial Ocean\n- **Timeframe**: March 2023\n...",
  "cached": false
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | Required | Google Gemini API key |
| `ARGO_DB_PATH` | `./agro_db_5_floats.db` | Path to SQLite database |
| `MAX_CONCURRENT_REQUESTS` | `4` | Max simultaneous requests |
| `MCP_CONNECT_TIMEOUT` | `20` | MCP connection timeout (seconds) |
| `REQUEST_TIMEOUT_SECONDS` | `180` | Request processing timeout |
| `THREAD_POOL_WORKERS` | `4` | Thread pool size |

### LLM Configuration

**Default Model**: `gemini/gemini-2.5-flash`

To use a local model (Ollama):
```python
llm = LLM(
    model="ollama/qwen2.5:0.5b",
    temperature=0.7,
    api_base="http://localhost:11434",
    api_key="dummy"
)
```

## 📦 Dependencies

```
crewai[tools]>=0.193.2
python-dotenv>=1.1.1
fastapi
uvicorn
google-genai (for Gemini transcription)
sounddevice (for voice recording)
soundfile (for WAV file handling)
whisper (optional, for local transcription)
```

## 🔐 Safety & Guardrails

- **Prompt Validation**: All queries pass through a safety filter
- **UNSAFE PROMPT Detection**: Blocks irrelevant or malicious queries
- **Structured Output**: Markdown formatting with clear sections
- **Rate Limiting**: Semaphore-based concurrency control

## 🎯 Features

✅ **Multi-agent orchestration** with sequential task execution  
✅ **Voice input** with Gemini transcription  
✅ **Database querying** via MCP SQLite adapter  
✅ **Data visualization** support  
✅ **REST API** for frontend integration  
✅ **Persistent connections** for production use  
✅ **Request caching** for performance  
✅ **Configurable LLMs** (Gemini, Ollama, etc.)  

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v
```

Test voice transcription:
```bash
python tests/gemini_voice_test.py
# or
python tests/voiceInputTest.py
```

Test FastAPI endpoint:
```bash
python tests/fapi-test.py
```

## 📝 License

This project is licensed under the **MIT License**. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Support

For questions or issues, please:
- Open a GitHub [Issue](https://github.com/YasirAhmadX/BlueQuery-backend/issues)
- Contact the maintainer: [@YasirAhmadX](https://github.com/YasirAhmadX)

## 🎓 Project Context

**BlueQuery** is part of the **Smart India Hackathon 2025/2026** initiative, focusing on oceanographic data analysis and accessibility through AI-powered conversational interfaces.

---

**Last Updated**: February 2026  
**Maintainer**: [YasirAhmadX](https://github.com/YasirAhmadX)


---

## 📌 Key Highlights of This README

1. **Clear Overview**: Explains what the project does
2. **Architecture Breakdown**: Details each component's role
3. **File Structure**: Easy to understand the codebase
4. **Database Schema**: Explains the ARGO data structure
5. **Quick Start Guide**: Step-by-step setup instructions
6. **API Examples**: Curl requests showing how to use it
7. **Configuration**: Environment variables and LLM options
8. **Safety Features**: Highlights guardrails
9. **Testing**: Instructions for running tests
10. **Context**: Mentions the SIH2026 hackathon


