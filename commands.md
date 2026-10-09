# 🛠️ Master Execution Guide (Commands Reference)

This reference contains all verified commands for running, testing, ingesting, and evaluating the Enterprise Agentic RAG system.

---

## 1. Environment Setup

Activate the virtual environment:
```powershell
# Windows PowerShell
.\.venv\Scripts\activate
```

Copy and verify the environment configuration:
```powershell
copy .env.example .env
```

Install production dependencies:
```powershell
pip install -r requirements-prod.txt
```

Install complete development & evaluation dependencies:
```powershell
pip install -r requirements.txt
```

---

## 2. Automated Test Execution

Run the complete automated test suite (unit and integration tests with mocked dependencies):

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Run specific test modules:
```powershell
# Unit tests
python -m unittest tests/unit/test_chunking.py
python -m unittest tests/unit/test_embeddings.py
python -m unittest tests/unit/test_retrieval.py
python -m unittest tests/unit/test_reranking.py
python -m unittest tests/unit/test_planner.py
python -m unittest tests/unit/test_responder.py
python -m unittest tests/unit/test_citations.py
python -m unittest tests/unit/test_guardrails.py
python -m unittest tests/unit/test_evaluation.py

# Integration tests
python -m unittest tests/integration/test_api.py
python -m unittest tests/integration/test_streaming.py
python -m unittest tests/integration/test_memory.py
```

---

## 3. Document Ingestion

Parses documents (PDF, Word DOCX with tables, PowerPoint PPTX with tables, HTML, TXT), generates embeddings, and uploads them to your Qdrant Cloud cluster with an IngestionReport:

### A. Append Ingestion (Adds new documents without wiping):
```powershell
python -m app.ingestion.processor DATA/true_data
```

### B. Fresh Ingestion (Explicit wipe & re-index):
```powershell
python -m app.ingestion.processor DATA/true_data --wipe
```

> ⚠️ WARNING: Never run `--wipe` against a production Qdrant collection.

---

## 4. Running the Unified Application

Start the FastAPI application:

```powershell
uvicorn app.main:app --reload --port 8000
```

- **Web Frontend**: http://localhost:8000
- **Interactive Swagger Docs**: http://localhost:8000/docs
- **Liveness Health Check**: http://localhost:8000/api/health
- **Readiness Probe**: http://localhost:8000/api/ready
- **Workflow Graph Mermaid**: http://localhost:8000/graph

---

## 5. Verification & Health Commands

### Test Liveness Health Endpoint:
```powershell
curl http://localhost:8000/api/health
```

### Test Readiness Probe:
```powershell
curl http://localhost:8000/api/ready
```

### Test Memory Reset Endpoint:
```powershell
curl -X POST http://localhost:8000/memory/clear -H "Content-Type: application/json" -d "{\"thread_id\": \"cli_test\"}"
```

---

## 6. Evaluation Suite

Run the automated evaluation suite via CLI:

```powershell
# Run guardrails evaluation
python -m evals.run_evals --mode guardrails

# Run live query pipeline
python -m evals.run_evals --mode pipeline

# Run RAGAS metrics evaluation
python -m evals.run_evals --mode metrics

# Run full evaluation (pipeline, guardrails, and RAGAS metrics)
python -m evals.run_evals --mode all
```
