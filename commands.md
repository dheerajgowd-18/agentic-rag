# 🛠️ Master Execution Guide (Commands Reference)

This reference contains all verified commands for running, ingesting, and testing the Enterprise Agentic RAG system.

---

## 1. Environment Setup

Activate the virtual environment:
```powershell
# Windows PowerShell
.\venv\Scripts\activate
```

Copy and verify the environment configuration:
```powershell
copy .env.example .env
```

---

## 2. Document Ingestion

Parses documents (PDF, Word DOCX, PowerPoint PPTX, HTML, TXT), generates embeddings, and uploads them to your Qdrant Cloud cluster.

### A. Fresh Ingestion (Wipes collection & re-indexes with current vector dimension):
```powershell
python -m app.ingestion.processor DATA/true_data --wipe
```

### B. Append Ingestion (Adds new documents to existing collection):
```powershell
python -m app.ingestion.processor DATA/true_data
```

---

## 3. Running the Unified Application

Start the FastAPI application. This serves both the REST/SSE backend and the **bespoke HTML/CSS/JS frontend**:

```powershell
uvicorn app.main:app --reload --port 8000
```

- **Web Frontend**: http://localhost:8000
- **Interactive Swagger Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health
- **Graph Visualization**: http://localhost:8000/graph

---

## 4. Evaluation Suite

Run the automated evaluation suite via CLI:

```powershell
# Run guardrails evaluation
python -m evals.run_evals --mode guardrails

# Run full evaluation (pipeline, guardrails, and RAGAS metrics)
python -m evals.run_evals --mode all
```

---

## 5. Verification & Health Commands

### Test Health Endpoint:
```powershell
curl http://localhost:8000/api/health
```

### Test Memory Reset Endpoint:
```powershell
curl -X POST http://localhost:8000/memory/clear -H "Content-Type: application/json" -d '{\"thread_id\": \"cli_test\"}'
```
