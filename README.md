# AI Teaching Assistant | RAG System

An end-to-end AI Teaching Assistant that enables professors to upload course materials and students to ask intelligent, context-aware questions grounded strictly in course content.

Built with a decoupled **FastAPI backend** + **React frontend**, hybrid RAG retrieval, per-student memory, and strict data isolation across professors, courses, and students.

---

## Architecture

```
frontend/          React 18 + Vite + Tailwind CSS  (port 3000)
backend/           FastAPI + uvicorn                (port 8000)
  app/             RAG modules (ingestion, nlp, retrieval, qa)
  auth/            JWT + bcrypt authentication
  routers/         REST API endpoints
  helpers.py       Course/material/index helpers
docker-compose.yml Orchestrates both services
data/              Persisted indexes, uploads, chat logs (volume-mounted)
```

---

## Features

**Professor portal**
- Create and manage courses
- Upload PDF and PPTX course materials
- Build / rebuild vector indexes
- Monitor index status and chunk count

**Student portal**
- Select from available courses
- Ask natural-language questions answered strictly from course materials
- Source citations with page/slide references
- Persistent chat history per student per course

**AI pipeline**
- Hybrid retrieval: BM25 + TF-IDF + FAISS dense embeddings
- Reciprocal Rank Fusion (RRF) for result merging
- Optional cross-encoder reranking
- Grounding validation to prevent hallucinations
- Multi-part question decomposition

---

## Quick Start (Docker)

**Prerequisites:** Docker Desktop, an OpenAI API key.

```bash
git clone https://github.com/AvikshithReddy/teaching-assistant-rag.git
cd teaching-assistant-rag

cp .env.example .env
# Edit .env — set OPENAI_API_KEY and a strong JWT_SECRET

docker compose up -d --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

---

## Usage

1. Open http://localhost:3000
2. Register as **professor** → create a course → upload PDFs/PPTX → rebuild index
3. Register as **student** → select a course → ask questions

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `JWT_SECRET` | Random secret for signing JWTs (use a long random string) |
| `RAG_STORAGE_BACKEND` | `local` (default) |

---

## Local Development (without Docker)

**Backend**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp ../.env.example ../.env  # fill in values
uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev
```

---

## Project Structure

```
backend/
  main.py                   FastAPI entry point
  helpers.py                Course/material/index helpers
  auth/
    jwt.py                  Token creation and verification
    password.py             bcrypt hashing
    users.py                SQLite user store (data/users.db)
  routers/
    auth.py                 POST /auth/register, /auth/login, GET /auth/me
    courses.py              Professor/student course CRUD
    materials.py            File upload and management
    index.py                Index rebuild and status
    chat.py                 Student Q&A and history
  app/
    config.py               Paths and settings
    ingestion/              PDF/PPTX loaders, chunking, preprocessing
    nlp/                    BM25, TF-IDF, embeddings, reranker
    retrieval/              Retriever with RRF fusion
    qa/                     RAG pipeline, chat memory

frontend/
  src/
    api/client.ts           Axios instance + typed API calls
    context/AuthContext.tsx JWT auth state + AuthGuard HOC
    pages/                  LoginPage, ProfessorPage, StudentPage
    components/
      ui/                   Button, Input, Badge, Card, Spinner, Alert
      professor/            CourseManager, MaterialUpload, IndexStatus
      student/              ChatWindow, SourceCard, CourseSelector
```

