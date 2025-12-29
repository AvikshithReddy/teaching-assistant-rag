 AI Teaching Assistant — Advanced RAG System

An end-to-end, production-grade AI Teaching Assistant that enables professors to upload course materials and allows students to ask intelligent, context-aware questions grounded strictly in course content.

Built using Retrieval-Augmented Generation (RAG) with hybrid retrieval, per-student memory, and strong data isolation across professors, courses, and students.

⸻

🚀 Key Features

👩‍🏫 Professor Portal
	•	Create and manage multiple courses
	•	Upload PDFs and PPTX course materials
	•	Automatic duplicate file detection
	•	Build/rebuild vector indexes
	•	Inspect indexed chunks for transparency
	•	Delete materials and courses safely

🎓 Student Portal
	•	Select courses across multiple professors
	•	Ask natural-language questions
	•	Answers grounded only in course materials
	•	Citations with page/slide references
	•	Persistent chat history per student per course
	•	Context-aware follow-up questions (LLM-style chat)

⸻

🧠 Core AI Capabilities
	•	Hybrid Retrieval
	•	TF-IDF (lexical)
	•	Dense embeddings (semantic)
	•	Keyword fallback for recall
	•	Optional Cross-Encoder Reranking
	•	Context-bounded LLM responses
	•	Per-student memory
	•	Strict hallucination prevention
	•	Course-isolated vector stores



┌───────────────┐
│ Course Files  │  (PDF, PPTX)
└──────┬────────┘
       ↓
┌─────────────────────┐
│ Ingestion Pipeline  │
│  - PDF Loader       │
│  - PPTX Loader      │
│  - Preprocessing    │
└──────┬──────────────┘
       ↓
┌────────────────────────────┐
│ Chunking + Metadata        │
│  - Page / Slide numbers    │
│  - Source tracking         │
└──────┬─────────────────────┘
       ↓
┌────────────────────────────┐
│ Hybrid Index                │
│  - TF-IDF Matrix            │
│  - FAISS Vector Index       │
└──────┬─────────────────────┘
       ↓
┌────────────────────────────┐
│ RAG Pipeline                │
│  - Hybrid retrieval         │
│  - Reranking (optional)     │
│  - LLM answer generation    │
└──────┬─────────────────────┘
       ↓
┌────────────────────────────┐
│ Student Chat Experience     │
│  - Persistent memory        │
│  - Context-aware answers    │
│  - Source citations         │
└────────────────────────────┘




git clone https://github.com/<your-username>/teaching-assistant-rag.git
cd teaching-assistant-rag



python -m venv .venv
source .venv/bin/activate  # macOS/Linux


pip install -r requirements.txt


OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL_NAME=gpt-4o-mini   # or any supported model



streamlit run main_app.py

