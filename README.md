# LocalGPT Pro ⚡

> **A ChatGPT-style assistant that knows your documents.**
> Powered 100% locally by **Qwen/Qwen2.5-Instruct** with multi-user isolation, document RAG, and neural LLM X-Ray inspection.

---

## 🌟 Overview

**LocalGPT Pro** is a private, multi-user local AI assistant. It combines **Firebase Authentication** for user accounts with **strict per-user data isolation** and a **ChatGPT-style SaaS interface**, while keeping all LLM inference and document embeddings **100% on your local machine** with zero external LLM API usage.

---

## 🚀 Key Features

- **🧠 100% Local LLM Inference**: Runs `Qwen/Qwen2.5-1.5B-Instruct` (with `0.5B` fallback) on local CPU or GPU. No external LLM APIs, no data sent to cloud servers.
- **📄 Document RAG (Knowledge Base)**: Upload PDFs, Word documents (`.docx`), Markdown, and text files. Queries retrieve relevant chunks using dense semantic embeddings and **FAISS** vector search with verified page citations.
- **🔬 Inline LLM X-Ray Drawer**: Collapsible neural inspector attached under assistant replies to visualize:
  - Token IDs & decoded tokens
  - Token embedding norms
  - Transformer layer representations
  - Multi-head self-attention heatmaps
  - Top-K vocabulary logits & next-token probability distribution
- **🔐 Multi-User Data Isolation**:
  - Firebase Google Sign-In and Email/Password authentication.
  - Per-user SQLite database (`users`, `chats`, `messages`, `documents`, `settings`).
  - Per-user isolated FAISS indices and document directories (`data/users/<uid>/`).
  - User A can never see or search User B's chats or documents.
- **🎨 Modern SaaS UI**:
  - Dark and Light mode toggle.
  - Live conversation search, pinning, and date grouping (*Pinned*, *Today*, *Yesterday*, *Previous 7 days*, *Older*).
  - Time-of-day greeting hero with starter prompt chips.
- **📥 Chat Export**: Export conversation threads directly to **Markdown (`.md`)** or **PDF (`.pdf`)** with source citation cards.

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Local LLM Engine** | Hugging Face Transformers (`AutoModelForCausalLM`), PyTorch, Qwen2.5 |
| **Vector Store & RAG** | FAISS (`IndexFlatIP`), Sentence Transformers, PyPDF, python-docx |
| **User Interface** | Streamlit, Custom CSS, Plotly (Attention & Logits) |
| **Authentication** | Firebase Web JS SDK, `firebase-admin` (Token Verification) |
| **Database** | SQLite, SQLAlchemy ORM |
| **Export Engine** | ReportLab (PDF), Markdown |

---

## 📂 Project Structure

```
LocalGPT-Pro/
├── app.py                      # Application router (login, workspace, docs, profile)
├── run_server.py               # Robust Windows launcher & port poller
├── run.bat                     # Windows double-click launcher
├── model.py                    # Local Qwen causal LM engine
├── tokenizer.py                # Tokenizer & token analyzer
├── chat.py                     # Streaming generation & message branching
├── memory.py                   # Per-user conversational memory
├── document_loader.py          # PDF, DOCX, TXT, MD parser
├── embeddings.py               # Local Qwen / SentenceTransformer embeddings
├── vector_store.py             # Per-user isolated FAISS vector index
├── rag.py                      # Per-user isolated RAG pipeline
├── xray.py                     # Token, embedding, layer, attention & logit inspector
├── visualization.py            # Plotly charts & attention heatmaps
├── database.py                 # Multi-user SQLAlchemy database engine
├── export.py                   # Markdown & ReportLab PDF exporters
├── requirements.txt            # Python dependencies
├── auth/                       # Firebase Auth & verification
│   ├── firebase_config.py      # Loads credentials from .env
│   ├── firebase_admin_init.py  # Cryptographic token verifier
│   ├── auth.py                 # User session & profile manager
│   ├── guards.py               # Route guards & email verification check
│   └── component/index.html    # Firebase Web JS SDK component
├── pages/                      # Application views
│   ├── login.py                # SaaS Hero landing page & auth portal
│   ├── workspace.py            # Main chat interface with inline X-Ray
│   ├── documents.py            # Document manager & chunk inspector
│   └── profile.py              # User settings, dark/light theme & LLM controls
├── ui/                         # Presentation layer
│   ├── theme.py                # Dark/Light theme definitions
│   ├── components.py           # Avatars, skeletons & greeting hero
│   └── sidebar.py              # Chat search, pinning, date groups & account menu
└── assets/                     # style.css & logo.svg
```

---

## ⚡ Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/bhavanamahantinmath-ui/LocalGPT-Pro.git
cd LocalGPT-Pro
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch LocalGPT Pro
On Windows, simply double-click:
```bash
run.bat
```
Or start via Python:
```bash
python run_server.py
```

The browser will automatically open at **http://127.0.0.1:8501**.

---

## 🔑 Firebase Configuration (Optional)

To enable live Google Sign-In and Email verification:
1. Create a project in the [Firebase Console](https://console.firebase.google.com/).
2. Enable **Authentication** (Google & Email/Password providers).
3. Under **Project Settings → General → Web Apps**, add your credentials to `.env`:
   ```env
   FIREBASE_API_KEY=your_api_key
   FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
   FIREBASE_PROJECT_ID=your_project_id
   FIREBASE_STORAGE_BUCKET=your_project.appspot.com
   FIREBASE_MESSAGING_SENDER_ID=your_sender_id
   FIREBASE_APP_ID=your_app_id
   ```
4. Download your Admin SDK Private Key as `serviceAccountKey.json`.
5. *(Quick demo accounts are also built-in for instant offline usage without Firebase).*

---

## 📜 License
MIT License. Built for local, private, and secure artificial intelligence.

