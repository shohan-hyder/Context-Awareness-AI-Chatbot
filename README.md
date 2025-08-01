# 🧠 Context-Aware AI Agent Chatbot

**A document-intelligent chatbot with persistent memory, powered by GROQ and LangChain.**

> 🔍 Ask questions about your PDFs, DOCX, and TXT files — while the agent remembers your conversations across sessions.

This lightweight, extensible AI agent enables **retrieval-augmented generation (RAG)** using GROQ’s high-speed LLMs and local embeddings.

---

## 🚀 Features

| ✅ | Feature |
|----|--------|
| 📄 | Multi-document support (PDF, DOCX, TXT) |
| 💬 | Persistent short-term memory across sessions |
| 📚 | Long-term knowledge retrieval from documents |
| ⚡ | Powered by **GROQ** (Llama-3, Mixtral) — ultra-fast inference |
| 🧠 | Local embeddings via `instructor-large` — no API key needed |
| 🔐 | Secure: API keys stored in `.env`, never committed |
| 🗃️ | Persistent vector store (FAISS) and chat history (SQLite) |
| 🖥️ | CLI-based interface — simple and responsive |

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shohan-hyder/Context-Awareness-AI-Chatbot.git
cd ai-agent