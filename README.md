### 1. Clone the Repository
```bash
git clone https://github.com/shohan-hyder/Context-Awareness-AI-Chatbot.git
cd Context-Awareness-AI-Chatbot
```
> 🔁 Replace `your-username` with your actual GitHub username.

### 2. Create `.env` File
Create a `.env` file in the project root:
```bash
touch .env
```

Add your **GROQ API Key**:
```env
GROQ_API_KEY=your_api_key_here
```
> 🔐 Get your free API key at [https://console.groq.com/keys]

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> 💡 First run downloads a local embedding model (~500MB). Subsequent runs are faster.

### 4. Add Your Documents
Place your files in the `docs/` folder:
```
docs/
├── research.pdf
├── notes.docx
└── info.txt
```

Supported formats: `.pdf`, `.docx`, `.txt`

### 5. Run the Agent
```bash
python main.py
```

You’ll see the interactive menu:
```
============================================================
🧠 AI AGENT: GROQ + Local Embeddings
============================================================
1️⃣  Chat & Remember (Personal Memory)
2️⃣  Ask About Documents
3️⃣  View Chat History
4️⃣  Rebuild Document Index
5️⃣  Exit
```

Use it to:
- Store personal info and recall later
- Query your documents
- View conversation history

### 6. Reset (If Needed)
| To Reset | Delete This |
|--------|-------------|
| Document index | `faiss_index/` folder |
| Chat history | `chat_memory.db` file |

> 💡 Re-run `python main.py` after deletion to rebuild.

---

## 🛡️ Security & Privacy
- 🔒 `.env` and `chat_memory.db` are in `.gitignore` — never uploaded
- 🌐 No external logging
- 🧾 All document processing is local

---

## 📂 Project Structure
```
Context-Awareness-AI-Chatbot/
├── main.py                   # Main script
├── requirements.txt          # Dependencies
├── .env                      # API key (ignored)
├── .gitignore                # Prevents sensitive files from upload
├── README.md                 # This file
├── docs/                     # Your documents
├── faiss_index/              # Auto-generated vector store
└── chat_memory.db            # Auto-generated chat history
```

---

> Made with ❤️ using **LangChain**, **GROQ**, and **FAISS**  
> A clean, context-aware AI agent for personal knowledge management.
