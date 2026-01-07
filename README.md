# INSIGHT BOT

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LLM%20Framework-green)](https://github.com/langchain-ai/langchain)
[![Gemini](https://img.shields.io/badge/Google-Gemini%20API-orange)](https://developers.google.com/ai)
[![License](https://img.shields.io/badge/License-MIT-purple)](./LICENSE)

---

Turn raw content into actionable understanding — INSIGHT BOT is an AI-powered, multi-modal insight engine built with Streamlit, LangChain, and Google Gemini. Ask smarter questions and receive deeper insights from PDFs, websites, images, audio, and video — all inside a single, elegant dashboard.

Table of contents
- Overview
- Key features
- Architecture
- Quickstart
- Installation
- Configuration
- Usage
- Project structure
- Deployment
- Contributing
- Troubleshooting
- Roadmap
- License
- Acknowledgements
- Contact

---

## Overview

INSIGHT BOT provides a single UI for ingesting multiple data formats and querying them with a Large Language Model (Google Gemini) through LangChain pipelines. It is built to be modular, extensible, and easy to run locally using Streamlit.

Use cases:
- Students & researchers: ask questions over PDFs, lecture notes, papers.
- Developers: summarize docs and web content.
- Content creators: extract insights from audio/video.
- Analysts: transform raw sources into concise, actionable output.

---

## Key features

- Multi-source question answering:
  - PDFs (research papers, notes, reports)
  - Websites (URL crawling & scraping)
  - Images (visual understanding)
  - Audio (speech-to-text → analysis)
  - Video (audio extraction + context)
- LangChain pipelines for structured prompts and context handling
- Google Gemini as the LLM backend for high-quality, contextual responses
- Streamlit dashboard with real-time, interactive responses
- Environment-driven configuration for keys and sensitive settings

---

## Architecture (high-level)

User Input → Data Ingestion (PDF / Web / Image / Audio / Video) → LangChain Prompt Pipeline → Google Gemini LLM → Insight Generation → Streamlit Dashboard Output

---

## Quickstart (local)

Prerequisites:
- Python 3.9+
- Git
- A Google Gemini API key (see Configuration)

Clone, configure, install and run:

```bash
git clone https://github.com/VIVEK-MARRI/INSIGHT_BOT.git
cd INSIGHT_BOT

# (optional) create & activate virtualenv
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -r requirements.txt

# copy example env and add your GOOGLE_API_KEY
cp .env.example .env

# run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Installation & dependencies

All runtime dependencies are listed in `requirements.txt`. It's recommended to install inside a virtual environment.

Example:

```bash
python -m pip install -r requirements.txt
```

Core dependencies include:
- streamlit
- langchain
- google-api-client or the official Gemini client (as appropriate)
- beautifulsoup4 (web parsing)
- python-dotenv
- pdf parsing libraries (PyPDF2 / pdfplumber / tika) depending on the implementation
- speech processing libraries (whisper / SpeechRecognition) if audio support is enabled

---

## Configuration

Copy `.env.example` to `.env` and populate the variables. Minimal example:

```env
# .env
GOOGLE_API_KEY=your_google_gemini_api_key_here
# Optional / future:
# VECTOR_DB_URL=
# OPENAI_API_KEY= (if using alternative backends)
```

Notes:
- Keep your API keys secret. Do not commit `.env` to source control.
- If using a cloud-hosted Gemini access mechanism, follow Google's instructions for service accounts and quotas.

---

## Usage

- Start the Streamlit app (`streamlit run app.py`)
- Choose a data source page:
  - Dashboard: overview & example prompts
  - PDF_QA: upload PDFs or drop multiple files
  - Web_QA: enter URLs for scraping and analysis
  - Media_QA: upload images, audio, or video
- Ask questions in natural language. The app extracts context, constructs prompts with LangChain, queries Gemini, and returns concise, context-aware answers and citations.

Tips:
- For large PDFs, try splitting into smaller documents or adjust chunk sizes in the LangChain config.
- When using audio/video, ensure the uploaded file is supported and has clear audio for better transcription quality.

---

## Project structure

```
INSIGHT_BOT/
├── app.py                # Streamlit entrypoint
├── pages/
│   ├── Dashboard.py
│   ├── PDF_QA.py
│   ├── Web_QA.py
│   └── Media_QA.py
├── htmlTemplates.py
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

---

## Deployment

Recommendations:
- For simple hosting, deploy Streamlit to Streamlit Community Cloud or a VM/container.
- For production or team use:
  - Containerize the app (Docker) and deploy to a cloud provider (GCP, AWS, Azure).
  - Use a secrets manager (GCP Secret Manager, AWS Secrets Manager) for API keys.
  - Add rate-limiting and caching for scraped pages and expensive LLM calls.
  - Integrate a vector DB (FAISS, Pinecone, Milvus) for persistent retrieval and better multi-turn memory.

Example Dockerfile (starter):

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Contributing

Contributions are welcome! A few guidelines to help your contribution get accepted quickly:
1. Fork the repository.
2. Create a descriptive branch (e.g., `feature/pdf-chunking`).
3. Make changes, add tests if applicable, and follow existing code style.
4. Submit a PR with a clear description and motivation.

Please open issues for bugs, feature requests, or discussion before large changes. Keep commits focused and atomic.

---

## Troubleshooting & FAQ

Q: The app shows an authentication error for Gemini.
- Make sure `GOOGLE_API_KEY` (or service account) is valid and has access to Gemini APIs.
- Check for network / proxy restrictions.

Q: PDFs are not parsed correctly.
- Try alternative PDF parsers (pdfplumber, PyPDF2) or check if the PDF is a scanned image (requires OCR).

Q: Audio transcriptions are inaccurate.
- Ensure audio is clear and sample rate is standard. Consider running a pre-processing step to remove noise.

If you encounter issues, please open an issue with a reproducible example and relevant logs.

---

## Roadmap / Future Enhancements

- Vector database integration (FAISS / Pinecone / Milvus)
- Conversation memory / multi-turn chat with context persistence
- Rich insight visualizations (charts, timelines, entity graphs)
- Multi-language support and translation pipelines
- Authentication & multi-user support for collaborative workflows
- CI/CD, tests, and deployment templates

---

## License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.

---

## Acknowledgements

- Google Gemini API — the LLM powering responses
- LangChain — LLM orchestration & prompt management
- Streamlit — rapid UI & dashboard
- BeautifulSoup and other open-source tools used for parsing and preprocessing

---

## Contact

Maintainer: VIVEK-MARRI  
Repository: https://github.com/VIVEK-MARRI/INSIGHT_BOT

If you find this project helpful, a star is appreciated! Contributions, feedback, and feature requests are welcome. .......
