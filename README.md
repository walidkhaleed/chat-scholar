
# Chat Scholar

Chat Scholar is a web application for interacting with academic documents and assessing essays using AI.

It combines PDF processing, retrieval-augmented generation (RAG), source citation checks, and rubric-based essay grading in a FastAPI application.

**Live demo:** https://chat-scholar-portfolio.onrender.com/app

## Features

### PDF Question Answering
- Upload and process PDF documents.
- Extract text and divide it into searchable chunks.
- Create vector embeddings and store them in FAISS.
- Retrieve relevant passages using semantic and keyword search.
- Combine retrieval results using reciprocal rank fusion (RRF).
- Generate document-grounded responses using Groq.
- Display source references and PDF page numbers.
- Maintain PDF conversation history.

### Essay Grading
- Create custom grading rubrics with weighted criteria.
- Submit essays for AI-assisted assessment.
- Receive criterion-level scores, feedback, and suggested improvements.
- Check generated evidence quotations against the submitted essay.
- Save and retrieve grading reports.

### Authentication and Security
- Password-based authentication.
- Session management and logout revocation.
- CSRF protection.
- Login rate limiting.
- Origin and host validation.
- Resource limits for public document uploads.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| AI inference | Groq API |
| Embeddings | FastEmbed / ONNX |
| Vector search | FAISS |
| Keyword retrieval | BM25 |
| Hybrid retrieval | Reciprocal Rank Fusion |
| PDF processing | PyMuPDF |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render |

## How It Works

### Document Question Answering

1. A user uploads a PDF.
2. The application extracts and chunks the text.
3. Chunks are converted into vector embeddings.
4. Relevant passages are retrieved using hybrid search.
5. Selected passages are provided to the language model.
6. The generated answer is checked for valid source references.
7. The answer and supporting passages are displayed to the user.

The citation validator checks source identifiers, sentence-level citation presence, and numerical references. These are structural and heuristic checks, not a guarantee of factual correctness.

### Essay Assessment

1. The user selects or creates a grading rubric.
2. An essay is submitted for assessment.
3. The language model generates criterion-level feedback and scores.
4. The application calculates the weighted final score.
5. Supporting quotations are checked against the essay text.
6. The grading report is saved for later retrieval.

AI-generated grading is intended for formative feedback and requires human review before consequential use.

## Local Development

The project was developed with Python 3.13.

Clone the repository:

```powershell
git clone https://github.com/walidkhaleed/chat-scholar.git
cd chat-scholar
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a local `.env` file and configure your Groq API key and required application settings.

Generate local authentication credentials:

```powershell
python setup_v11_auth.py
```

This script requires an existing `.env` file and generates a password hash and session signing secret.

Start the application:

```powershell
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/app.

Local configuration may require additional environment settings depending on the embedding backend and security configuration.

## Testing

The repository includes automated tests covering authentication, session security, document processing, retrieval, citation validation, and grading.

Example regression commands:

```powershell
python -m unittest discover -s tests -p "test_*.py" -q
```   

Some integration tests require the corresponding embedding dependencies and environment configuration.

During the final citation-fix regression, 159 tests passed across the V11 and V12 suites.

## Deployment

The application is deployed on Render using `render.yaml` and `requirements-render.txt`.

Deployment secrets are configured through environment variables and are not committed to the repository.

The hosted demonstration uses Render's free service tier.

## Limitations

- AI-generated responses may be inaccurate.
- Citation verification does not establish full semantic support.
- Essay grading requires human oversight.
- The public demonstration applies PDF upload and processing limits.
- The free hosting environment may have cold starts.
- Uploaded documents and saved data are not guaranteed to persist across Render restarts or redeployments.

## Project Status

Chat Scholar V12 is a completed portfolio project with a deployed web interface, document retrieval, source-aware question answering, essay grading, and automated regression tests.

The project is intended to demonstrate practical backend development, AI integration, information retrieval, application security, and deployment.
