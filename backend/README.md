# Quote Extractor Backend (FastAPI)

This service exposes REST endpoints to extract and format quotes for social media.

Endpoints:
- GET / — Health check
- POST /extract — Extract quotes from uploaded file (.txt, .srt, .vtt) or transcript text
- POST /format — Format quotes for social platforms with optional hashtag/url/style
- GET /docs/usage — Usage notes and curl examples

Run locally:
- pip install -r requirements.txt
- uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Generate OpenAPI:
- python -m src.api.generate_openapi

Notes:
- For production, restrict CORS origins via environment configuration.
- Timestamp parsing supports SRT/VTT ranges like `00:00:01,000 --> 00:00:05,000`.
