from typing import List, Optional
import re

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from pydantic import BaseModel, Field, HttpUrl

# Application metadata and OpenAPI tags
openapi_tags = [
    {
        "name": "Health",
        "description": "Service status and operational checks.",
    },
    {
        "name": "Quotes",
        "description": "Extract quotes from transcripts or uploaded files and format them for social media.",
    },
    {
        "name": "Docs",
        "description": "Documentation helpers and usage notes.",
    },
]

app = FastAPI(
    title="Quote Extractor for Social Media",
    description=(
        "Backend API to extract key quotes from transcripts or uploaded files and format them for social sharing. "
        "Supports text transcript input or file uploads (txt, vtt, srt)."
    ),
    version="1.0.0",
    openapi_tags=openapi_tags,
)

# CORS - open for development; restrict origins in production via .env
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set to your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Models
# -----------------------------

class Quote(BaseModel):
    """Represents a single extracted quote with optional timestamp range in seconds."""
    text: str = Field(..., description="The extracted quote text.")
    start_time: Optional[float] = Field(None, description="Start time in seconds, if available.")
    end_time: Optional[float] = Field(None, description="End time in seconds, if available.")
    source: Optional[str] = Field(None, description="Optional source context, e.g., speaker or file name.")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence of extraction between 0 and 1, if available."
    )


class ExtractRequest(BaseModel):
    """Request model for quote extraction from provided transcript text."""
    transcript: str = Field(..., description="Raw transcript text.")
    top_k: int = Field(5, ge=1, le=50, description="Maximum number of quotes to return.")
    min_length: int = Field(40, ge=10, le=500, description="Minimum quote length in characters.")
    max_length: int = Field(240, ge=40, le=2000, description="Maximum quote length in characters.")
    include_timestamps: bool = Field(False, description="Attempt to parse timestamps if present (SRT/VTT-like).")


class ExtractResponse(BaseModel):
    """Response model containing a list of extracted quotes."""
    quotes: List[Quote] = Field(..., description="List of extracted quotes.")


class FormatRequest(BaseModel):
    """Request model to format one or more quotes for social media use."""
    quotes: List[Quote] = Field(..., description="Quotes to format.")
    platform: str = Field(
        "generic",
        description="Target platform for formatting. e.g., 'twitter', 'linkedin', 'instagram', 'generic'.",
    )
    hashtag: Optional[str] = Field(None, description="Optional hashtag to append.")
    url: Optional[HttpUrl] = Field(None, description="Optional URL to include.")
    style: Optional[str] = Field(
        None,
        description="Optional style hint, e.g., 'concise', 'punchy', 'inspirational'.",
    )
    max_chars: Optional[int] = Field(
        None, ge=40, le=4000, description="Optional hard character limit override per quote."
    )


class FormattedQuote(BaseModel):
    """Represents a quote text formatted for social sharing."""
    original: Quote = Field(..., description="Original quote object.")
    formatted: str = Field(..., description="Formatted text.")
    length: int = Field(..., description="Character length of the formatted text.")


class FormatResponse(BaseModel):
    """Response model containing formatted quotes."""
    items: List[FormattedQuote] = Field(..., description="List of formatted quote texts.")


# -----------------------------
# Helper functions (internal)
# -----------------------------

def _parse_time_str_to_seconds(time_str: str) -> Optional[float]:
    """
    Converts time strings (e.g., 00:01:23.456 or 00:01:23,456) to seconds.
    Returns None if parsing fails.
    """
    match = re.match(r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})([.,](?P<ms>\d{1,3}))?$", time_str.strip())
    if not match:
        return None
    h = int(match.group("h"))
    m = int(match.group("m"))
    s = int(match.group("s"))
    ms = match.group("ms")
    total = h * 3600 + m * 60 + s
    if ms:
        total += int(ms) / (1000 if len(ms) == 3 else 10**len(ms))
    return float(total)


def _extract_from_srt_or_vtt(text: str, top_k: int, min_len: int, max_len: int) -> List[Quote]:
    """
    Rudimentary extraction from SRT/VTT-like content:
    - Groups by time blocks
    - Joins lines into segments
    - Chooses longer segments within allowed length range
    """
    quotes: List[Quote] = []
    # Find timestamp lines like 00:00:01,000 --> 00:00:03,000 or 00:00:01.000 --> 00:00:03.000
    time_line = re.compile(
        r"(?P<start>\d{2}:\d{2}:\d{2}[.,]\d{1,3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}[.,]\d{1,3})"
    )
    blocks = re.split(r"\n\s*\n", text.strip(), flags=re.MULTILINE)
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        # Optional numeric index line
        if len(lines) >= 2 and re.fullmatch(r"\d+", lines[0]):
            lines = lines[1:]
        if not lines:
            continue
        # Timestamp line
        m = time_line.match(lines[0])
        text_lines = lines[1:] if m else lines
        segment = " ".join([re.sub(r"<.*?>", "", l) for l in text_lines]).strip()
        segment = re.sub(r"\s+", " ", segment)
        if len(segment) < min_len:
            continue
        if len(segment) > max_len:
            # Try to split into sentences
            parts = re.split(r"(?<=[.!?])\s+", segment)
            for p in parts:
                p = p.strip()
                if min_len <= len(p) <= max_len:
                    quotes.append(
                        Quote(
                            text=p,
                            start_time=_parse_time_str_to_seconds(m.group("start")) if m else None,
                            end_time=_parse_time_str_to_seconds(m.group("end")) if m else None,
                            confidence=0.6,
                        )
                    )
        else:
            quotes.append(
                Quote(
                    text=segment,
                    start_time=_parse_time_str_to_seconds(m.group("start")) if m else None,
                    end_time=_parse_time_str_to_seconds(m.group("end")) if m else None,
                    confidence=0.7,
                )
            )
        if len(quotes) >= top_k:
            break
    return quotes[:top_k]


def _extract_from_plain_text(text: str, top_k: int, min_len: int, max_len: int) -> List[Quote]:
    """
    Plain text heuristic:
    - Split into sentences
    - Prefer sentences with quotation marks or that are relatively longer and self-contained
    """
    candidates: List[str] = []
    # First, take quoted snippets
    candidates.extend(re.findall(r"“([^”]+)”|\"([^\"]+)\"", text))
    normalized: List[str] = []
    for pair in candidates:
        snippet = pair[0] or pair[1]
        snippet = snippet.strip()
        if snippet:
            normalized.append(snippet)

    # Also split by sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for s in sentences:
        s = s.strip()
        if len(s) >= min_len and len(s) <= max_len:
            normalized.append(s)

    # Deduplicate while preserving order
    seen = set()
    unique_norm = []
    for s in normalized:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique_norm.append(s)

    # Score by length closeness to the middle of [min_len, max_len]
    target = (min_len + max_len) / 2.0
    scored = sorted(unique_norm, key=lambda s: abs(len(s) - target))
    quotes = [Quote(text=s, confidence=0.5 + min(0.4, len(s) / max_len * 0.4)) for s in scored[:top_k]]
    return quotes


def _guess_is_timed(text: str) -> bool:
    """Quickly guess whether transcript looks like SRT/VTT content."""
    return bool(re.search(r"-->\s+\d{2}:\d{2}:\d{2}[.,]\d{1,3}", text)) or bool(
        re.search(r"\d{2}:\d{2}:\d{2}[.,]\d{1,3}", text)
    )


def _format_quote_text(q: Quote, platform: str, hashtag: Optional[str], url: Optional[str], style: Optional[str], max_chars: Optional[int]) -> str:
    """
    Format a quote for social sharing based on platform and style hints.
    """
    base = q.text.strip()
    # Simple style tweaks
    if style == "concise" and len(base) > 200:
        base = base[:197].rstrip() + "..."
    elif style == "punchy":
        base = re.sub(r"\s+", " ", base)
    elif style == "inspirational":
        base = f"“{base}”"

    tail_parts = []
    if hashtag:
        tag = hashtag if hashtag.startswith("#") else f"#{hashtag}"
        tail_parts.append(tag)
    if url:
        tail_parts.append(str(url))

    join_tail = " ".join(tail_parts)
    if join_tail:
        candidate = f"{base} {join_tail}"
    else:
        candidate = base

    # Platform-specific conservative limits
    platform_limits = {
        "twitter": 280,
        "x": 280,
        "linkedin": 3000,
        "instagram": 2200,
        "generic": 1000,
    }
    limit = max_chars or platform_limits.get(platform.lower(), 1000)

    if len(candidate) > limit:
        # Reserve 1 char for ellipsis if needed
        trunc = max(0, limit - 1)
        candidate = candidate[:trunc].rstrip() + "…"

    return candidate


# -----------------------------
# Routes
# -----------------------------

@app.get("/", tags=["Health"], summary="Health Check")
def health_check():
    """
    PUBLIC_INTERFACE
    Health check for the Quote Extractor service.
    Returns a simple JSON indicating the service is up.
    """
    return {"message": "Healthy"}


@app.post(
    "/extract",
    response_model=ExtractResponse,
    summary="Extract quotes",
    description="Extract key quotes from a provided transcript or uploaded file (txt, srt, vtt).",
    tags=["Quotes"],
    responses={
        200: {"description": "Quotes extracted successfully."},
        400: {"description": "Invalid input or file format."},
    },
)
def extract_quotes(
    file: Optional[UploadFile] = File(None, description="Upload transcript file: .txt, .srt, or .vtt."),
    transcript: Optional[str] = Form(None, description="Raw transcript text if not uploading a file."),
    top_k: int = Form(5, description="Max number of quotes to return."),
    min_length: int = Form(40, description="Minimum quote length in characters."),
    max_length: int = Form(240, description="Maximum quote length in characters."),
    include_timestamps: bool = Form(False, description="Attempt to parse timestamps if present (SRT/VTT-like)."),
):
    """
    PUBLIC_INTERFACE
    Extract quotes from either:
    - A multipart/form-data file upload (.txt, .srt, .vtt), or
    - Direct transcript text via form field.

    Parameters:
    - file: Optional uploaded file containing transcript.
    - transcript: Optional raw transcript text.
    - top_k: Number of quotes to return.
    - min_length: Minimum length for quotes.
    - max_length: Maximum length for quotes.
    - include_timestamps: Parse SRT/VTT timestamps when available.

    Returns:
    - ExtractResponse: List of extracted quotes with optional timestamps.
    """
    if not file and not transcript:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'transcript'.")

    content: str
    source_name: Optional[str] = None
    if file:
        source_name = file.filename
        data = file.file.read()
        try:
            content = data.decode("utf-8", errors="ignore")
        except Exception:
            raise HTTPException(status_code=400, detail="Unable to read uploaded file as text.")
        ext = (file.filename or "").lower()
        is_timed = ext.endswith(".srt") or ext.endswith(".vtt") or include_timestamps or _guess_is_timed(content)
        if is_timed:
            quotes = _extract_from_srt_or_vtt(content, top_k, min_length, max_length)
        else:
            quotes = _extract_from_plain_text(content, top_k, min_length, max_length)
        for q in quotes:
            if not q.source:
                q.source = source_name
    else:
        content = transcript or ""
        is_timed = include_timestamps or _guess_is_timed(content)
        if is_timed:
            quotes = _extract_from_srt_or_vtt(content, top_k, min_length, max_length)
        else:
            quotes = _extract_from_plain_text(content, top_k, min_length, max_length)

    return ExtractResponse(quotes=quotes)


@app.post(
    "/format",
    response_model=FormatResponse,
    summary="Format quotes for social media",
    description="Format previously extracted quotes for social sharing on specified platforms.",
    tags=["Quotes"],
    responses={
        200: {"description": "Quotes formatted successfully."},
        422: {"description": "Validation error in provided quotes."},
    },
)
def format_quotes(payload: FormatRequest = Body(..., description="Formatting options and quotes to format.")):
    """
    PUBLIC_INTERFACE
    Formats one or more quotes for social media.

    Parameters:
    - payload: FormatRequest containing quotes, platform target, optional hashtag/url/style/max_chars.

    Returns:
    - FormatResponse: List of formatted quotes with lengths and original mapping.
    """
    items: List[FormattedQuote] = []
    for q in payload.quotes:
        formatted = _format_quote_text(
            q=q,
            platform=payload.platform,
            hashtag=payload.hashtag,
            url=str(payload.url) if payload.url else None,
            style=payload.style,
            max_chars=payload.max_chars,
        )
        items.append(FormattedQuote(original=q, formatted=formatted, length=len(formatted)))
    return FormatResponse(items=items)


@app.get(
    "/docs/usage",
    tags=["Docs"],
    summary="API Usage Notes",
    description="General usage notes for the Quote Extractor API. No WebSocket endpoints are currently provided.",
)
def usage_notes():
    """
    PUBLIC_INTERFACE
    Provide general usage notes and examples for clients integrating with this API.
    """
    return {
        "notes": [
            "POST /extract supports either multipart file uploads (.txt, .srt, .vtt) or form fields with 'transcript'.",
            "Use include_timestamps=true to parse SRT/VTT timestamps if present.",
            "POST /format formats returned quotes with optional hashtag/url/style and platform-specific limits.",
        ],
        "examples": {
            "extract_multipart": "curl -F 'file=@sample.srt' -F 'top_k=5' https://<host>/extract",
            "extract_text": "curl -F 'transcript=Your text here' https://<host>/extract",
            "format": "curl -H 'Content-Type: application/json' -d '{\"quotes\":[{\"text\":\"...\"}],\"platform\":\"twitter\",\"hashtag\":\"MyShow\"}' https://<host>/format",
        },
    }
