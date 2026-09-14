"""
Automated Threat Modeling — Demo FastAPI Backend.

Provides:
    GET  /health          — Liveness probe.
    POST /analyze         — Safe HTTP inspection → DiscoveryResult → Asset
                            identification → STRIDE threat modeling → RAG
                            retrieval → structured JSON response.

Safety Contract:
    - Only makes non-destructive HTTP GET requests to the target URL.
    - Respects a strict timeout and response-size cap.
    - Never submits forms, brutes force, fuzzes, scans ports, or
      performs any active/offensive action.
    - All threat findings are clearly labelled as POTENTIAL / INFERRED.
    - RAG results are retrieved from the local knowledge base only.

Architecture (no modifications to existing pipeline modules):
    URL input
        → safe_http_inspect()           (httpx GET, passive analysis only)
        → build_discovery_result()      (constructs real DiscoveryResult)
        → identify_assets()             (app.security.assets — unchanged)
        → model_threats()              (app.security.threats — unchanged)
        → rag_retrieve_for_threats()    (app.rag.* pipeline — unchanged)
        → JSON response
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Prevent macOS duplicate-OpenMP crash when PyTorch + FAISS are co-loaded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# pyrefly: ignore [missing-import]
import httpx
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Existing pipeline imports (DO NOT rewrite these modules)
# ---------------------------------------------------------------------------
from app.models.schemas import (
    CookieInfo,
    DiscoveryResult,
    Endpoint,
    Form,
    FormField,
    ObservationStatus,
    Page,
    SecurityHeaderInfo,
    SecurityIndicators,
    Technology,
)
from app.rag.chunking import clean_and_chunk_documents
from app.rag.embeddings import EmbeddingModel
from app.rag.ingestion import load_knowledge_documents
from app.rag.vector_store import FAISSVectorStore
from app.security.assets import identify_assets
from app.security.threats import model_threats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
HTTP_TIMEOUT_SECONDS = 10
MAX_RESPONSE_BYTES = 512 * 1024  # 512 KB cap — avoids downloading huge pages
RAG_TOP_K = 3  # Number of chunks to retrieve per threat

# Security headers we explicitly check for
SECURITY_HEADERS = [
    "content-security-policy",
    "strict-transport-security",
    "x-frame-options",
    "x-content-type-options",
    "referrer-policy",
    "permissions-policy",
    "x-xss-protection",
    "cross-origin-opener-policy",
    "cross-origin-resource-policy",
]

# ---------------------------------------------------------------------------
# RAG state — initialised once at startup, reused for every /analyze request
# ---------------------------------------------------------------------------
_rag_store: Optional[FAISSVectorStore] = None
_rag_embedder: Optional[EmbeddingModel] = None


def _build_rag_pipeline() -> tuple[FAISSVectorStore, EmbeddingModel]:
    """
    One-time initialisation of the RAG knowledge pipeline.

    Uses the existing Phase 4 pipeline exactly as designed:
        load_knowledge_documents()      — Phase 4 Step 1 (ingestion)
        clean_and_chunk_documents()     — Phase 4 Step 3 (chunking)
        EmbeddingModel / embed_chunks() — Phase 4 Step 4 (embeddings)
        FAISSVectorStore                — Phase 4 Step 5 (vector store)

    Returns:
        (FAISSVectorStore, EmbeddingModel) — both ready for search.
    """
    logger.info("Initialising RAG pipeline from knowledge base: %s", KNOWLEDGE_BASE_DIR)

    # Step 1: Ingest all knowledge documents
    docs = load_knowledge_documents(KNOWLEDGE_BASE_DIR, recursive=True)
    logger.info("Loaded %d knowledge documents.", len(docs))

    # Step 3: Clean and chunk
    chunks = clean_and_chunk_documents(docs)
    logger.info("Produced %d knowledge chunks.", len(chunks))

    # Step 4: Embed — model loaded once here, reused on every /analyze call
    embedder = EmbeddingModel()
    embedded = embedder.embed_chunks(chunks)
    logger.info(
        "Embedded %d chunks with model '%s' (dim=%d).",
        len(chunks),
        embedder.model_name,
        embedder.embedding_dim,
    )

    # Step 5: Build FAISS index
    store = FAISSVectorStore.from_embedded_chunks(embedded)
    logger.info("FAISS index built: %d vectors, dimension=%d.", store.ntotal, store.dimension)

    return store, embedder


# ---------------------------------------------------------------------------
# Lifespan — build RAG once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_store, _rag_embedder
    try:
        _rag_store, _rag_embedder = _build_rag_pipeline()
        logger.info("RAG pipeline ready.")
    except Exception as exc:
        logger.error("RAG pipeline initialisation failed: %s", exc)
        # Server still starts; /analyze will return a clean error if RAG is unavailable.
    yield
    # Cleanup (nothing persistent to release)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Automated Threat Modeling API",
    description=(
        "Demo API: safe HTTP inspection → DiscoveryResult → asset identification "
        "→ STRIDE threat modeling → RAG retrieval from local cybersecurity knowledge base."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: allow the locally-opened frontend HTML file to call the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must start with http:// or https://")
        if not parsed.netloc:
            raise ValueError("URL must include a valid hostname")
        return v


# ---------------------------------------------------------------------------
# Safe HTTP inspection
# ---------------------------------------------------------------------------

def _parse_title(html: str) -> Optional[str]:
    """Extract <title> text from raw HTML, case-insensitively."""
    m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _detect_technologies(headers: Dict[str, str], html: str) -> List[Technology]:
    """
    Lightweight, non-intrusive technology detection from HTTP response headers
    and limited HTML surface.  Only marks things as detected when positive
    evidence exists in the response itself.
    """
    techs: List[Technology] = []
    server = headers.get("server", "")
    x_powered = headers.get("x-powered-by", "")
    content_type = headers.get("content-type", "")

    def _add(name: str, cats: List[str], evidence: str, confidence: float = 0.85) -> None:
        techs.append(
            Technology(
                name=name,
                categories=cats,
                confidence=confidence,
                evidence=evidence,
            )
        )

    # Server header
    if server:
        for sig, name in [
            ("nginx", "Nginx"),
            ("apache", "Apache HTTP Server"),
            ("cloudflare", "Cloudflare"),
            ("iis", "Microsoft IIS"),
            ("lighttpd", "Lighttpd"),
            ("gunicorn", "Gunicorn"),
            ("uvicorn", "Uvicorn"),
            ("caddy", "Caddy"),
        ]:
            if sig.lower() in server.lower():
                _add(name, ["Web Server"], f"Server: {server}")
                break

    # X-Powered-By
    for sig, name, cat in [
        ("php", "PHP", "Server-side Language"),
        ("asp.net", "ASP.NET", "Web Framework"),
        ("express", "Express.js", "Web Framework"),
        ("django", "Django", "Web Framework"),
    ]:
        if sig in x_powered.lower():
            _add(name, [cat], f"X-Powered-By: {x_powered}")

    # HTML generator meta tag
    meta_gen = re.search(r'<meta[^>]+name=["\']generator["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if meta_gen:
        gen = meta_gen.group(1)
        _add(gen, ["CMS / Generator"], f'<meta name="generator" content="{gen}">', 0.9)

    # Cloudflare-specific header
    if "cf-ray" in headers:
        if not any(t.name == "Cloudflare" for t in techs):
            _add("Cloudflare", ["CDN / WAF"], "CF-Ray header present")

    # Content type hints
    if "text/html" in content_type:
        pass  # too generic to infer a technology

    return techs


def _build_security_indicators(
    headers: Dict[str, str],
    final_url: str,
) -> SecurityIndicators:
    """Build SecurityIndicators from observed HTTP response headers."""
    header_summary: List[SecurityHeaderInfo] = []
    for name in SECURITY_HEADERS:
        present = name in headers
        header_summary.append(
            SecurityHeaderInfo(
                name=name,
                present=present,
                value=headers.get(name),
                recommendation=(
                    None if present else f"Consider adding the '{name}' response header."
                ),
            )
        )

    # Cookies — httpx does not expose Set-Cookie per-cookie easily at the
    # header level without a cookie jar, so we parse the raw header.
    raw_set_cookies = headers.get("set-cookie", "")
    cookie_infos: List[CookieInfo] = []
    if raw_set_cookies:
        cookie_name_m = re.match(r"^([^=]+)=", raw_set_cookies)
        if cookie_name_m:
            cookie_infos.append(
                CookieInfo(
                    name=cookie_name_m.group(1).strip(),
                    secure="secure" in raw_set_cookies.lower(),
                    httponly="httponly" in raw_set_cookies.lower(),
                    samesite=_extract_samesite(raw_set_cookies),
                )
            )

    https_enforced = final_url.startswith("https://")

    return SecurityIndicators(
        headers=dict(headers),
        security_headers_summary=header_summary,
        cookies=cookie_infos,
        https_enforced=https_enforced,
        cors_wildcard=headers.get("access-control-allow-origin") == "*",
        robots_txt_present=None,  # not checked — would require additional request
        sitemap_present=None,
        ssl_certificate_valid=None,  # not verified programmatically here
    )


def _extract_samesite(set_cookie: str) -> Optional[str]:
    m = re.search(r"samesite=(\w+)", set_cookie, re.IGNORECASE)
    return m.group(1).capitalize() if m else None


def _extract_forms_from_html(html: str, base_url: str) -> List[Form]:
    """Extract <form> elements from raw HTML (best-effort, non-intrusive)."""
    forms = []
    for form_match in re.finditer(
        r"<form([^>]*)>(.*?)</form>", html, re.IGNORECASE | re.DOTALL
    ):
        attrs_raw = form_match.group(1)
        body = form_match.group(2)

        action_m = re.search(r'action=["\']([^"\']*)["\']', attrs_raw, re.IGNORECASE)
        method_m = re.search(r'method=["\']([^"\']*)["\']', attrs_raw, re.IGNORECASE)
        action = action_m.group(1) if action_m else base_url
        method = (method_m.group(1) if method_m else "GET").upper()

        fields: List[FormField] = []
        for inp in re.finditer(
            r"<input([^>]*)>", body, re.IGNORECASE
        ):
            inp_attrs = inp.group(1)
            name_m = re.search(r'name=["\']([^"\']*)["\']', inp_attrs, re.IGNORECASE)
            type_m = re.search(r'type=["\']([^"\']*)["\']', inp_attrs, re.IGNORECASE)
            req_m = re.search(r'\brequired\b', inp_attrs, re.IGNORECASE)
            if name_m:
                fields.append(
                    FormField(
                        name=name_m.group(1),
                        field_type=(type_m.group(1) if type_m else "text"),
                        required=bool(req_m),
                    )
                )

        # Infer form purpose
        form_type: Optional[str] = None
        form_text = (action + " ".join(f.name for f in fields)).lower()
        if any(kw in form_text for kw in ["login", "signin", "password", "auth"]):
            form_type = "login"
        elif any(kw in form_text for kw in ["register", "signup", "sign-up"]):
            form_type = "register"
        elif "search" in form_text:
            form_type = "search"

        forms.append(
            Form(
                action=action,
                method=method,
                fields=fields,
                form_type=form_type,
                page_url=base_url,
            )
        )
    return forms


def safe_http_inspect(url: str) -> Dict[str, Any]:
    """
    Perform a single, safe, read-only HTTP GET to the target URL.

    Collects:
        - Final URL after redirects
        - HTTP status code
        - Response headers (lowercased keys)
        - Page title (from HTML)
        - Up to MAX_RESPONSE_BYTES of response body
        - Forms visible in the response HTML
        - Technology signals from headers/HTML
        - Security indicator headers

    Contract:
        - GET only.  No authentication.  No form submission.
        - Hard timeout of HTTP_TIMEOUT_SECONDS.
        - Hard body cap of MAX_RESPONSE_BYTES.
        - Follows redirects (normal browser behaviour).
        - Does NOT crawl links, sub-pages, or any other URL.
    """
    with httpx.Client(
        timeout=HTTP_TIMEOUT_SECONDS,
        follow_redirects=True,
        max_redirects=5,
    ) as client:
        resp = client.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AutoThreatModeling-Demo/0.1; +research)"
            },
        )

    # Normalise headers to lowercase keys
    headers: Dict[str, str] = {k.lower(): v for k, v in resp.headers.items()}
    content_type = headers.get("content-type", "")
    final_url = str(resp.url)

    # Cap body to avoid processing huge downloads
    raw_bytes = resp.content[:MAX_RESPONSE_BYTES]
    try:
        html = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        html = ""

    return {
        "status_code": resp.status_code,
        "final_url": final_url,
        "headers": headers,
        "content_type": content_type,
        "html": html if "html" in content_type else "",
        "is_html": "html" in content_type,
    }


def build_discovery_result(url: str, inspection: Dict[str, Any]) -> DiscoveryResult:
    """
    Construct a real DiscoveryResult from safe HTTP inspection data.

    Strictly follows the Pydantic schema defined in app/models/schemas.py.
    Only populates fields for which positive evidence exists.
    Unknown information stays empty or None — never fabricated.
    """
    headers = inspection["headers"]
    html = inspection["html"]
    final_url = inspection["final_url"]
    status_code = inspection["status_code"]
    is_html = inspection["is_html"]

    # Page title
    title = _parse_title(html) if is_html else None

    # Technology detection from headers + HTML
    technologies = _detect_technologies(headers, html)

    # Security indicators
    security_indicators = _build_security_indicators(headers, final_url)

    # Forms extracted from HTML
    forms = _extract_forms_from_html(html, final_url) if is_html else []

    # The observed page itself
    parsed = urlparse(final_url)
    page = Page(
        url=final_url,
        path=parsed.path or "/",
        status_code=status_code,
        title=title,
        content_type=inspection["content_type"],
        links=[],   # We only loaded one page — no crawling
        forms=forms,
    )

    # Observed endpoints (from forms — the form action URLs are candidate endpoints)
    observed_endpoints: List[Endpoint] = []
    for form in forms:
        if form.action and form.action != final_url:
            observed_endpoints.append(
                Endpoint(
                    path=form.action,
                    method=form.method,
                    authentication_required=ObservationStatus.UNKNOWN,
                )
            )

    return DiscoveryResult(
        url=url,
        technologies=technologies,
        pages=[page],
        endpoints=observed_endpoints,
        forms=forms,
        security_indicators=security_indicators,
        metadata={
            "final_url": final_url,
            "http_status": status_code,
            "content_type": inspection["content_type"],
            "https_enforced": final_url.startswith("https://"),
            "inspection_note": (
                "Single safe GET request only. No crawling. "
                "No form submission. No offensive actions."
            ),
        },
    )


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

_STRIDE_QUERIES: Dict[str, str] = {
    "Spoofing": (
        "authentication failures identity verification credential stuffing "
        "brute force account protection rate limiting MFA multi-factor"
    ),
    "Tampering": (
        "input validation injection broken access control unauthorized modification "
        "data integrity CSRF parameter tampering"
    ),
    "Repudiation": (
        "audit logging non-repudiation event logging accountability "
        "log integrity tamper-proof audit trail"
    ),
    "Information Disclosure": (
        "sensitive data exposure information leakage encryption data protection "
        "error messages misconfiguration access control"
    ),
    "Denial of Service": (
        "resource exhaustion rate limiting throttling DoS denial of service "
        "availability protection account lockout"
    ),
    "Elevation of Privilege": (
        "broken access control authorization enforcement privilege escalation "
        "IDOR insecure direct object reference role-based access"
    ),
}


def _build_rag_query(threat) -> str:  # type: ignore[no-untyped-def]
    """
    Construct a focused security query from a threat's category and title.

    Uses the STRIDE category as the primary signal (deterministic mapping),
    then appends key terms from the threat title for specificity.
    """
    base_query = _STRIDE_QUERIES.get(threat.category.value, "security threat vulnerability defense")
    # Pull meaningful words (>4 chars) from the threat title for context
    title_words = [w for w in threat.title.split() if len(w) > 4]
    title_fragment = " ".join(title_words[:6])
    return f"{base_query} {title_fragment}".strip()


def rag_retrieve_for_threats(threats) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    """
    For each threat, embed a defensive security query and retrieve the top
    relevant KnowledgeChunks from the FAISS vector store.

    Uses:
        EmbeddingModel.embed_text()       — Phase 4 Step 4
        FAISSVectorStore.search()         — Phase 4 Step 5

    Returns a list of dicts, one per threat, each containing:
        threat_id   — the originating threat's ID
        query       — the exact query string that was embedded
        results     — list of VectorSearchResult-derived dicts with provenance
    """
    if _rag_store is None or _rag_embedder is None:
        raise RuntimeError("RAG pipeline not initialised.")

    retrieval_results: List[Dict[str, Any]] = []

    for threat in threats:
        query = _build_rag_query(threat)
        query_vector = _rag_embedder.embed_text(query)
        search_hits = _rag_store.search(query_vector, top_k=RAG_TOP_K)

        chunk_results = []
        for hit in search_hits:
            chunk = hit.chunk
            fm = chunk.framework_metadata

            chunk_results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "title": chunk.section_title or chunk.document_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "source_type": chunk.source_type.value if chunk.source_type else None,
                    "url": chunk.url,
                    "similarity": round(float(hit.score), 4),
                    "framework_metadata": {
                        "owasp": [
                            {"code": r.code, "name": r.name, "url": r.url}
                            for r in fm.owasp
                        ],
                        "cwe": [
                            {"cwe_id": r.cwe_id, "name": r.name, "url": r.url}
                            for r in fm.cwe
                        ],
                        "mitre": [
                            {
                                "technique_id": r.technique_id,
                                "technique_name": r.technique_name,
                                "tactic": r.tactic,
                                "url": r.url,
                            }
                            for r in fm.mitre
                        ],
                        "nist": [
                            {
                                "control_id": r.control_id,
                                "title": r.title,
                                "publication": r.publication,
                                "url": r.url,
                            }
                            for r in fm.nist
                        ],
                    },
                }
            )

        retrieval_results.append(
            {
                "threat_id": threat.id,
                "query": query,
                "results": chunk_results,
            }
        )

    return retrieval_results

def build_attack_paths(assets: List[Any], threats: List[Any]) -> Dict[str, Any]:
    """
    Build a defensive, evidence-based attack-path graph.

    This graph represents potential relationships between observed/inferred
    assets and threats. It does not perform or imply exploitation.
    """
    nodes: List[Dict[str, str]] = []
    edges: List[Dict[str, str]] = []

    asset_ids = set()

    for asset in assets:
        node_id = f"asset-{asset.id}"
        asset_ids.add(asset.id)

        nodes.append(
            {
                "id": node_id,
                "label": asset.name,
                "type": "asset",
            }
        )

    for threat in threats:
        threat_node_id = f"threat-{threat.id}"

        nodes.append(
            {
                "id": threat_node_id,
                "label": threat.title,
                "type": "threat",
            }
        )

        for asset_id in threat.affected_asset_ids:
            if asset_id not in asset_ids:
                continue

            edges.append(
                {
                    "id": f"edge-{asset_id}-{threat.id}",
                    "source": f"asset-{asset_id}",
                    "target": threat_node_id,
                    "label": "potential threat",
                }
            )

    return {
        "nodes": nodes,
        "edges": edges,
    }


def build_recommendations(threats: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert inferred threats into defensive recommendations.

    Recommendations are intentionally high-level and non-exploitative.
    """
    recommendations: List[Dict[str, Any]] = []

    for threat in threats:
        if threat.category == "Spoofing":
            recommendation = (
                "Review authentication and session controls, including "
                "strong identity verification, session protection, "
                "rate limiting, and appropriate authentication safeguards."
            )
        elif threat.category == "Tampering":
            recommendation = (
                "Review authorization and integrity controls for the affected "
                "component, including server-side validation and protection "
                "of state-changing operations."
            )
        elif threat.category == "Information Disclosure":
            recommendation = (
                "Review access controls, transport security, cookie security, "
                "and handling of sensitive information to reduce unintended exposure."
            )
        elif threat.category == "Denial of Service":
            recommendation = (
                "Review rate limiting, concurrency controls, resource limits, "
                "and monitoring for resource-intensive requests."
            )
        elif threat.category == "Elevation of Privilege":
            recommendation = (
                "Review authorization boundaries and ensure privileged "
                "functionality is restricted to appropriately authorized users."
            )
        elif threat.category == "Repudiation":
            recommendation = (
                "Review security logging and audit trails for important "
                "security-sensitive and state-changing operations."
            )
        else:
            recommendation = (
                "Review the affected component's security controls and "
                "validate the potential threat against implementation evidence."
            )

        priority = "MEDIUM"

        if threat.confidence >= 0.85:
            priority = "HIGH"
        elif threat.confidence < 0.60:
            priority = "LOW"

        for asset_id in threat.affected_asset_ids:
            recommendations.append(
                {
                    "id": f"recommendation-{threat.id}-{asset_id}",
                    "recommendation": recommendation,
                    "priority": priority,
                    "related_threat_id": threat.id,
                    "related_asset_id": asset_id,
                }
            )

    return recommendations


def build_summary(
    assets: List[Any],
    threats: List[Any],
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the summary expected by the React frontend.

    Risk here is a presentation-level summary of the highest threat category.
    It does not claim that a vulnerability has been confirmed.
    """
    critical_count = 0
    high_count = 0
    medium_count = 0
    low_count = 0

    for threat in threats:
        confidence = float(threat.confidence)

        if confidence >= 0.90:
            critical_count += 1
        elif confidence >= 0.80:
            high_count += 1
        elif confidence >= 0.60:
            medium_count += 1
        else:
            low_count += 1

    if critical_count > 0:
        overall_risk = "CRITICAL"
    elif high_count > 0:
        overall_risk = "HIGH"
    elif medium_count > 0:
        overall_risk = "MEDIUM"
    elif low_count > 0:
        overall_risk = "LOW"
    else:
        overall_risk = "LOW"

    return {
        "overall_risk": overall_risk,
        "assets_count": len(assets),
        "threats_count": len(threats),
        "critical_threats": critical_count,
        "high_threats": high_count,
        "medium_threats": medium_count,
        "low_threats": low_count,
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    """Liveness probe. Also reports RAG readiness."""
    return {
        "status": "ok",
        "rag_ready": _rag_store is not None,
        "rag_vectors": _rag_store.ntotal if _rag_store else 0,
        "rag_model": _rag_embedder.model_name if _rag_embedder else None,
    }


@app.post("/analyze")
async def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    Main analysis endpoint.

    Pipeline:
        1. Safe HTTP inspection (single GET, passive only)
        2. Build real DiscoveryResult from observed evidence
        3. identify_assets() — existing Phase 3 implementation
        4. model_threats()   — existing STRIDE implementation
        5. RAG retrieval     — existing Phase 4 pipeline
        6. Return structured JSON
    """
    url = request.url

    # --- Step 1: Safe HTTP inspection ---
    try:
        inspection = safe_http_inspect(url)
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Request to '{url}' timed out after {HTTP_TIMEOUT_SECONDS}s.",
        )
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to '{url}': {exc}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Network error reaching '{url}': {exc}",
        )

    # --- Step 2: Build DiscoveryResult ---
    try:
        discovery = build_discovery_result(url, inspection)
    except Exception as exc:
        logger.exception("Failed to build DiscoveryResult for %s", url)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error building discovery result: {exc}",
        )

    # --- Step 3: Asset identification (existing pipeline, unchanged) ---
    assets = identify_assets(discovery)

    # --- Step 4: STRIDE threat modeling (existing pipeline, unchanged) ---
    threats = model_threats(assets, discovery)

    # --- Step 5: RAG retrieval ---
    if _rag_store is None or _rag_embedder is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG pipeline is not ready. The knowledge base may have failed to "
                "initialise at startup. Check server logs."
            ),
        )
    try:
        rag_results = rag_retrieve_for_threats(threats)
    except Exception as exc:
        logger.exception("RAG retrieval failed for %s", url)
        raise HTTPException(status_code=500, detail=f"RAG retrieval error: {exc}")

    # --- Step 6: Build dashboard analysis ---
    attack_paths = build_attack_paths(assets, threats)
    recommendations = build_recommendations(threats)
    summary = build_summary(assets, threats, recommendations)

    # --- Step 7: Serialise and return ---
    return {
        "summary": summary,
        "assets": [a.model_dump(mode="json") for a in assets],
        "threats": [t.model_dump(mode="json") for t in threats],
        "attack_paths": attack_paths,
        "recommendations": recommendations,

        # Additional backend information.
        "url": url,
        "discovery": discovery.model_dump(mode="json"),
        "rag_results": rag_results,
    }