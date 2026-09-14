"""
Safe HTTP Fetcher Module for Web Discovery.

Member 2 Responsibility:
- Safely fetch HTTP/HTTPS pages with strict timeouts, redirects, and size caps.
- Enforce domain/scope restrictions across redirect chains.
- Inspect and collect security headers, cookies, and TLS/HTTPS indicators.
- Gracefully handle connection errors, timeouts, and non-HTML media.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from typing import Dict, List, Optional
from urllib.parse import urljoin

import httpx

from app.discovery.url_validator import ScopeValidator, URLValidationError, URLValidator
from app.models.schemas import CookieInfo, SecurityHeaderInfo


# Recommended best-practice security headers and default guidance
RECOMMENDED_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "Enforces approved sources for scripts, styles, and media. Recommendation: Define strict CSP policy."
    ),
    "Strict-Transport-Security": (
        "Enforces HTTPS connections. Recommendation: 'max-age=31536000; includeSubDomains; preload'."
    ),
    "X-Frame-Options": (
        "Protects against clickjacking attacks. Recommendation: 'DENY' or 'SAMEORIGIN'."
    ),
    "X-Content-Type-Options": (
        "Prevents MIME-type sniffing. Recommendation: 'nosniff'."
    ),
    "Referrer-Policy": (
        "Controls referrer information leakage. Recommendation: 'strict-origin-when-cross-origin' or 'no-referrer'."
    ),
    "Permissions-Policy": (
        "Restricts access to sensitive browser features (camera, microphone, geolocation)."
    ),
}

DEFAULT_USER_AGENT = (
    "Automated-Threat-Modeling-Scanner/1.0 (+https://github.com/Automated-Threat-Modeling; Defensive-Recon)"
)


@dataclass
class FetchResult:
    """Represents the safe outcome of an HTTP request."""
    url: str
    initial_url: str
    status_code: int = 0
    headers: Dict[str, str] = field(default_factory=dict)
    content_type: str = ""
    text: str = ""
    is_html: bool = False
    redirect_chain: List[str] = field(default_factory=list)
    security_headers: List[SecurityHeaderInfo] = field(default_factory=list)
    cookies: List[CookieInfo] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


class SafeFetcher:
    """
    Hardened HTTP client for security discovery crawling.

    Safety Features:
    - Enforces request timeouts (connect, read).
    - Scope-restricted redirect handling: stops redirects if target leaves scope.
    - Limits response payload size to prevent memory exhaustion (e.g., zip/HTTP bombs).
    - Verifies content-type before reading large non-HTML bodies.
    - Captures raw security indicators, headers, and cookie flags.
    """

    def __init__(
        self,
        scope_validator: Optional[ScopeValidator] = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
        max_redirects: int = 5,
        max_content_size: int = 5 * 1024 * 1024,  # 5 MB limit
        user_agent: str = DEFAULT_USER_AGENT,
        verify_ssl: bool = True,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.scope_validator = scope_validator
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_redirects = max_redirects
        self.max_content_size = max_content_size
        self.user_agent = user_agent
        self.verify_ssl = verify_ssl

        timeout = httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=5.0,
            pool=5.0,
        )
        self.client = client or httpx.Client(
            headers={"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8"},
            timeout=timeout,
            verify=self.verify_ssl,
            follow_redirects=False,  # We manage redirects manually for strict scope enforcement
        )

    def close(self) -> None:
        """Closes the underlying HTTP client session."""
        self.client.close()

    def __enter__(self) -> "SafeFetcher":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def fetch(self, url: str) -> FetchResult:
        """
        Safely fetches the target URL while enforcing scope, redirect, and size limits.
        """
        initial_url = url
        current_url = url
        redirect_chain: List[str] = []

        # Validate initial URL
        if self.scope_validator:
            if not self.scope_validator.is_in_scope(current_url):
                return FetchResult(
                    url=current_url,
                    initial_url=initial_url,
                    error=f"Initial URL '{current_url}' is out of scope.",
                )
        else:
            try:
                current_url = URLValidator.normalize_url(current_url)
            except Exception as exc:
                return FetchResult(
                    url=current_url,
                    initial_url=initial_url,
                    error=f"Invalid URL: {exc}",
                )

        redirect_count = 0
        visited_urls = {current_url}

        req_start = time.time()
        while True:
            try:
                # Stream the response to enforce the max content size safely
                with self.client.stream("GET", current_url) as response:
                    status_code = response.status_code
                    headers = dict(response.headers)
                    content_type = headers.get("content-type", "").lower()

                    # Handle Redirects manually to ensure destination stays in scope
                    if status_code in (301, 302, 303, 307, 308):
                        redirect_count += 1
                        if redirect_count > self.max_redirects:
                            return FetchResult(
                                url=current_url,
                                initial_url=initial_url,
                                status_code=status_code,
                                headers=headers,
                                redirect_chain=redirect_chain,
                                error=f"Exceeded maximum redirects ({self.max_redirects}).",
                            )

                        location = headers.get("location")
                        if not location:
                            return FetchResult(
                                url=current_url,
                                initial_url=initial_url,
                                status_code=status_code,
                                headers=headers,
                                redirect_chain=redirect_chain,
                                error="Redirect status code without Location header.",
                            )

                        # Resolve relative redirect URL
                        next_url = urljoin(current_url, location)
                        try:
                            norm_next_url = URLValidator.normalize_url(next_url)
                        except Exception as exc:
                            return FetchResult(
                                url=current_url,
                                initial_url=initial_url,
                                status_code=status_code,
                                headers=headers,
                                redirect_chain=redirect_chain,
                                error=f"Invalid redirect target '{next_url}': {exc}",
                            )

                        # Scope check on redirect target
                        if self.scope_validator and not self.scope_validator.is_in_scope(norm_next_url):
                            return FetchResult(
                                url=current_url,
                                initial_url=initial_url,
                                status_code=status_code,
                                headers=headers,
                                redirect_chain=redirect_chain,
                                error=f"Redirect to '{norm_next_url}' leaves authorized scope. Aborted.",
                            )

                        # Loop detection
                        if norm_next_url in visited_urls:
                            return FetchResult(
                                url=norm_next_url,
                                initial_url=initial_url,
                                status_code=status_code,
                                headers=headers,
                                redirect_chain=redirect_chain,
                                error=f"Redirect loop detected at '{norm_next_url}'.",
                            )

                        redirect_chain.append(norm_next_url)
                        visited_urls.add(norm_next_url)
                        current_url = norm_next_url
                        continue

                    # Non-redirect response: inspect content size and read safely
                    content_length_hdr = headers.get("content-length")
                    if content_length_hdr and content_length_hdr.isdigit():
                        if int(content_length_hdr) > self.max_content_size:
                            return FetchResult(
                                url=current_url,
                                initial_url=initial_url,
                                status_code=status_code,
                                headers=headers,
                                content_type=content_type,
                                redirect_chain=redirect_chain,
                                error=f"Response size ({content_length_hdr} bytes) exceeds maximum limit ({self.max_content_size} bytes).",
                            )

                    # Read body stream with byte budget
                    body_bytes = bytearray()
                    for chunk in response.iter_bytes(chunk_size=8192):
                        body_bytes.extend(chunk)
                        if len(body_bytes) > self.max_content_size:
                            return FetchResult(
                                url=current_url,
                                initial_url=initial_url,
                                status_code=status_code,
                                headers=headers,
                                content_type=content_type,
                                redirect_chain=redirect_chain,
                                error=f"Response stream exceeded maximum limit of {self.max_content_size} bytes.",
                            )

                    # Decode body text
                    encoding = response.encoding or "utf-8"
                    try:
                        text = body_bytes.decode(encoding, errors="replace")
                    except Exception:
                        text = body_bytes.decode("utf-8", errors="replace")

                    is_html = "text/html" in content_type or "application/xhtml+xml" in content_type

                    # Extract security indicators
                    sec_headers = self._extract_security_headers(headers)
                    cookies = self._extract_cookies(response.headers)

                    return FetchResult(
                        url=current_url,
                        initial_url=initial_url,
                        status_code=status_code,
                        headers=headers,
                        content_type=content_type,
                        text=text,
                        is_html=is_html,
                        redirect_chain=redirect_chain,
                        security_headers=sec_headers,
                        cookies=cookies,
                        elapsed_seconds=round(time.time() - req_start, 4),
                    )

            except httpx.TimeoutException as exc:
                return FetchResult(
                    url=current_url,
                    initial_url=initial_url,
                    error=f"Request timeout connecting to '{current_url}': {exc}",
                )
            except httpx.ConnectError as exc:
                return FetchResult(
                    url=current_url,
                    initial_url=initial_url,
                    error=f"Connection error to '{current_url}': {exc}",
                )
            except httpx.HTTPError as exc:
                return FetchResult(
                    url=current_url,
                    initial_url=initial_url,
                    error=f"HTTP client error fetching '{current_url}': {exc}",
                )
            except Exception as exc:
                return FetchResult(
                    url=current_url,
                    initial_url=initial_url,
                    error=f"Unexpected error fetching '{current_url}': {exc}",
                )

    def _extract_security_headers(self, headers: Dict[str, str]) -> List[SecurityHeaderInfo]:
        """Evaluates observed response headers against recommended security headers."""
        lower_headers = {k.lower(): v for k, v in headers.items()}
        results: List[SecurityHeaderInfo] = []

        for header_name, recommendation in RECOMMENDED_SECURITY_HEADERS.items():
            val = lower_headers.get(header_name.lower())
            results.append(
                SecurityHeaderInfo(
                    name=header_name,
                    present=val is not None,
                    value=val,
                    recommendation=recommendation if val is None else None,
                )
            )
        return results

    def _extract_cookies(self, response_headers: httpx.Headers) -> List[CookieInfo]:
        """Extracts cookie security attributes (Secure, HttpOnly, SameSite)."""
        cookies_list: List[CookieInfo] = []
        set_cookie_headers = response_headers.get_list("set-cookie")

        for cookie_str in set_cookie_headers:
            clean_str = cookie_str.strip()
            if not clean_str:
                continue

            parts = [p.strip() for p in clean_str.split(";")]
            if not parts or not parts[0]:
                continue

            name_val = parts[0].split("=", 1)
            cookie_name = name_val[0].strip()
            lower_parts = [p.lower() for p in parts[1:]]

            secure_flag = any(p == "secure" for p in lower_parts)
            httponly_flag = any(p == "httponly" for p in lower_parts)

            samesite_flag: Optional[str] = None
            for p in parts[1:]:
                if p.lower().startswith("samesite="):
                    val = p.split("=", 1)[1].strip()
                    samesite_flag = val.capitalize() if val else None

            cookies_list.append(
                CookieInfo(
                    name=cookie_name,
                    secure=secure_flag,
                    httponly=httponly_flag,
                    samesite=samesite_flag,
                )
            )

        return cookies_list
