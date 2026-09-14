"""
Web Discovery Crawler and Orchestrator.

Member 2 Responsibility:
- Safe, controlled web discovery engine for target applications.
- Breadth-first crawling restricted strictly to authorized domain scope.
- Resource bounds: max pages, max crawl depth, rate limiting politeness delay.
- Discovers pages, HTML forms, endpoints, and security indicators.
- Assembles the final DiscoveryResult conforming to the team Pydantic contract.
"""

from __future__ import annotations

import collections
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlsplit

from app.discovery.fetcher import SafeFetcher
from app.discovery.link_extractor import HTMLLinkExtractor
from app.discovery.url_validator import ScopeMode, ScopeValidator, URLValidationError, URLValidator
from app.models.schemas import (
    CookieInfo,
    DiscoveryResult,
    Endpoint,
    Form,
    Page,
    SecurityHeaderInfo,
    SecurityIndicators,
    Technology,
)


class WebDiscoveryEngine:
    """
    Safe, scope-restricted web discovery orchestrator.

    Conducts authorized reconnaissance of target web applications while strictly
    respecting scope boundaries, resource limits, and security constraints.
    """

    def __init__(
        self,
        max_pages: int = 20,
        max_depth: int = 3,
        request_delay: float = 0.0,
        scope_mode: ScopeMode = ScopeMode.SAME_DOMAIN,
        allowed_domains: Optional[Set[str]] = None,
        allow_localhost: bool = False,
        verify_ssl: bool = True,
        check_robots: bool = True,
        check_sitemap: bool = True,
        fetcher: Optional[SafeFetcher] = None,
    ) -> None:
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.request_delay = request_delay
        self.scope_mode = scope_mode
        self.allowed_domains = allowed_domains or set()
        self.allow_localhost = allow_localhost
        self.verify_ssl = verify_ssl
        self.check_robots = check_robots
        self.check_sitemap = check_sitemap
        self._custom_fetcher = fetcher

    def discover(self, target_url: str) -> DiscoveryResult:
        """
        Executes controlled web discovery against the authorized target URL.

        Returns a structured DiscoveryResult conforming to the shared Pydantic contract.
        Raises URLValidationError or SSRFSecurityError if target_url is invalid/unsafe.
        """
        start_time = time.time()

        # 1. Validate target URL and configure scope
        validator = URLValidator(allow_localhost=self.allow_localhost)
        validated_target = validator.validate_url(target_url)

        scope_validator = ScopeValidator(
            base_url=validated_target,
            scope_mode=self.scope_mode,
            allowed_domains=self.allowed_domains,
            allow_localhost=self.allow_localhost,
        )

        fetcher = self._custom_fetcher or SafeFetcher(
            scope_validator=scope_validator,
            verify_ssl=self.verify_ssl,
        )
        link_extractor = HTMLLinkExtractor(validator=validator)

        # 2. Tracking collections
        visited_urls: Set[str] = set()
        queue: collections.deque[tuple[str, int]] = collections.deque([(validated_target, 0)])

        discovered_pages: List[Page] = []
        endpoints_map: Dict[str, Endpoint] = {}
        forms_map: Dict[str, Form] = {}
        all_security_headers: List[SecurityHeaderInfo] = []
        all_cookies_map: Dict[str, CookieInfo] = {}
        technologies_map: Dict[str, Technology] = {}

        raw_headers: Dict[str, Optional[str]] = {}
        https_enforced: Optional[bool] = None
        ssl_valid: Optional[bool] = True if validated_target.startswith("https://") else None
        robots_present: Optional[bool] = None
        sitemap_present: Optional[bool] = None
        cors_wildcard: bool = False

        # 3. Probe robots.txt if enabled
        if self.check_robots:
            robots_url = urljoin(validated_target, "/robots.txt")
            if scope_validator.is_in_scope(robots_url):
                res = fetcher.fetch(robots_url)
                robots_present = (res.status_code == 200)
                if robots_present and res.text:
                    self._parse_robots_txt(res.text, validated_target, queue, scope_validator)

        # 4. Probe sitemap.xml if enabled
        if self.check_sitemap:
            sitemap_url = urljoin(validated_target, "/sitemap.xml")
            if scope_validator.is_in_scope(sitemap_url):
                res = fetcher.fetch(sitemap_url)
                sitemap_present = (res.status_code == 200)
                if sitemap_present and res.text:
                    self._parse_sitemap_xml(res.text, queue, scope_validator)

        # 5. BFS Crawl Loop
        while queue and len(visited_urls) < self.max_pages:
            current_url, depth = queue.popleft()

            if current_url in visited_urls:
                continue
            visited_urls.add(current_url)

            # Politeness rate limiting
            if self.request_delay > 0 and len(visited_urls) > 1:
                time.sleep(self.request_delay)

            # Fetch page
            fetch_res = fetcher.fetch(current_url)

            # Check HTTPS enforcement on root
            if current_url == validated_target and validated_target.startswith("http://"):
                if fetch_res.url.startswith("https://") or any(r.startswith("https://") for r in fetch_res.redirect_chain):
                    https_enforced = True
                else:
                    https_enforced = False

            # Capture headers and indicators from initial pages
            if not raw_headers and fetch_res.headers:
                raw_headers = {k: v for k, v in fetch_res.headers.items()}
                all_security_headers = fetch_res.security_headers

            # Check CORS wildcard
            for h_key, h_val in fetch_res.headers.items():
                if h_key.lower() == "access-control-allow-origin" and h_val.strip() == "*":
                    cors_wildcard = True

            # Track cookies
            for cookie in fetch_res.cookies:
                all_cookies_map[cookie.name] = cookie

            # Detect technologies from response headers and content
            self._detect_technologies(fetch_res, technologies_map)

            # Parse page content if successful
            page_path = urlsplit(fetch_res.url).path or "/"
            page_links: List[str] = []
            page_forms: List[Form] = []
            page_endpoint_paths: List[str] = []
            page_title: Optional[str] = None

            if fetch_res.status_code and fetch_res.text:
                extracted = link_extractor.extract(fetch_res.text, fetch_res.url)
                page_title = extracted.title
                page_links = extracted.links
                page_forms = extracted.forms

                # Register discovered forms
                for form in extracted.forms:
                    form_key = f"{form.method}:{form.action}:{len(form.fields)}"
                    forms_map[form_key] = form

                # Register discovered endpoints
                for ep in extracted.endpoints:
                    ep_key = f"{ep.method}:{ep.path}"
                    endpoints_map[ep_key] = ep
                    page_endpoint_paths.append(ep.path)

                # Queue discovered links if within depth limit
                if depth < self.max_depth:
                    for link in extracted.links:
                        if scope_validator.is_in_scope(link) and link not in visited_urls:
                            queue.append((link, depth + 1))

            discovered_pages.append(
                Page(
                    url=fetch_res.url,
                    path=page_path,
                    status_code=fetch_res.status_code or None,
                    title=page_title,
                    content_type=fetch_res.content_type or None,
                    links=page_links,
                    forms=page_forms,
                    endpoints=list(set(page_endpoint_paths)),
                )
            )

        # 6. Build consolidated SecurityIndicators
        security_indicators = SecurityIndicators(
            headers=raw_headers,
            security_headers_summary=all_security_headers,
            cookies=list(all_cookies_map.values()),
            https_enforced=https_enforced,
            ssl_certificate_valid=ssl_valid,
            robots_txt_present=robots_present,
            sitemap_present=sitemap_present,
            cors_wildcard=cors_wildcard,
            raw_indicators={
                "pages_scanned": len(discovered_pages),
                "unique_urls_visited": len(visited_urls),
            },
        )

        scan_duration = time.time() - start_time

        # 7. Construct final DiscoveryResult
        return DiscoveryResult(
            url=validated_target,
            scan_timestamp=datetime.now(timezone.utc),
            technologies=list(technologies_map.values()),
            pages=discovered_pages,
            endpoints=list(endpoints_map.values()),
            forms=list(forms_map.values()),
            security_indicators=security_indicators,
            metadata={
                "pages_discovered": len(discovered_pages),
                "endpoints_discovered": len(endpoints_map),
                "forms_discovered": len(forms_map),
                "max_depth_reached": self.max_depth,
                "scan_duration_seconds": round(scan_duration, 3),
            },
        )

    def _parse_robots_txt(
        self,
        robots_text: str,
        base_url: str,
        queue: collections.deque,
        scope: ScopeValidator,
    ) -> None:
        """Extracts allowed paths and sitemaps from robots.txt without exceeding scope."""
        for line in robots_text.splitlines():
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue

            if clean.lower().startswith("sitemap:"):
                sitemap_loc = clean.split(":", 1)[1].strip()
                if scope.is_in_scope(sitemap_loc):
                    queue.append((sitemap_loc, 1))

            elif clean.lower().startswith("allow:") or clean.lower().startswith("disallow:"):
                parts = clean.split(":", 1)
                if len(parts) == 2:
                    path = parts[1].strip()
                    if path and not path.startswith("*") and not path.endswith("*"):
                        candidate = urljoin(base_url, path)
                        if scope.is_in_scope(candidate):
                            queue.append((candidate, 1))

    def _parse_sitemap_xml(
        self,
        sitemap_text: str,
        queue: collections.deque,
        scope: ScopeValidator,
    ) -> None:
        """Extracts URLs from sitemap.xml <loc> tags within allowed scope."""
        loc_urls = re.findall(r"<loc>(.*?)</loc>", sitemap_text, re.IGNORECASE)
        for loc in loc_urls:
            clean_loc = loc.strip()
            if scope.is_in_scope(clean_loc):
                queue.append((clean_loc, 1))

    def _detect_technologies(
        self,
        fetch_res: Any,
        technologies: Dict[str, Technology],
    ) -> None:
        """Extracts technology indicators from headers and HTML."""
        headers = {k.lower(): v for k, v in fetch_res.headers.items()}

        # 1. Server header
        server = headers.get("server")
        if server and "server" not in technologies:
            technologies["server"] = Technology(
                name=server.split("/")[0],
                version=server.split("/")[1] if "/" in server else None,
                categories=["Web Server"],
                confidence=1.0,
                evidence=f"Server response header: '{server}'",
            )

        # 2. X-Powered-By header
        powered_by = headers.get("x-powered-by")
        if powered_by and "powered_by" not in technologies:
            technologies["powered_by"] = Technology(
                name=powered_by.split("/")[0],
                version=powered_by.split("/")[1] if "/" in powered_by else None,
                categories=["Application Framework"],
                confidence=1.0,
                evidence=f"X-Powered-By header: '{powered_by}'",
            )

        # 3. HTML signatures
        if fetch_res.text:
            lower_text = fetch_res.text.lower()
            if "react" in lower_text and "react" not in technologies:
                technologies["react"] = Technology(
                    name="React",
                    categories=["Frontend Framework"],
                    confidence=0.9,
                    evidence="Detected React patterns in script/DOM",
                )
            if "vue" in lower_text and "vue" not in technologies:
                technologies["vue"] = Technology(
                    name="Vue.js",
                    categories=["Frontend Framework"],
                    confidence=0.9,
                    evidence="Detected Vue patterns in script/DOM",
                )
            if "angular" in lower_text and "angular" not in technologies:
                technologies["angular"] = Technology(
                    name="Angular",
                    categories=["Frontend Framework"],
                    confidence=0.9,
                    evidence="Detected Angular patterns in script/DOM",
                )
