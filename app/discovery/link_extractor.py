"""
HTML Link, Form, and Endpoint Extraction Module.

Member 2 Responsibility:
- Extract links from HTML pages and normalize/deduplicate them.
- Extract HTML forms and their input parameters with form type inference.
- Discover referenced API endpoints and relative routes.
- Resolve relative URLs against page base URL and <base> tags.
- Discard non-navigable schemes (javascript:, mailto:, data:, tel:).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlsplit

from app.discovery.url_validator import URLValidator
from app.models.schemas import Endpoint, Form, FormField, ObservationStatus


# Non-navigable schemes to ignore during link extraction
NON_NAVIGABLE_SCHEMES = {"javascript", "mailto", "tel", "data", "sms", "callto"}

# Common API endpoint path pattern for inline discovery
ENDPOINT_REGEX = re.compile(r"""(?:["'])(/(?:api|v\d+|auth|admin|user|users|token|login|logout|webhook|graphql)[a-zA-Z0-9_\-\./?=&%]*)(?:["'])""")


class ExtractedData:
    """Holds parsed links, forms, endpoints, and metadata from an HTML document."""

    def __init__(self) -> None:
        self.title: Optional[str] = None
        self.links: List[str] = []
        self.resource_links: List[str] = []  # script src, link href, img src
        self.forms: List[Form] = []
        self.endpoints: List[Endpoint] = []
        self.base_href: Optional[str] = None


class _HTMLDocumentParser(HTMLParser):
    """
    Stateful SAX-style HTML parser to collect links, forms, fields, title, and base href.
    """

    def __init__(self, page_url: str) -> None:
        super().__init__()
        self.page_url = page_url
        self.effective_base_url = page_url

        self.title_text: List[str] = []
        self.in_title: bool = False

        self.raw_links: List[str] = []
        self.raw_resources: List[str] = []

        # Form extraction tracking
        self.current_form: Optional[Dict] = None
        self.forms: List[Dict] = []

        # Script content collection for endpoint regex discovery
        self.in_script: bool = False
        self.script_buffer: List[str] = []
        self.inline_scripts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_dict = {k.lower(): (v or "") for k, v in attrs}

        # Check for <base href="...">
        if tag == "base" and "href" in attr_dict:
            base_cand = attr_dict["href"].strip()
            if base_cand:
                self.effective_base_url = urljoin(self.page_url, base_cand)

        # Check <title>
        elif tag == "title":
            self.in_title = True

        # Check <script>
        elif tag == "script":
            self.in_script = True
            if "src" in attr_dict:
                self.raw_resources.append(attr_dict["src"])

        # Check <a href="...">
        elif tag == "a" and "href" in attr_dict:
            self.raw_links.append(attr_dict["href"])

        # Check <link href="...">
        elif tag == "link" and "href" in attr_dict:
            self.raw_resources.append(attr_dict["href"])

        # Check <img src="..."> / <iframe src="...">
        elif tag in ("img", "iframe", "video", "audio", "source") and "src" in attr_dict:
            self.raw_resources.append(attr_dict["src"])

        # Check <meta http-equiv="refresh" content="...;url=...">
        elif tag == "meta" and attr_dict.get("http-equiv", "").lower() == "refresh":
            content = attr_dict.get("content", "")
            match = re.search(r"url\s*=\s*([^\s;]+)", content, re.IGNORECASE)
            if match:
                self.raw_links.append(match.group(1).strip("'\""))

        # Form Handling
        elif tag == "form":
            action = attr_dict.get("action", "") or ""
            method = (attr_dict.get("method", "GET") or "GET").upper()
            form_id = attr_dict.get("id") or attr_dict.get("name") or None
            self.current_form = {
                "action": action,
                "method": method,
                "form_id": form_id,
                "fields": [],
            }

        # Form Input Fields
        elif tag == "input" and self.current_form is not None:
            name = attr_dict.get("name") or attr_dict.get("id")
            if name:
                field_type = (attr_dict.get("type") or "text").lower()
                required = "required" in attr_dict
                value = attr_dict.get("value") or None
                self.current_form["fields"].append(
                    FormField(
                        name=name,
                        field_type=field_type,
                        required=required,
                        value=value,
                    )
                )

        elif tag in ("select", "textarea") and self.current_form is not None:
            name = attr_dict.get("name") or attr_dict.get("id")
            if name:
                field_type = tag
                required = "required" in attr_dict
                self.current_form["fields"].append(
                    FormField(
                        name=name,
                        field_type=field_type,
                        required=required,
                        value=None,
                    )
                )

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        elif tag == "script":
            self.in_script = False
            if self.script_buffer:
                self.inline_scripts.append("".join(self.script_buffer))
                self.script_buffer = []
        elif tag == "form":
            if self.current_form is not None:
                self.forms.append(self.current_form)
                self.current_form = None

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_text.append(data)
        elif self.in_script:
            self.script_buffer.append(data)


class HTMLLinkExtractor:
    """
    Extracts, resolves, and deduplicates links, forms, and endpoints from HTML.
    """

    def __init__(self, validator: Optional[URLValidator] = None) -> None:
        self.validator = validator or URLValidator(allow_localhost=True)

    def extract(self, html_content: str, page_url: str) -> ExtractedData:
        """
        Parses HTML and extracts structured links, forms, endpoints, and page metadata.
        """
        data = ExtractedData()
        if not html_content or not isinstance(html_content, str):
            return data

        parser = _HTMLDocumentParser(page_url)
        try:
            parser.feed(html_content)
        except Exception:
            # Tolerant HTML parsing: continue with whatever was extracted before any parsing quirk
            pass

        data.title = "".join(parser.title_text).strip() or None
        data.base_href = parser.effective_base_url

        # 1. Process navigation links
        seen_links: Set[str] = set()
        resolved_links: List[str] = []
        for raw_link in parser.raw_links:
            resolved = self._resolve_and_normalize(raw_link, parser.effective_base_url)
            if resolved and resolved not in seen_links:
                seen_links.add(resolved)
                resolved_links.append(resolved)
        data.links = resolved_links

        # 2. Process resource links
        seen_resources: Set[str] = set()
        resolved_resources: List[str] = []
        for raw_res in parser.raw_resources:
            resolved = self._resolve_and_normalize(raw_res, parser.effective_base_url)
            if resolved and resolved not in seen_resources:
                seen_resources.add(resolved)
                resolved_resources.append(resolved)
        data.resource_links = resolved_resources

        # 3. Process forms
        processed_forms: List[Form] = []
        for raw_form in parser.forms:
            action_raw = raw_form["action"]
            resolved_action = self._resolve_and_normalize(action_raw, parser.effective_base_url) or action_raw or page_url
            fields: List[FormField] = raw_form["fields"]
            form_type = self._infer_form_type(resolved_action, fields)

            form_obj = Form(
                form_id=raw_form["form_id"],
                action=resolved_action,
                method=raw_form["method"],
                fields=fields,
                form_type=form_type,
                page_url=page_url,
            )
            processed_forms.append(form_obj)
        data.forms = processed_forms

        # 4. Discover endpoints from forms, scripts, and links
        data.endpoints = self._extract_endpoints(data, parser.inline_scripts, page_url)

        return data

    def _resolve_and_normalize(self, link: str, base_url: str) -> Optional[str]:
        """Resolves relative link against base URL, drops non-navigable schemes, and normalizes."""
        if not link or not isinstance(link, str):
            return None

        clean_link = link.strip()
        if not clean_link or clean_link.startswith("#"):
            return None

        try:
            parts = urlsplit(clean_link)
        except Exception:
            return None

        scheme = parts.scheme.lower()
        if scheme in NON_NAVIGABLE_SCHEMES:
            return None

        # Resolve relative URL to absolute URL
        absolute_url = urljoin(base_url, clean_link)

        try:
            return self.validator.normalize_url(absolute_url)
        except Exception:
            return None

    def _infer_form_type(self, action: str, fields: List[FormField]) -> Optional[str]:
        """Heuristically infers the security purpose of an HTML form."""
        field_names = [f.name.lower() for f in fields]
        field_types = [f.field_type.lower() for f in fields]
        action_lower = action.lower()

        # Check for authentication form
        has_password = "password" in field_types
        has_user = any(
            re.search(r"user|email|login|username|account", name) for name in field_names
        )
        if has_password and (has_user or "login" in action_lower or "auth" in action_lower):
            return "login"

        if has_password and ("register" in action_lower or "signup" in action_lower or any("confirm" in n for n in field_names)):
            return "register"

        # Check for search form
        if any(name in ("q", "query", "search", "keyword", "term") for name in field_names) or "search" in action_lower:
            return "search"

        # Check for contact/feedback form
        if any(name in ("message", "comment", "feedback", "subject", "body") for name in field_names):
            return "contact"

        # Fallback to action keyword
        if "login" in action_lower or "signin" in action_lower:
            return "login"
        if "register" in action_lower or "signup" in action_lower:
            return "register"

        return "standard"

    def _extract_endpoints(
        self,
        extracted: ExtractedData,
        inline_scripts: List[str],
        page_url: str,
    ) -> List[Endpoint]:
        """Extracts API endpoints and form action endpoints discovered on the page."""
        endpoints_map: Dict[str, Endpoint] = {}

        # 1. Endpoints from forms
        for form in extracted.forms:
            try:
                parts = urlsplit(form.action)
                path = parts.path or "/"
            except Exception:
                path = form.action

            key = f"{form.method}:{path}"
            if key not in endpoints_map:
                endpoints_map[key] = Endpoint(
                    path=path,
                    method=form.method,
                    authentication_required=ObservationStatus.UNKNOWN,
                    content_type="application/x-www-form-urlencoded" if form.method == "POST" else None,
                )

        # 2. Endpoints discovered in inline scripts or text
        for script in inline_scripts:
            matches = ENDPOINT_REGEX.findall(script)
            for raw_endpoint in matches:
                path = raw_endpoint.split("?")[0]
                key = f"GET:{path}"
                if key not in endpoints_map:
                    endpoints_map[key] = Endpoint(
                        path=path,
                        method="GET",
                        authentication_required=ObservationStatus.UNKNOWN,
                    )

        # 3. Endpoints from links that match API pattern
        for link in extracted.links:
            try:
                parts = urlsplit(link)
                path = parts.path
                if any(path.startswith(prefix) for prefix in ("/api/", "/v1/", "/v2/", "/auth/", "/admin/")):
                    key = f"GET:{path}"
                    if key not in endpoints_map:
                        endpoints_map[key] = Endpoint(
                            path=path,
                            method="GET",
                            authentication_required=ObservationStatus.UNKNOWN,
                        )
            except Exception:
                continue

        return list(endpoints_map.values())
