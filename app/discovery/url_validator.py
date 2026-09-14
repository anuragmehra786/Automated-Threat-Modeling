"""
URL Validation, Normalization, and Scope Restriction Module.

Member 2 Responsibility:
- Robust URL syntax validation and normalization.
- Scope restriction (same-domain or subdomains).
- SSRF and private network defense (preventing unauthorized internal requests).
"""

from __future__ import annotations

import ipaddress
import posixpath
import re
from enum import Enum
from typing import List, Optional, Set
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit


class ScopeMode(str, Enum):
    """Scope restriction mode for web discovery."""
    SAME_DOMAIN = "same_domain"    # Exact hostname match
    SUBDOMAINS = "subdomains"      # Hostname and any of its subdomains


class URLValidationError(ValueError):
    """Base exception for invalid or unsafe URLs."""
    pass


class ScopeViolationError(URLValidationError):
    """Raised when a candidate URL falls outside the authorized discovery scope."""
    pass


class SSRFSecurityError(URLValidationError):
    """Raised when a URL attempts to access private, loopback, or cloud metadata IP ranges."""
    pass


# Reserved cloud metadata IP addresses and hostnames
CLOUD_METADATA_IPS = {
    "169.254.169.254",  # AWS, Azure, GCP, OpenStack instance metadata
    "100.100.100.200",  # Alibaba Cloud metadata
    "169.254.170.2",    # AWS ECS task metadata
}

BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "instance-data",
}


class URLValidator:
    """
    Validates and normalizes URLs to ensure safe, deterministic crawling.

    Guards against:
    - Non-HTTP/HTTPS schemes (e.g., file://, gopher://, javascript:, data:)
    - SSRF targeting private IP ranges, loopback, or cloud metadata endpoints
    - Malformed hostnames and port numbers
    """

    ALLOWED_SCHEMES = {"http", "https"}

    def __init__(
        self,
        allow_localhost: bool = False,
        allow_private_ips: bool = False,
    ) -> None:
        self.allow_localhost = allow_localhost
        self.allow_private_ips = allow_private_ips

    def validate_url(self, url: str) -> str:
        """
        Validates URL syntax, scheme, and destination safety.

        Returns the normalized canonical URL if valid.
        Raises URLValidationError, SSRFSecurityError if invalid or unsafe.
        """
        if not url or not isinstance(url, str):
            raise URLValidationError("URL must be a non-empty string.")

        clean_url = url.strip()
        if not clean_url:
            raise URLValidationError("URL cannot be empty or blank.")

        try:
            parts = urlsplit(clean_url)
        except Exception as exc:
            raise URLValidationError(f"Malformed URL structure: {exc}") from exc

        # 1. Scheme check
        scheme = parts.scheme.lower()
        if scheme not in self.ALLOWED_SCHEMES:
            raise URLValidationError(
                f"Unsupported URL scheme '{parts.scheme}'. Only HTTP and HTTPS are permitted."
            )

        # 2. Hostname validation
        hostname = parts.hostname
        if not hostname:
            raise URLValidationError(f"URL '{clean_url}' is missing a valid hostname.")

        hostname_clean = hostname.rstrip(".").lower()
        if not hostname_clean:
            raise URLValidationError("URL contains an invalid empty hostname.")

        # 3. Port check
        try:
            port = parts.port
        except ValueError as exc:
            raise URLValidationError(f"Invalid port number: {exc}") from exc

        if port is not None:
            if not (1 <= port <= 65535):
                raise URLValidationError(f"Invalid port number: {port}")

        # 4. SSRF and Private IP checking
        self._check_destination_safety(hostname_clean)

        return self.normalize_url(clean_url)

    def _check_destination_safety(self, hostname: str) -> None:
        """Inspects hostname for loopback, private IPs, or cloud metadata endpoints."""
        # Check explicit blocked hostnames
        if not self.allow_localhost:
            if hostname in BLOCKED_HOSTNAMES or hostname.endswith(".localhost") or hostname.endswith(".local"):
                raise SSRFSecurityError(
                    f"Access to localhost/local domain '{hostname}' is blocked for SSRF prevention."
                )

        if hostname in CLOUD_METADATA_IPS:
            raise SSRFSecurityError(
                f"Access to cloud metadata address '{hostname}' is strictly forbidden."
            )

        # Check if the hostname is an IP address
        try:
            ip_obj = ipaddress.ip_address(hostname)
        except ValueError:
            # Not an IP literal, regular domain name
            return

        # Check loopback addresses (127.0.0.0/8, ::1)
        if ip_obj.is_loopback:
            if not self.allow_localhost:
                raise SSRFSecurityError(
                    f"Access to loopback IP '{hostname}' is blocked for SSRF prevention."
                )
            return  # Explicitly allowed when allow_localhost is True

        if ip_obj.is_link_local:
            raise SSRFSecurityError(
                f"Access to link-local IP '{hostname}' is blocked for SSRF prevention."
            )

        if ip_obj.is_private and not self.allow_private_ips:
            raise SSRFSecurityError(
                f"Access to private network IP '{hostname}' is blocked for SSRF prevention."
            )

        if ip_obj.is_multicast or ip_obj.is_reserved or ip_obj.is_unspecified:
            raise SSRFSecurityError(
                f"Access to reserved/multicast IP '{hostname}' is blocked."
            )

    @classmethod
    def normalize_url(cls, url: str) -> str:
        """
        Produces a canonical, deterministic representation of a URL.

        - Lowercases scheme and hostname
        - Removes default ports (:80 for HTTP, :443 for HTTPS)
        - Removes URL fragments (#section)
        - Collapses duplicate slashes and normalizes relative dot segments
        - Standardizes empty path to '/'
        - Sorts query parameters for deduplication
        """
        parts = urlsplit(url.strip())
        scheme = parts.scheme.lower()
        hostname = (parts.hostname or "").rstrip(".").lower()

        # Port normalization
        port = parts.port
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            netloc = hostname
        elif port is not None:
            # Check IPv6 bracket formatting
            if ":" in hostname and not hostname.startswith("["):
                netloc = f"[{hostname}]:{port}"
            else:
                netloc = f"{hostname}:{port}"
        else:
            if ":" in hostname and not hostname.startswith("["):
                netloc = f"[{hostname}]"
            else:
                netloc = hostname

        # Path normalization: collapse duplicate slashes and resolve '.' / '..'
        path = parts.path or "/"
        has_trailing_slash = path.endswith("/") and len(path) > 1

        # Replace duplicate slashes
        path = re.sub(r"/+", "/", path)
        normalized_path = posixpath.normpath(path)
        if has_trailing_slash and not normalized_path.endswith("/"):
            normalized_path += "/"

        # Ensure leading slash
        if not normalized_path.startswith("/"):
            normalized_path = "/" + normalized_path

        # Query normalization: sort parameters for deduplication
        query = ""
        if parts.query:
            query_tuples = parse_qsl(parts.query, keep_blank_values=True)
            sorted_tuples = sorted(query_tuples, key=lambda pair: (pair[0], pair[1]))
            query = urlencode(sorted_tuples)

        # Discard fragments entirely (they are client-side only and cause crawl duplication)
        fragment = ""

        return urlunsplit((scheme, netloc, normalized_path, query, fragment))


class ScopeValidator:
    """
    Enforces that URLs visited during discovery stay within the target's authorized scope.
    """

    def __init__(
        self,
        base_url: str,
        scope_mode: ScopeMode = ScopeMode.SAME_DOMAIN,
        allowed_domains: Optional[Set[str]] = None,
        path_prefix: Optional[str] = None,
        allow_localhost: bool = False,
    ) -> None:
        self.validator = URLValidator(allow_localhost=allow_localhost)
        self.base_url = self.validator.validate_url(base_url)
        self.scope_mode = scope_mode

        base_parts = urlsplit(self.base_url)
        self.base_hostname = (base_parts.hostname or "").lower()
        self.base_scheme = base_parts.scheme.lower()
        self.base_port = base_parts.port

        self.allowed_domains = {d.lower().strip() for d in (allowed_domains or set())}
        self.allowed_domains.add(self.base_hostname)

        # Path prefix restriction (e.g. only crawl /app/ subtree if specified)
        self.path_prefix = path_prefix
        if self.path_prefix and not self.path_prefix.startswith("/"):
            self.path_prefix = "/" + self.path_prefix

    def is_in_scope(self, candidate_url: str) -> bool:
        """
        Determines whether the candidate URL is within authorized scope.
        Returns False rather than raising exceptions on scope violations.
        """
        try:
            norm_url = self.validator.validate_url(candidate_url)
        except URLValidationError:
            return False

        parts = urlsplit(norm_url)
        candidate_host = (parts.hostname or "").lower()

        # Check domain scope
        if self.scope_mode == ScopeMode.SAME_DOMAIN:
            if candidate_host not in self.allowed_domains:
                return False
        elif self.scope_mode == ScopeMode.SUBDOMAINS:
            # Match exact allowed domain or subdomains (*.example.com)
            matched = False
            for allowed in self.allowed_domains:
                if candidate_host == allowed or candidate_host.endswith("." + allowed):
                    matched = True
                    break
            if not matched:
                return False

        # Check path prefix restriction if set
        if self.path_prefix:
            if not parts.path.startswith(self.path_prefix):
                return False

        return True

    def assert_in_scope(self, candidate_url: str) -> str:
        """
        Validates URL and verifies scope.
        Returns normalized URL if in scope, otherwise raises ScopeViolationError.
        """
        norm_url = self.validator.validate_url(candidate_url)
        if not self.is_in_scope(norm_url):
            raise ScopeViolationError(
                f"URL '{candidate_url}' is outside authorized scope (Base: '{self.base_hostname}', Mode: '{self.scope_mode.value}')."
            )
        return norm_url
