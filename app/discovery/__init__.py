"""
Discovery package for Automated Threat Modeling (Member 2).

Provides safe web discovery, URL validation, scope restriction,
HTTP page fetching, link/form extraction, and crawler orchestration.
"""

from app.discovery.crawler import WebDiscoveryEngine
from app.discovery.fetcher import FetchResult, SafeFetcher
from app.discovery.link_extractor import ExtractedData, HTMLLinkExtractor
from app.discovery.url_validator import (
    ScopeMode,
    ScopeValidator,
    SSRFSecurityError,
    ScopeViolationError,
    URLValidationError,
    URLValidator,
)
from app.models.schemas import DiscoveryResult


def discover_target(target_url: str, **kwargs) -> DiscoveryResult:
    """
    Convenience function to run safe web discovery against a target URL.
    """
    engine = WebDiscoveryEngine(**kwargs)
    return engine.discover(target_url)


__all__ = [
    "URLValidator",
    "ScopeValidator",
    "ScopeMode",
    "URLValidationError",
    "ScopeViolationError",
    "SSRFSecurityError",
    "SafeFetcher",
    "FetchResult",
    "HTMLLinkExtractor",
    "ExtractedData",
    "WebDiscoveryEngine",
    "discover_target",
]
