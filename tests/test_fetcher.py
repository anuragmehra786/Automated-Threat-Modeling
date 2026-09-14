"""
Unit tests for Safe HTTP Fetcher (Member 2).

Uses httpx.MockTransport to test timeouts, redirects, security headers,
cookie extraction, and size limit enforcement safely without external network calls.
"""

import unittest
import httpx

from app.discovery.fetcher import SafeFetcher
from app.discovery.url_validator import ScopeMode, ScopeValidator


class TestSafeFetcher(unittest.TestCase):
    def test_successful_html_fetch_and_security_headers(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                headers={
                    "Content-Type": "text/html; charset=utf-8",
                    "Content-Security-Policy": "default-src 'self'",
                    "X-Frame-Options": "DENY",
                    "Server": "nginx/1.22.0",
                },
                text="<html><head><title>Test Page</title></head><body><h1>Hello</h1></body></html>",
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        fetcher = SafeFetcher(client=client)

        result = fetcher.fetch("https://example.com/test")
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.is_html)
        self.assertIn("<h1>Hello</h1>", result.text)

        # Verify security headers evaluation
        sec_headers_map = {h.name: h for h in result.security_headers}
        self.assertTrue(sec_headers_map["Content-Security-Policy"].present)
        self.assertEqual(sec_headers_map["Content-Security-Policy"].value, "default-src 'self'")
        self.assertTrue(sec_headers_map["X-Frame-Options"].present)
        self.assertFalse(sec_headers_map["Strict-Transport-Security"].present)
        self.assertIsNotNone(sec_headers_map["Strict-Transport-Security"].recommendation)

    def test_cookie_security_flags_extraction(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                headers=[
                    ("Content-Type", "text/html"),
                    ("Set-Cookie", "session_id=sec123; Path=/; Secure; HttpOnly; SameSite=Strict"),
                    ("Set-Cookie", "tracking_id=trk456; Path=/"),
                ],
                text="<html><body>Cookie test</body></html>",
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        fetcher = SafeFetcher(client=client)

        result = fetcher.fetch("https://example.com/")
        cookies_map = {c.name: c for c in result.cookies}

        self.assertIn("session_id", cookies_map)
        self.assertTrue(cookies_map["session_id"].secure)
        self.assertTrue(cookies_map["session_id"].httponly)
        self.assertEqual(cookies_map["session_id"].samesite, "Strict")

        self.assertIn("tracking_id", cookies_map)
        self.assertFalse(cookies_map["tracking_id"].secure)
        self.assertFalse(cookies_map["tracking_id"].httponly)

    def test_safe_redirect_within_scope(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/old-home":
                return httpx.Response(
                    status_code=301,
                    headers={"Location": "/new-home"},
                )
            return httpx.Response(
                status_code=200,
                headers={"Content-Type": "text/html"},
                text="<html><body>New Home</body></html>",
            )

        scope = ScopeValidator(base_url="https://example.com/")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        fetcher = SafeFetcher(scope_validator=scope, client=client)

        result = fetcher.fetch("https://example.com/old-home")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.url, "https://example.com/new-home")
        self.assertEqual(result.redirect_chain, ["https://example.com/new-home"])
        self.assertIsNone(result.error)

    def test_redirect_aborted_when_leaving_scope(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=302,
                headers={"Location": "https://evil-external-site.com/steal"},
            )

        scope = ScopeValidator(base_url="https://example.com/")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        fetcher = SafeFetcher(scope_validator=scope, client=client)

        result = fetcher.fetch("https://example.com/open-redirect")
        self.assertIsNotNone(result.error)
        self.assertIn("leaves authorized scope", result.error)
        self.assertEqual(result.status_code, 302)

    def test_redirect_loop_detection(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/loop-a":
                return httpx.Response(status_code=302, headers={"Location": "/loop-b"})
            return httpx.Response(status_code=302, headers={"Location": "/loop-a"})

        scope = ScopeValidator(base_url="https://example.com/")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        fetcher = SafeFetcher(scope_validator=scope, client=client)

        result = fetcher.fetch("https://example.com/loop-a")
        self.assertIsNotNone(result.error)
        self.assertIn("Redirect loop detected", result.error)

    def test_response_size_limit_enforcement(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            # Over 100 KB
            return httpx.Response(
                status_code=200,
                headers={"Content-Type": "text/html", "Content-Length": "150000"},
                text="A" * 150000,
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        # Max limit set to 50 KB
        fetcher = SafeFetcher(client=client, max_content_size=50000)

        result = fetcher.fetch("https://example.com/huge-file")
        self.assertIsNotNone(result.error)
        self.assertIn("exceeds maximum limit", result.error)

    def test_timeout_and_connection_error_handling(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Read timed out after 5 seconds")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        fetcher = SafeFetcher(client=client)

        result = fetcher.fetch("https://example.com/slow")
        self.assertIsNotNone(result.error)
        self.assertIn("Request timeout", result.error)


if __name__ == "__main__":
    unittest.main()
