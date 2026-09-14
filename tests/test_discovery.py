"""
Integration and unit tests for WebDiscoveryEngine (Member 2).

Tests safe crawling, resource caps, scope boundaries, robots/sitemap parsing,
and DiscoveryResult schema conformance.
"""

import unittest
import httpx

from app.discovery.crawler import WebDiscoveryEngine
from app.discovery.fetcher import SafeFetcher
from app.discovery.url_validator import SSRFSecurityError
from app.models.schemas import DiscoveryResult


class TestWebDiscoveryEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_site = {
            "https://example.com/": {
                "status": 200,
                "headers": {
                    "Content-Type": "text/html",
                    "Server": "nginx/1.24.0",
                    "X-Powered-By": "Express",
                    "Content-Security-Policy": "default-src 'self'",
                    "X-Frame-Options": "DENY",
                },
                "text": """
                <html>
                <head><title>Home Page</title></head>
                <body>
                    <h1>Welcome</h1>
                    <a href="/about">About Us</a>
                    <a href="/login">Sign In</a>
                    <a href="https://external-tracker.com/pixel">Out of Scope</a>
                </body>
                </html>
                """,
            },
            "https://example.com/about": {
                "status": 200,
                "headers": {"Content-Type": "text/html"},
                "text": """
                <html>
                <head><title>About</title></head>
                <body>
                    <p>Company info</p>
                    <a href="/team">Our Team</a>
                </body>
                </html>
                """,
            },
            "https://example.com/login": {
                "status": 200,
                "headers": {
                    "Content-Type": "text/html",
                    "Set-Cookie": "session_id=tok123; Secure; HttpOnly",
                },
                "text": """
                <html>
                <head><title>Login</title></head>
                <body>
                    <form id="auth-form" action="/api/v1/auth" method="POST">
                        <input type="text" name="username" required>
                        <input type="password" name="password" required>
                        <input type="submit" value="Log In">
                    </form>
                </body>
                </html>
                """,
            },
            "https://example.com/team": {
                "status": 200,
                "headers": {"Content-Type": "text/html"},
                "text": "<html><head><title>Team</title></head><body>Leadership</body></html>",
            },
            "https://example.com/robots.txt": {
                "status": 200,
                "headers": {"Content-Type": "text/plain"},
                "text": """
                User-agent: *
                Disallow: /admin
                Sitemap: https://example.com/sitemap.xml
                """,
            },
            "https://example.com/sitemap.xml": {
                "status": 200,
                "headers": {"Content-Type": "application/xml"},
                "text": """
                <?xml version="1.0" encoding="UTF-8"?>
                <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
                    <url><loc>https://example.com/contact</loc></url>
                </urlset>
                """,
            },
            "https://example.com/contact": {
                "status": 200,
                "headers": {"Content-Type": "text/html"},
                "text": "<html><head><title>Contact</title></head><body>Contact Us</body></html>",
            },
            "https://example.com/admin": {
                "status": 403,
                "headers": {"Content-Type": "text/html"},
                "text": "<html><body>Forbidden Admin Area</body></html>",
            },
        }

    def _create_mock_fetcher(self) -> tuple[SafeFetcher, list[str]]:
        requested_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            requested_urls.append(url_str)
            if url_str in self.mock_site:
                page_data = self.mock_site[url_str]
                return httpx.Response(
                    status_code=page_data["status"],
                    headers=page_data["headers"],
                    text=page_data["text"],
                )
            return httpx.Response(status_code=404, text="Not Found")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        fetcher = SafeFetcher(client=client)
        return fetcher, requested_urls

    def test_full_discovery_conforms_to_schema(self) -> None:
        fetcher, requested_urls = self._create_mock_fetcher()
        engine = WebDiscoveryEngine(
            max_pages=10,
            max_depth=3,
            check_robots=True,
            check_sitemap=True,
            fetcher=fetcher,
        )

        result = engine.discover("https://example.com/")

        # Validate DiscoveryResult structure
        self.assertIsInstance(result, DiscoveryResult)
        self.assertEqual(result.url, "https://example.com/")
        self.assertGreater(len(result.pages), 0)

        # External domain must NEVER have been requested
        self.assertNotIn("https://external-tracker.com/pixel", requested_urls)

        # Verify page titles and URLs
        page_urls = [p.url for p in result.pages]
        self.assertIn("https://example.com/", page_urls)
        self.assertIn("https://example.com/about", page_urls)
        self.assertIn("https://example.com/login", page_urls)

        # Verify form extraction
        self.assertEqual(len(result.forms), 1)
        form = result.forms[0]
        self.assertEqual(form.action, "https://example.com/api/v1/auth")
        self.assertEqual(form.method, "POST")
        self.assertEqual(form.form_type, "login")

        # Verify endpoint extraction
        endpoint_paths = [ep.path for ep in result.endpoints]
        self.assertIn("/api/v1/auth", endpoint_paths)

        # Verify technology detection
        tech_names = [t.name for t in result.technologies]
        self.assertIn("nginx", tech_names)
        self.assertIn("Express", tech_names)

        # Verify security indicators
        self.assertTrue(result.security_indicators.robots_txt_present)
        self.assertTrue(result.security_indicators.sitemap_present)
        self.assertEqual(len(result.security_indicators.cookies), 1)
        self.assertTrue(result.security_indicators.cookies[0].secure)
        self.assertTrue(result.security_indicators.cookies[0].httponly)

        # Metadata checks
        self.assertIn("pages_discovered", result.metadata)
        self.assertIn("scan_duration_seconds", result.metadata)

    def test_max_pages_limit(self) -> None:
        fetcher, requested_urls = self._create_mock_fetcher()
        # Limit to 2 pages max
        engine = WebDiscoveryEngine(
            max_pages=2,
            check_robots=False,
            check_sitemap=False,
            fetcher=fetcher,
        )
        result = engine.discover("https://example.com/")
        self.assertLessEqual(len(result.pages), 2)

    def test_max_depth_limit(self) -> None:
        fetcher, requested_urls = self._create_mock_fetcher()
        # Depth 0 means only root is crawled
        engine = WebDiscoveryEngine(
            max_pages=10,
            max_depth=0,
            check_robots=False,
            check_sitemap=False,
            fetcher=fetcher,
        )
        result = engine.discover("https://example.com/")
        page_urls = [p.url for p in result.pages]
        self.assertEqual(page_urls, ["https://example.com/"])

    def test_ssrf_blocking_on_engine_entry(self) -> None:
        engine = WebDiscoveryEngine(allow_localhost=False)
        with self.assertRaises(SSRFSecurityError):
            engine.discover("http://169.254.169.254/")
        with self.assertRaises(SSRFSecurityError):
            engine.discover("http://127.0.0.1:8000/")


if __name__ == "__main__":
    unittest.main()
