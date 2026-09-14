"""
Unit tests for HTML link, form, and endpoint extraction (Member 2).
"""

import unittest

from app.discovery.link_extractor import HTMLLinkExtractor


class TestHTMLLinkExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = HTMLLinkExtractor()
        self.base_url = "https://example.com/app/index.html"

    def test_extract_navigation_links(self) -> None:
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Dashboard</title></head>
        <body>
            <a href="/dashboard">Dashboard</a>
            <a href="profile.html">Profile</a>
            <a href="https://example.com/settings#tab1">Settings with Hash</a>
            <a href="https://external.com/docs">External Docs</a>
            <a href="#local-anchor">Local Anchor</a>
            <a href="javascript:void(0)">JS link</a>
            <a href="mailto:support@example.com">Email Us</a>
        </body>
        </html>
        """
        extracted = self.extractor.extract(html, self.base_url)

        self.assertEqual(extracted.title, "Dashboard")
        self.assertIn("https://example.com/dashboard", extracted.links)
        self.assertIn("https://example.com/app/profile.html", extracted.links)
        self.assertIn("https://example.com/settings", extracted.links)
        self.assertIn("https://external.com/docs", extracted.links)

        # Ensure non-navigable and pure fragments were discarded
        for link in extracted.links:
            self.assertFalse(link.startswith("javascript:"))
            self.assertFalse(link.startswith("mailto:"))
            self.assertFalse(link.startswith("#"))
            self.assertNotIn("#tab1", link)

    def test_extract_resource_links(self) -> None:
        html = """
        <html>
        <head>
            <link rel="stylesheet" href="/static/css/main.css">
            <script src="/static/js/bundle.js"></script>
        </head>
        <body>
            <img src="/static/images/logo.png" alt="Logo">
            <iframe src="/embedded/widget.html"></iframe>
        </body>
        </html>
        """
        extracted = self.extractor.extract(html, self.base_url)

        self.assertIn("https://example.com/static/css/main.css", extracted.resource_links)
        self.assertIn("https://example.com/static/js/bundle.js", extracted.resource_links)
        self.assertIn("https://example.com/static/images/logo.png", extracted.resource_links)
        self.assertIn("https://example.com/embedded/widget.html", extracted.resource_links)

    def test_extract_forms_and_infer_types(self) -> None:
        html = """
        <html>
        <body>
            <!-- Login Form -->
            <form id="login-form" action="/api/v1/auth/login" method="POST">
                <input type="text" name="username" required>
                <input type="password" name="password" required>
                <input type="hidden" name="csrf_token" value="abc123xyz">
                <button type="submit">Sign In</button>
            </form>

            <!-- Search Form -->
            <form id="search-box" action="/search" method="GET">
                <input type="text" name="q" placeholder="Search...">
                <select name="category">
                    <option value="all">All</option>
                </select>
            </form>

            <!-- Contact Form -->
            <form action="/contact-us" method="POST">
                <input type="email" name="sender_email" required>
                <textarea name="message" required></textarea>
            </form>
        </body>
        </html>
        """
        extracted = self.extractor.extract(html, self.base_url)

        self.assertEqual(len(extracted.forms), 3)

        # Login form checks
        login_form = next(f for f in extracted.forms if f.form_id == "login-form")
        self.assertEqual(login_form.method, "POST")
        self.assertEqual(login_form.action, "https://example.com/api/v1/auth/login")
        self.assertEqual(login_form.form_type, "login")
        self.assertEqual(len(login_form.fields), 3)

        field_names = [f.name for f in login_form.fields]
        self.assertIn("username", field_names)
        self.assertIn("password", field_names)
        self.assertIn("csrf_token", field_names)

        csrf_field = next(f for f in login_form.fields if f.name == "csrf_token")
        self.assertEqual(csrf_field.value, "abc123xyz")

        # Search form checks
        search_form = next(f for f in extracted.forms if f.form_id == "search-box")
        self.assertEqual(search_form.method, "GET")
        self.assertEqual(search_form.form_type, "search")

        # Contact form checks
        contact_form = next(f for f in extracted.forms if f.action == "https://example.com/contact-us")
        self.assertEqual(contact_form.form_type, "contact")

    def test_extract_endpoints_from_scripts_and_forms(self) -> None:
        html = """
        <html>
        <head>
            <script>
                fetch('/api/v2/users/me', { method: 'GET' });
                const authEndpoint = "/auth/refresh-token";
                axios.post('/api/v1/checkout', data);
            </script>
        </head>
        <body>
            <form action="/login" method="POST">
                <input type="password" name="pwd">
            </form>
            <a href="/api/v1/status">Check Status API</a>
        </body>
        </html>
        """
        extracted = self.extractor.extract(html, self.base_url)

        endpoint_paths = [ep.path for ep in extracted.endpoints]
        self.assertIn("/api/v2/users/me", endpoint_paths)
        self.assertIn("/auth/refresh-token", endpoint_paths)
        self.assertIn("/api/v1/checkout", endpoint_paths)
        self.assertIn("/login", endpoint_paths)
        self.assertIn("/api/v1/status", endpoint_paths)

    def test_base_href_resolution(self) -> None:
        html = """
        <html>
        <head>
            <base href="https://example.com/v2/portal/">
        </head>
        <body>
            <a href="overview.html">Overview</a>
        </body>
        </html>
        """
        extracted = self.extractor.extract(html, "https://example.com/page.html")
        self.assertEqual(extracted.links, ["https://example.com/v2/portal/overview.html"])

    def test_meta_refresh_redirect(self) -> None:
        html = """
        <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/new-landing.html">
        </head>
        </html>
        """
        extracted = self.extractor.extract(html, self.base_url)
        self.assertIn("https://example.com/new-landing.html", extracted.links)

    def test_empty_and_corrupt_html_graceful_handling(self) -> None:
        extracted1 = self.extractor.extract("", self.base_url)
        self.assertEqual(extracted1.links, [])
        self.assertIsNone(extracted1.title)

        corrupt_html = "<<<<a>>>href=broken <form><<>>"
        extracted2 = self.extractor.extract(corrupt_html, self.base_url)
        self.assertIsInstance(extracted2.links, list)


if __name__ == "__main__":
    unittest.main()
