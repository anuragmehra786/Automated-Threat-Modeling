"""
Unit tests for URL validation, normalization, SSRF protection, and scope restriction (Member 2).
"""

import unittest

from app.discovery.url_validator import (
    ScopeMode,
    ScopeValidator,
    ScopeViolationError,
    SSRFSecurityError,
    URLValidationError,
    URLValidator,
)


class TestURLValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = URLValidator(allow_localhost=False)
        self.dev_validator = URLValidator(allow_localhost=True)

    def test_valid_http_and_https_urls(self) -> None:
        valid_urls = [
            "https://example.com",
            "http://example.com/",
            "https://sub.domain.example.com/path/to/resource?q=1",
            "https://example.com:8443/api/v1",
            "http://example.com:8080/dashboard",
        ]
        for url in valid_urls:
            normalized = self.validator.validate_url(url)
            self.assertTrue(normalized.startswith("http"))

    def test_rejected_unsupported_schemes(self) -> None:
        unsupported = [
            "ftp://example.com",
            "file:///etc/passwd",
            "gopher://example.com",
            "javascript:alert(1)",
            "data:text/html,<h1>test</h1>",
            "mailto:admin@example.com",
            "tel:+1234567890",
        ]
        for url in unsupported:
            with self.assertRaises(URLValidationError, msg=f"Should reject {url}"):
                self.validator.validate_url(url)

    def test_empty_and_malformed_urls(self) -> None:
        malformed = [
            "",
            "   ",
            "http://",
            "https://:8080",
            "not-a-url",
            "http:///path",
        ]
        for url in malformed:
            with self.assertRaises(URLValidationError):
                self.validator.validate_url(url)

    def test_port_validation(self) -> None:
        with self.assertRaises(URLValidationError):
            self.validator.validate_url("http://example.com:70000")
        with self.assertRaises(URLValidationError):
            self.validator.validate_url("http://example.com:0")
        with self.assertRaises(URLValidationError):
            self.validator.validate_url("http://example.com:-80")

    def test_normalization_lowercase_scheme_and_host(self) -> None:
        norm = URLValidator.normalize_url("HTTPS://EXAMPLE.COM/Page")
        self.assertEqual(norm, "https://example.com/Page")

    def test_normalization_strip_default_ports(self) -> None:
        self.assertEqual(
            URLValidator.normalize_url("http://example.com:80/home"),
            "http://example.com/home",
        )
        self.assertEqual(
            URLValidator.normalize_url("https://example.com:443/home"),
            "https://example.com/home",
        )
        # Non-default ports must be preserved
        self.assertEqual(
            URLValidator.normalize_url("http://example.com:8080/home"),
            "http://example.com:8080/home",
        )
        self.assertEqual(
            URLValidator.normalize_url("https://example.com:8443/home"),
            "https://example.com:8443/home",
        )

    def test_normalization_strip_fragments(self) -> None:
        norm = URLValidator.normalize_url("https://example.com/docs#section-1")
        self.assertEqual(norm, "https://example.com/docs")
        self.assertNotIn("#", norm)

    def test_normalization_relative_dot_segments(self) -> None:
        self.assertEqual(
            URLValidator.normalize_url("https://example.com/a/b/../c"),
            "https://example.com/a/c",
        )
        self.assertEqual(
            URLValidator.normalize_url("https://example.com/a/./b/./c"),
            "https://example.com/a/b/c",
        )

    def test_normalization_collapse_duplicate_slashes(self) -> None:
        self.assertEqual(
            URLValidator.normalize_url("https://example.com//admin///login"),
            "https://example.com/admin/login",
        )

    def test_normalization_query_parameter_sorting(self) -> None:
        url1 = URLValidator.normalize_url("https://example.com/search?b=2&a=1&c=3")
        url2 = URLValidator.normalize_url("https://example.com/search?c=3&a=1&b=2")
        self.assertEqual(url1, url2)
        self.assertEqual(url1, "https://example.com/search?a=1&b=2&c=3")

    def test_ssrf_blocking_private_ips(self) -> None:
        private_ips = [
            "http://10.0.0.1/admin",
            "http://10.254.10.5/",
            "http://172.16.0.1/",
            "http://172.31.255.255/",
            "http://192.168.1.1/",
            "http://192.168.100.50/api",
        ]
        for ip_url in private_ips:
            with self.assertRaises(SSRFSecurityError, msg=f"Should block private IP: {ip_url}"):
                self.validator.validate_url(ip_url)

    def test_ssrf_blocking_cloud_metadata(self) -> None:
        metadata_targets = [
            "http://169.254.169.254/latest/meta-data/",
            "http://100.100.100.200/latest/meta-data/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://instance-data/latest/meta-data/",
        ]
        for meta_url in metadata_targets:
            with self.assertRaises(SSRFSecurityError, msg=f"Should block metadata URL: {meta_url}"):
                self.validator.validate_url(meta_url)

    def test_ssrf_blocking_loopback_and_localhost(self) -> None:
        loopback_urls = [
            "http://127.0.0.1:8000/",
            "http://127.0.1.1/",
            "http://localhost:3000/",
            "http://test.localhost/",
        ]
        for lb_url in loopback_urls:
            with self.assertRaises(SSRFSecurityError, msg=f"Should block loopback: {lb_url}"):
                self.validator.validate_url(lb_url)

    def test_allow_localhost_when_explicitly_enabled(self) -> None:
        # For mock unit testing, allow_localhost can be explicitly enabled
        norm = self.dev_validator.validate_url("http://127.0.0.1:8080/test")
        self.assertEqual(norm, "http://127.0.0.1:8080/test")

        norm2 = self.dev_validator.validate_url("http://localhost:5000/api")
        self.assertEqual(norm2, "http://localhost:5000/api")


class TestScopeValidator(unittest.TestCase):
    def test_same_domain_scope(self) -> None:
        scope = ScopeValidator(
            base_url="https://example.com/app",
            scope_mode=ScopeMode.SAME_DOMAIN,
        )
        self.assertTrue(scope.is_in_scope("https://example.com/app/dashboard"))
        self.assertTrue(scope.is_in_scope("https://example.com/login"))
        self.assertTrue(scope.is_in_scope("http://example.com/api"))

        # External domains rejected
        self.assertFalse(scope.is_in_scope("https://attacker.com/steal"))
        self.assertFalse(scope.is_in_scope("https://google.com/search"))
        # Subdomains rejected in SAME_DOMAIN mode
        self.assertFalse(scope.is_in_scope("https://sub.example.com/page"))

    def test_subdomains_scope(self) -> None:
        scope = ScopeValidator(
            base_url="https://example.com",
            scope_mode=ScopeMode.SUBDOMAINS,
        )
        self.assertTrue(scope.is_in_scope("https://example.com/"))
        self.assertTrue(scope.is_in_scope("https://api.example.com/v1"))
        self.assertTrue(scope.is_in_scope("https://auth.internal.example.com/login"))

        # Outside root domain rejected
        self.assertFalse(scope.is_in_scope("https://example.org"))
        self.assertFalse(scope.is_in_scope("https://fakeexample.com"))

    def test_path_prefix_scope(self) -> None:
        scope = ScopeValidator(
            base_url="https://example.com/portal",
            path_prefix="/portal",
        )
        self.assertTrue(scope.is_in_scope("https://example.com/portal/dashboard"))
        self.assertTrue(scope.is_in_scope("https://example.com/portal/settings"))
        # Outside path prefix rejected
        self.assertFalse(scope.is_in_scope("https://example.com/admin"))
        self.assertFalse(scope.is_in_scope("https://example.com/other"))

    def test_assert_in_scope_raises_exception(self) -> None:
        scope = ScopeValidator(base_url="https://example.com")
        self.assertEqual(
            scope.assert_in_scope("https://example.com/valid"),
            "https://example.com/valid",
        )
        with self.assertRaises(ScopeViolationError):
            scope.assert_in_scope("https://other-domain.com/evil")


if __name__ == "__main__":
    unittest.main()
