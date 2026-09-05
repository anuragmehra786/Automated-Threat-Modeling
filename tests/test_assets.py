"""
Unit tests for the Security Asset Identification and Reasoning Layer (app/security/assets.py).

Validates:
1. Direct public endpoint asset identification (OBSERVED, confidence=0.95, criticality=LOW).
2. Authentication service inference from login-related evidence (INFERRED, criticality=HIGH).
3. Admin interface identification from explicit management endpoints/pages (INFERRED, criticality=HIGH).
4. Ambiguous paths (e.g. /dashboard) without admin markers do NOT trigger admin interface assets.
5. User data interface identification from registration/profile evidence (INFERRED, criticality=MEDIUM).
6. Session management mechanism inference with tightened epistemic wording (INFERRED, criticality=MEDIUM).
7. Technology stack component asset identification (OBSERVED, criticality=LOW).
8. Evidence list preservation on all generated assets.
9. Proper assignment of ObservationStatus (OBSERVED vs INFERRED).
10. Confidence scores remain valid and bounded between 0.0 and 1.0.
11. Duplicate evidence deduplication without asset count inflation.
12. Insufficient evidence does NOT create speculative internal assets (no imaginary databases, Redis, etc.).
13. Handling of empty or minimal DiscoveryResult payloads.
"""

import unittest
from datetime import datetime, timezone

from app.models.schemas import (
    Asset,
    AssetType,
    CookieInfo,
    DiscoveryResult,
    Endpoint,
    EndpointParameter,
    Form,
    FormField,
    ObservationStatus,
    Page,
    SecurityIndicators,
    SeverityLevel,
    Technology,
)
from app.security.assets import AssetIdentifier, identify_assets


class TestAssetIdentification(unittest.TestCase):
    """Test suite for asset reasoning logic."""

    def setUp(self):
        self.identifier = AssetIdentifier()

    def test_empty_discovery_result(self):
        """Test handling of empty or minimal discovery input."""
        discovery = DiscoveryResult(url="https://empty.example.com")
        assets = identify_assets(discovery)
        self.assertIsInstance(assets, list)
        self.assertEqual(len(assets), 0)

    def test_direct_technology_component_identification(self):
        """Test that detected technologies produce OBSERVED technology component assets."""
        discovery = DiscoveryResult(
            url="https://example.com",
            technologies=[
                Technology(
                    name="Nginx",
                    version="1.24.0",
                    categories=["Web Server", "Reverse Proxy"],
                    confidence=0.95,
                    evidence="Server: nginx/1.24.0 response header"
                ),
                Technology(
                    name="React",
                    version="18.2.0",
                    categories=["Frontend", "UI Framework"],
                    confidence=1.0,
                    evidence="Found react-dom script tag in DOM"
                )
            ]
        )

        assets = identify_assets(discovery)
        self.assertEqual(len(assets), 2)

        nginx_asset = next((a for a in assets if "nginx" in a.id), None)
        self.assertIsNotNone(nginx_asset)
        self.assertEqual(nginx_asset.type, AssetType.TECHNOLOGY_COMPONENT)
        self.assertEqual(nginx_asset.status, ObservationStatus.OBSERVED)
        self.assertEqual(nginx_asset.confidence, 0.95)
        self.assertEqual(nginx_asset.criticality, SeverityLevel.LOW)
        self.assertTrue(any("nginx" in ev.lower() for ev in nginx_asset.evidence))

        react_asset = next((a for a in assets if "react" in a.id), None)
        self.assertIsNotNone(react_asset)
        self.assertEqual(react_asset.status, ObservationStatus.OBSERVED)
        self.assertEqual(react_asset.confidence, 1.0)
        self.assertEqual(react_asset.criticality, SeverityLevel.LOW)

    def test_direct_public_endpoints_identification(self):
        """Test that endpoints and pages produce OBSERVED public endpoint assets with 0.95 confidence."""
        discovery = DiscoveryResult(
            url="https://example.com",
            endpoints=[
                Endpoint(
                    path="/api/v1/search",
                    method="GET",
                    parameters=[EndpointParameter(name="q", in_location="query", required=True)]
                ),
                Endpoint(
                    path="/api/v1/feedback",
                    method="POST",
                    parameters=[EndpointParameter(name="message", in_location="body", required=True)]
                )
            ],
            pages=[
                Page(
                    url="https://example.com/about",
                    path="/about",
                    status_code=200,
                    title="About Us"
                )
            ]
        )

        assets = identify_assets(discovery)
        endpoint_assets = [a for a in assets if a.type == AssetType.PUBLIC_ENDPOINT]
        self.assertEqual(len(endpoint_assets), 3)

        for asset in endpoint_assets:
            self.assertEqual(asset.status, ObservationStatus.OBSERVED)
            self.assertEqual(asset.confidence, 0.95)
            self.assertEqual(asset.criticality, SeverityLevel.LOW)
            self.assertTrue(len(asset.evidence) > 0)
            self.assertTrue(len(asset.associated_endpoints) > 0)

    def test_authentication_service_inference_with_form_and_endpoint(self):
        """Test authentication service inference when login form and POST endpoint are observed."""
        discovery = DiscoveryResult(
            url="https://example.com",
            endpoints=[
                Endpoint(
                    path="/api/v1/auth/login",
                    method="POST",
                    parameters=[
                        EndpointParameter(name="username", in_location="body", required=True),
                        EndpointParameter(name="password", in_location="body", required=True)
                    ]
                )
            ],
            forms=[
                Form(
                    form_id="user-login",
                    action="/api/v1/auth/login",
                    method="POST",
                    form_type="login",
                    fields=[
                        FormField(name="user_email", field_type="email", required=True),
                        FormField(name="user_pass", field_type="password", required=True)
                    ],
                    page_url="https://example.com/login"
                )
            ]
        )

        assets = identify_assets(discovery)
        auth_asset = next((a for a in assets if a.type == AssetType.AUTHENTICATION_SERVICE), None)

        self.assertIsNotNone(auth_asset)
        self.assertEqual(auth_asset.id, "asset-auth-service")
        self.assertEqual(auth_asset.status, ObservationStatus.INFERRED)
        self.assertEqual(auth_asset.criticality, SeverityLevel.HIGH)
        # Strong evidence (form + password field + login route) gives high confidence
        self.assertGreaterEqual(auth_asset.confidence, 0.90)
        self.assertLessEqual(auth_asset.confidence, 1.0)
        self.assertIn("/api/v1/auth/login", auth_asset.associated_endpoints)
        self.assertTrue(any("password" in ev.lower() for ev in auth_asset.evidence))

    def test_explicit_admin_interface_inference(self):
        """Test administrative interface inference from explicit admin endpoints and pages."""
        discovery = DiscoveryResult(
            url="https://example.com",
            endpoints=[
                Endpoint(path="/admin/users", method="GET"),
                Endpoint(path="/wp-admin", method="GET")
            ],
            pages=[
                Page(
                    url="https://example.com/admin/login",
                    path="/admin/login",
                    status_code=200,
                    title="Administrator Control Panel"
                )
            ]
        )

        assets = identify_assets(discovery)
        admin_asset = next((a for a in assets if a.type == AssetType.ADMIN_INTERFACE), None)

        self.assertIsNotNone(admin_asset)
        self.assertEqual(admin_asset.id, "asset-admin-interface")
        self.assertEqual(admin_asset.status, ObservationStatus.INFERRED)
        self.assertEqual(admin_asset.criticality, SeverityLevel.HIGH)
        self.assertGreaterEqual(admin_asset.confidence, 0.85)
        self.assertTrue(any("/admin" in ep for ep in admin_asset.associated_endpoints))
        self.assertTrue(any("admin" in ev.lower() for ev in admin_asset.evidence))

    def test_ambiguous_dashboard_does_not_create_admin_asset(self):
        """Test that ambiguous paths like /dashboard without admin markers do NOT trigger admin interface assets."""
        discovery = DiscoveryResult(
            url="https://example.com",
            endpoints=[
                Endpoint(path="/dashboard", method="GET")
            ],
            pages=[
                Page(
                    url="https://example.com/dashboard",
                    path="/dashboard",
                    status_code=200,
                    title="Customer Dashboard"
                )
            ]
        )

        assets = identify_assets(discovery)
        admin_asset = next((a for a in assets if a.type == AssetType.ADMIN_INTERFACE), None)
        self.assertIsNone(admin_asset, "Ambiguous /dashboard path should not create an ADMIN_INTERFACE asset")

    def test_user_data_interface_inference(self):
        """Test user data & account interface inference from registration and profile evidence."""
        discovery = DiscoveryResult(
            url="https://example.com",
            endpoints=[
                Endpoint(path="/api/v1/users/profile", method="GET"),
                Endpoint(path="/api/v1/account/settings", method="PUT")
            ],
            forms=[
                Form(
                    form_id="register-form",
                    action="/api/v1/users/register",
                    method="POST",
                    form_type="register",
                    fields=[
                        FormField(name="email", field_type="email", required=True),
                        FormField(name="full_name", field_type="text", required=True)
                    ]
                )
            ]
        )

        assets = identify_assets(discovery)
        user_asset = next((a for a in assets if a.type == AssetType.USER_DATA_INTERFACE), None)

        self.assertIsNotNone(user_asset)
        self.assertEqual(user_asset.status, ObservationStatus.INFERRED)
        self.assertEqual(user_asset.criticality, SeverityLevel.MEDIUM)
        self.assertIn("/api/v1/users/profile", user_asset.associated_endpoints)
        self.assertTrue(len(user_asset.evidence) >= 2)

    def test_session_management_inference_and_wording(self):
        """Test session management inference from discovered security cookies with tightened wording."""
        discovery = DiscoveryResult(
            url="https://example.com",
            security_indicators=SecurityIndicators(
                cookies=[
                    CookieInfo(name="session_id", secure=True, httponly=True, samesite="Lax"),
                    CookieInfo(name="jwt_token", secure=True, httponly=True, samesite="Strict")
                ]
            )
        )

        assets = identify_assets(discovery)
        session_asset = next((a for a in assets if a.type == AssetType.SESSION_MANAGEMENT), None)

        self.assertIsNotNone(session_asset)
        self.assertEqual(session_asset.status, ObservationStatus.INFERRED)
        self.assertEqual(session_asset.confidence, 0.90)
        self.assertEqual(session_asset.criticality, SeverityLevel.MEDIUM)
        # Check tightened epistemic wording in description and evidence
        self.assertIn("unknown", session_asset.description.lower())
        self.assertTrue(any("session_id" in ev for ev in session_asset.evidence))
        self.assertTrue(any("unknown" in ev.lower() for ev in session_asset.evidence))

    def test_no_speculative_internal_assets_created(self):
        """
        Verify that generic discovery evidence does NOT create speculative internal assets
        such as phantom databases, redis caches, or private microservices.
        """
        discovery = DiscoveryResult(
            url="https://example.com",
            technologies=[Technology(name="Bootstrap", categories=["CSS Framework"])],
            pages=[Page(url="https://example.com/faq", path="/faq", status_code=200, title="FAQ")]
        )

        assets = identify_assets(discovery)
        # Check that no database, internal service, or sensitive data assets are created
        disallowed_types = {AssetType.DATABASE, AssetType.INTERNAL_SERVICE, AssetType.SENSITIVE_DATA}
        for asset in assets:
            self.assertNotIn(asset.type, disallowed_types, f"Speculative asset created: {asset.name} ({asset.type})")

    def test_deduplication_and_evidence_consolidation(self):
        """Test that redundant discovery signals are merged into consolidated assets with combined evidence."""
        discovery = DiscoveryResult(
            url="https://example.com",
            endpoints=[
                Endpoint(path="/login", method="GET"),
                Endpoint(path="/login", method="POST")
            ],
            forms=[
                Form(
                    action="/login",
                    method="POST",
                    form_type="login",
                    fields=[FormField(name="password", field_type="password")]
                )
            ]
        )

        assets = identify_assets(discovery)
        auth_assets = [a for a in assets if a.type == AssetType.AUTHENTICATION_SERVICE]
        # Should be exactly 1 consolidated authentication service asset, not multiple duplicates
        self.assertEqual(len(auth_assets), 1)
        # Evidence should contain all observations
        self.assertGreaterEqual(len(auth_assets[0].evidence), 2)

    def test_all_asset_confidences_bounded(self):
        """Verify that all generated assets strictly respect 0.0 <= confidence <= 1.0."""
        discovery = DiscoveryResult(
            url="https://example.com",
            technologies=[Technology(name="Django", confidence=0.88)],
            endpoints=[Endpoint(path="/api/v1/checkout", method="POST")],
            forms=[Form(action="/api/v1/checkout", form_type="payment", fields=[FormField(name="card")])],
            security_indicators=SecurityIndicators(cookies=[CookieInfo(name="sessionid")])
        )

        assets = identify_assets(discovery)
        self.assertTrue(len(assets) > 0)
        for asset in assets:
            self.assertGreaterEqual(asset.confidence, 0.0)
            self.assertLessEqual(asset.confidence, 1.0)
            self.assertIsInstance(asset.status, ObservationStatus)
            self.assertTrue(len(asset.evidence) > 0)


if __name__ == "__main__":
    unittest.main()
