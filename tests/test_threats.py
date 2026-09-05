"""
Unit tests for the Deterministic STRIDE Threat Modeling Layer (app/security/threats.py).

Validates:
1. Authentication service produces appropriate Spoofing & DoS threats with defensive wording.
2. Admin interface produces appropriate Elevation of Privilege & Tampering threats.
3. User data interface produces Information Disclosure & Tampering threats.
4. Session management produces relevant session spoofing & disclosure threats.
5. Public endpoint produces sensible potential threats (Tampering for state-changing HTTP methods).
6. Technology-only components do NOT automatically produce vulnerability claims.
7. Threat status is consistently assigned as INFERRED (threat hypothesis, not confirmed vulnerability).
8. Confidence is valid and bounded between 0.0 and 1.0.
9. Evidence from affected assets is strictly preserved in generated threats.
10. Prerequisites contain defensive reachable conditions, NOT operational attack-planning recipes.
11. No specific password hashing algorithms (bcrypt/argon2) are speculatively assumed.
12. Duplicate threat signals are consolidated safely per asset category without cross-category pollution.
13. Empty or minimal asset lists produce empty threat lists (no speculative phantom threats).
"""

import unittest

from app.models.schemas import (
    Asset,
    AssetType,
    ObservationStatus,
    SeverityLevel,
    STRIDECategory,
    Threat,
)
from app.security.threats import ThreatModeler, model_threats


class TestThreatModeling(unittest.TestCase):
    """Test suite for STRIDE threat modeling engine."""

    def setUp(self):
        self.modeler = ThreatModeler()

    def test_empty_assets_produces_empty_threats(self):
        """Test that empty asset lists produce zero threats without errors."""
        threats = model_threats([])
        self.assertEqual(len(threats), 0)

    def test_auth_service_threat_generation(self):
        """Test that Authentication Service assets produce Spoofing and DoS STRIDE threats with defensive wording."""
        auth_asset = Asset(
            id="asset-auth-service-01",
            name="User Authentication Service",
            type=AssetType.AUTHENTICATION_SERVICE,
            description="Handles login verification and user identity issuance.",
            status=ObservationStatus.INFERRED,
            confidence=0.95,
            evidence=["Observed POST /api/v1/auth/login", "Discovered login form with password field"],
            associated_endpoints=["/api/v1/auth/login"],
            criticality=SeverityLevel.HIGH
        )

        threats = model_threats([auth_asset])
        self.assertGreaterEqual(len(threats), 2)

        # Check for Spoofing threat
        spoof_threat = next((t for t in threats if t.category == STRIDECategory.SPOOFING), None)
        self.assertIsNotNone(spoof_threat)
        self.assertEqual(spoof_threat.status, ObservationStatus.INFERRED)
        self.assertEqual(spoof_threat.title, "Potential Authentication Abuse and Identity Spoofing")
        self.assertIn("asset-auth-service-01", spoof_threat.affected_asset_ids)
        self.assertGreaterEqual(spoof_threat.confidence, 0.70)
        self.assertTrue(any("login" in ev.lower() for ev in spoof_threat.evidence))
        # Verify defensive prerequisite wording
        self.assertIn("Authentication endpoint is network reachable", spoof_threat.prerequisites)
        self.assertNotIn("Candidate user credential list", spoof_threat.prerequisites)

        # Check for Denial of Service threat
        dos_threat = next((t for t in threats if t.category == STRIDECategory.DENIAL_OF_SERVICE), None)
        self.assertIsNotNone(dos_threat)
        self.assertEqual(dos_threat.status, ObservationStatus.INFERRED)
        self.assertEqual(dos_threat.title, "Potential Resource Exhaustion on Authentication Processing")
        self.assertIn("asset-auth-service-01", dos_threat.affected_asset_ids)
        # Verify no speculative hashing algorithms are mentioned
        self.assertNotIn("bcrypt", dos_threat.description.lower())
        self.assertNotIn("argon2", dos_threat.description.lower())

    def test_admin_interface_threat_generation(self):
        """Test that Admin Interface assets produce Elevation of Privilege and Tampering threats."""
        admin_asset = Asset(
            id="asset-admin-01",
            name="Administrative Management Interface",
            type=AssetType.ADMIN_INTERFACE,
            description="Administrative control portal exposing configuration functions.",
            status=ObservationStatus.INFERRED,
            confidence=0.90,
            evidence=["Discovered administrative portal at /admin/dashboard"],
            associated_endpoints=["/admin/dashboard"],
            criticality=SeverityLevel.HIGH
        )

        threats = model_threats([admin_asset])
        self.assertGreaterEqual(len(threats), 2)

        # Check Elevation of Privilege
        elev_threat = next((t for t in threats if t.category == STRIDECategory.ELEVATION_OF_PRIVILEGE), None)
        self.assertIsNotNone(elev_threat)
        self.assertEqual(elev_threat.status, ObservationStatus.INFERRED)
        self.assertIn("asset-admin-01", elev_threat.affected_asset_ids)
        self.assertIn("Administrative interface is reachable", elev_threat.prerequisites)

        # Check Tampering
        tamper_threat = next((t for t in threats if t.category == STRIDECategory.TAMPERING), None)
        self.assertIsNotNone(tamper_threat)
        self.assertIn("asset-admin-01", tamper_threat.affected_asset_ids)

    def test_user_data_interface_threat_generation(self):
        """Test that User Data Interface assets produce Information Disclosure and Tampering threats."""
        user_asset = Asset(
            id="asset-user-data-01",
            name="User Profile Interface",
            type=AssetType.USER_DATA_INTERFACE,
            description="Interface handling customer profile settings.",
            status=ObservationStatus.INFERRED,
            confidence=0.85,
            evidence=["Observed GET /api/v1/users/profile"],
            associated_endpoints=["/api/v1/users/profile"],
            criticality=SeverityLevel.MEDIUM
        )

        threats = model_threats([user_asset])
        self.assertGreaterEqual(len(threats), 2)

        # Check Information Disclosure
        infodisc_threat = next((t for t in threats if t.category == STRIDECategory.INFORMATION_DISCLOSURE), None)
        self.assertIsNotNone(infodisc_threat)
        self.assertEqual(infodisc_threat.status, ObservationStatus.INFERRED)
        self.assertIn("asset-user-data-01", infodisc_threat.affected_asset_ids)
        self.assertIn("User data interface is reachable", infodisc_threat.prerequisites)

        # Check Tampering
        tamper_threat = next((t for t in threats if t.category == STRIDECategory.TAMPERING), None)
        self.assertIsNotNone(tamper_threat)
        self.assertIn("asset-user-data-01", tamper_threat.affected_asset_ids)

    def test_session_management_threat_generation(self):
        """Test that Session Management assets produce Spoofing and Information Disclosure threats."""
        session_asset = Asset(
            id="asset-session-01",
            name="Session Management Mechanism",
            type=AssetType.SESSION_MANAGEMENT,
            description="Inferred session management based on session cookies.",
            status=ObservationStatus.INFERRED,
            confidence=0.90,
            evidence=["Observed cookie 'session_id' with Secure=True, HttpOnly=True"],
            associated_endpoints=[],
            criticality=SeverityLevel.MEDIUM
        )

        threats = model_threats([session_asset])
        self.assertGreaterEqual(len(threats), 2)

        spoof_threat = next((t for t in threats if t.category == STRIDECategory.SPOOFING), None)
        self.assertIsNotNone(spoof_threat)
        self.assertIn("asset-session-01", spoof_threat.affected_asset_ids)
        self.assertIn("Session management mechanism issues state tokens", spoof_threat.prerequisites)

        disc_threat = next((t for t in threats if t.category == STRIDECategory.INFORMATION_DISCLOSURE), None)
        self.assertIsNotNone(disc_threat)
        self.assertIn("asset-session-01", disc_threat.affected_asset_ids)

    def test_payment_interface_threat_generation(self):
        """Test that Payment Interface assets produce Tampering and Repudiation threats."""
        payment_asset = Asset(
            id="asset-payment-01",
            name="Payment & Checkout Interface",
            type=AssetType.PAYMENT_INTERFACE,
            description="Financial transaction handling workflow.",
            status=ObservationStatus.INFERRED,
            confidence=0.85,
            evidence=["Observed endpoint POST /api/v1/checkout"],
            associated_endpoints=["/api/v1/checkout"],
            criticality=SeverityLevel.HIGH
        )

        threats = model_threats([payment_asset])
        self.assertGreaterEqual(len(threats), 2)

        tamper_threat = next((t for t in threats if t.category == STRIDECategory.TAMPERING), None)
        self.assertIsNotNone(tamper_threat)
        self.assertIn("asset-payment-01", tamper_threat.affected_asset_ids)
        self.assertIn("Payment workflow is reachable", tamper_threat.prerequisites)

        repudiation_threat = next((t for t in threats if t.category == STRIDECategory.REPUDIATION), None)
        self.assertIsNotNone(repudiation_threat)
        self.assertIn("asset-payment-01", repudiation_threat.affected_asset_ids)

    def test_public_endpoint_threat_generation_rules(self):
        """Test that public endpoints only generate threats for state-changing methods (POST/PUT/DELETE), not static GET."""
        post_endpoint_asset = Asset(
            id="asset-post-ep",
            name="Public Endpoint: POST /api/feedback",
            type=AssetType.PUBLIC_ENDPOINT,
            description="Public POST endpoint.",
            status=ObservationStatus.OBSERVED,
            confidence=0.95,
            evidence=["Directly observed POST /api/feedback"],
            associated_endpoints=["/api/feedback"],
            criticality=SeverityLevel.LOW,
            metadata={"method": "POST"}
        )

        get_endpoint_asset = Asset(
            id="asset-get-page",
            name="Public Page: About Us",
            type=AssetType.PUBLIC_ENDPOINT,
            description="Static public about page.",
            status=ObservationStatus.OBSERVED,
            confidence=0.95,
            evidence=["Directly observed webpage /about"],
            associated_endpoints=["/about"],
            criticality=SeverityLevel.LOW,
            metadata={"method": "GET"}
        )

        # POST endpoint should produce Tampering threat
        post_threats = model_threats([post_endpoint_asset])
        self.assertEqual(len(post_threats), 1)
        self.assertEqual(post_threats[0].category, STRIDECategory.TAMPERING)

        # Static GET page should NOT produce automatic noisy threats
        get_threats = model_threats([get_endpoint_asset])
        self.assertEqual(len(get_threats), 0)

    def test_technology_components_do_not_produce_vulnerabilities(self):
        """Verify that technology components alone do NOT produce speculative vulnerability claims."""
        tech_asset = Asset(
            id="asset-tech-nginx",
            name="Nginx Component",
            type=AssetType.TECHNOLOGY_COMPONENT,
            description="Detected Nginx 1.24.0 web server.",
            status=ObservationStatus.OBSERVED,
            confidence=0.95,
            evidence=["Server: nginx/1.24.0 header"],
            associated_endpoints=[],
            criticality=SeverityLevel.LOW
        )

        threats = model_threats([tech_asset])
        self.assertEqual(len(threats), 0, "Technology component should not automatically produce vulnerability threats")

    def test_deduplication_consolidates_same_type_assets_without_cross_category_merge(self):
        """Test that multiple assets matching the same threat category consolidate, while distinct asset types remain separate."""
        auth_asset_1 = Asset(
            id="asset-auth-01",
            name="Login Endpoint",
            type=AssetType.AUTHENTICATION_SERVICE,
            description="Login route 1",
            status=ObservationStatus.INFERRED,
            confidence=0.80,
            evidence=["Route /login"],
            associated_endpoints=["/login"]
        )
        auth_asset_2 = Asset(
            id="asset-auth-02",
            name="API Auth Endpoint",
            type=AssetType.AUTHENTICATION_SERVICE,
            description="Login route 2",
            status=ObservationStatus.INFERRED,
            confidence=0.90,
            evidence=["Route /api/auth"],
            associated_endpoints=["/api/auth"]
        )
        session_asset = Asset(
            id="asset-session-01",
            name="Session Store",
            type=AssetType.SESSION_MANAGEMENT,
            description="Session cookie",
            status=ObservationStatus.INFERRED,
            confidence=0.85,
            evidence=["Cookie session_id"]
        )

        threats = model_threats([auth_asset_1, auth_asset_2, session_asset])
        
        # Check auth spoofing is consolidated into 1
        auth_spoof = [t for t in threats if t.category == STRIDECategory.SPOOFING and "asset-auth-01" in t.affected_asset_ids]
        self.assertEqual(len(auth_spoof), 1)
        self.assertIn("asset-auth-01", auth_spoof[0].affected_asset_ids)
        self.assertIn("asset-auth-02", auth_spoof[0].affected_asset_ids)
        self.assertEqual(auth_spoof[0].confidence, 0.90)

        # Check session spoofing remains separate from auth spoofing
        session_spoof = [t for t in threats if t.category == STRIDECategory.SPOOFING and "asset-session-01" in t.affected_asset_ids]
        self.assertEqual(len(session_spoof), 1)
        self.assertEqual(session_spoof[0].affected_asset_ids, ["asset-session-01"])

    def test_all_threat_confidences_bounded_and_inferred(self):
        """Verify that all generated threats have bounded confidence (0.0 to 1.0) and status INFERRED."""
        assets = [
            Asset(id="a1", name="Auth", type=AssetType.AUTHENTICATION_SERVICE, description="Auth", status=ObservationStatus.INFERRED, confidence=0.9, evidence=["e1"]),
            Asset(id="a2", name="Admin", type=AssetType.ADMIN_INTERFACE, description="Admin", status=ObservationStatus.INFERRED, confidence=0.9, evidence=["e2"]),
            Asset(id="a3", name="Session", type=AssetType.SESSION_MANAGEMENT, description="Session", status=ObservationStatus.INFERRED, confidence=0.9, evidence=["e3"]),
            Asset(id="a4", name="UserData", type=AssetType.USER_DATA_INTERFACE, description="UserData", status=ObservationStatus.INFERRED, confidence=0.9, evidence=["e4"]),
            Asset(id="a5", name="Payment", type=AssetType.PAYMENT_INTERFACE, description="Payment", status=ObservationStatus.INFERRED, confidence=0.9, evidence=["e5"]),
        ]

        threats = model_threats(assets)
        self.assertTrue(len(threats) > 0)
        for t in threats:
            self.assertEqual(t.status, ObservationStatus.INFERRED)
            self.assertGreaterEqual(t.confidence, 0.0)
            self.assertLessEqual(t.confidence, 1.0)
            self.assertIsInstance(t.category, STRIDECategory)
            self.assertTrue(len(t.evidence) > 0)
            self.assertTrue(len(t.affected_asset_ids) > 0)


if __name__ == "__main__":
    unittest.main()
