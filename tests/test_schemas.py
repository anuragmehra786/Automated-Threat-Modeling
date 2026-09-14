"""
Unit tests for shared Pydantic data contracts and schemas.

Validates:
- DiscoveryResult schema validation and extensibility
- SecurityAnalysis schema validation and serialization
- Asset, Threat, Risk, AttackPath, and Recommendation models
- Deterministic risk scoring calculation and severity mapping
- Validation rejection of invalid inputs (out of bounds scores, negative confidence)
- JSON serialization/deserialization integrity
"""

import unittest
from datetime import datetime
# pyrefly: ignore [missing-import]
from pydantic import ValidationError

from app.models.schemas import (
    Asset,
    AssetType,
    AttackPath,
    AttackPathEdge,
    AttackPathNode,
    AttackPathNodeType,
    CookieInfo,
    CWEReference,
    DiscoveryResult,
    Endpoint,
    EndpointParameter,
    Form,
    FormField,
    FrameworkMappings,
    MITREReference,
    ObservationStatus,
    OWASPReference,
    Page,
    Recommendation,
    Risk,
    SecurityAnalysis,
    SecurityAnalysisSummary,
    SecurityHeaderInfo,
    SecurityIndicators,
    SeverityLevel,
    STRIDECategory,
    Technology,
    Threat,
)


class TestSchemas(unittest.TestCase):
    """Test suite for Pydantic contracts and data validation."""

    def test_observation_status_values(self):
        """Ensure observation status enums follow the core design principle."""
        self.assertEqual(ObservationStatus.OBSERVED.value, "OBSERVED")
        self.assertEqual(ObservationStatus.INFERRED.value, "INFERRED")
        self.assertEqual(ObservationStatus.UNKNOWN.value, "UNKNOWN")

    def test_discovery_result_valid(self):
        """Verify DiscoveryResult contract handles valid discovery data."""
        discovery = DiscoveryResult(
            url="https://example.com",
            technologies=[
                Technology(
                    name="React",
                    version="18.2.0",
                    categories=["Frontend", "UI Library"],
                    confidence=0.95,
                    evidence="Discovered react-dom bundle script tag"
                ),
                Technology(
                    name="Express",
                    version="4.18.2",
                    categories=["Backend", "Web Framework"],
                    confidence=0.85,
                    evidence="X-Powered-By: Express header"
                )
            ],
            pages=[
                Page(
                    url="https://example.com/login",
                    path="/login",
                    status_code=200,
                    title="User Login",
                    links=["https://example.com/register", "https://example.com/forgot-password"]
                )
            ],
            endpoints=[
                Endpoint(
                    path="/api/v1/login",
                    method="POST",
                    parameters=[
                        EndpointParameter(name="username", in_location="body", required=True),
                        EndpointParameter(name="password", in_location="body", required=True)
                    ],
                    authentication_required=ObservationStatus.UNKNOWN
                )
            ],
            forms=[
                Form(
                    form_id="login-form",
                    action="/api/v1/login",
                    method="POST",
                    form_type="login",
                    fields=[
                        FormField(name="username", field_type="text", required=True),
                        FormField(name="password", field_type="password", required=True)
                    ],
                    page_url="https://example.com/login"
                )
            ],
            security_indicators=SecurityIndicators(
                headers={"server": "nginx", "x-powered-by": "Express"},
                security_headers_summary=[
                    SecurityHeaderInfo(
                        name="Content-Security-Policy",
                        present=False,
                        recommendation="Implement strict Content-Security-Policy"
                    ),
                    SecurityHeaderInfo(
                        name="Strict-Transport-Security",
                        present=True,
                        value="max-age=31536000; includeSubDomains"
                    )
                ],
                cookies=[
                    CookieInfo(name="session_id", secure=True, httponly=True, samesite="Lax")
                ],
                https_enforced=True,
                ssl_certificate_valid=True,
                robots_txt_present=True,
                cors_wildcard=False
            )
        )

        self.assertEqual(discovery.url, "https://example.com")
        self.assertEqual(len(discovery.technologies), 2)
        self.assertEqual(len(discovery.forms), 1)
        self.assertEqual(discovery.forms[0].fields[1].field_type, "password")
        self.assertTrue(discovery.security_indicators.https_enforced)

    def test_discovery_extensibility_extra_fields_ignored(self):
        """Ensure extra fields from Member 2's discovery engine do not raise ValidationError."""
        raw_payload = {
            "url": "https://example.com",
            "future_experimental_field": "some_value",
            "technologies": [
                {
                    "name": "Django",
                    "unexpected_meta": 123
                }
            ]
        }
        discovery = DiscoveryResult.model_validate(raw_payload)
        self.assertEqual(discovery.url, "https://example.com")
        self.assertEqual(discovery.technologies[0].name, "Django")

    def test_asset_creation_and_validation(self):
        """Test Asset schema validation and bounds."""
        asset = Asset(
            id="asset-auth-01",
            name="User Authentication Module",
            type=AssetType.AUTHENTICATION_SERVICE,
            description="Handles username and password verification for standard accounts",
            status=ObservationStatus.INFERRED,
            confidence=0.90,
            evidence=["Observed POST /api/v1/login endpoint", "Discovered login form with password field"],
            associated_endpoints=["/api/v1/login"]
        )
        self.assertEqual(asset.type, AssetType.AUTHENTICATION_SERVICE)
        self.assertEqual(asset.status, ObservationStatus.INFERRED)
        self.assertEqual(asset.confidence, 0.90)

        # Negative test: confidence out of bounds
        with self.assertRaises(ValidationError):
            Asset(
                id="asset-invalid",
                name="Invalid",
                type=AssetType.OTHER,
                description="Invalid confidence",
                confidence=1.5  # Invalid > 1.0
            )

    def test_deterministic_risk_scoring(self):
        """Test deterministic score computation and severity mapping."""
        # Test 1: Minimum boundary (1 * 1 = 1 -> INFORMATIONAL)
        risk_min = Risk(
            id="risk-01",
            threat_id="threat-01",
            likelihood=1,
            impact=1,
            rationale="Very low likelihood and informational impact"
        )
        self.assertEqual(risk_min.score, 1)
        self.assertEqual(risk_min.severity, SeverityLevel.INFORMATIONAL)

        # Test 2: Low boundary (2 * 2 = 4 -> LOW)
        risk_low = Risk(
            id="risk-02",
            threat_id="threat-02",
            likelihood=2,
            impact=2,
            rationale="Low likelihood and low impact"
        )
        self.assertEqual(risk_low.score, 4)
        self.assertEqual(risk_low.severity, SeverityLevel.LOW)

        # Test 3: Medium boundary (3 * 3 = 9 -> MEDIUM)
        risk_med = Risk(
            id="risk-03",
            threat_id="threat-03",
            likelihood=3,
            impact=3,
            rationale="Medium likelihood and impact"
        )
        self.assertEqual(risk_med.score, 9)
        self.assertEqual(risk_med.severity, SeverityLevel.MEDIUM)

        # Test 4: High boundary (4 * 4 = 16 -> HIGH)
        risk_high = Risk(
            id="risk-04",
            threat_id="threat-04",
            likelihood=4,
            impact=4,
            rationale="High likelihood and impact"
        )
        self.assertEqual(risk_high.score, 16)
        self.assertEqual(risk_high.severity, SeverityLevel.HIGH)

        # Test 5: Maximum boundary (5 * 5 = 25 -> CRITICAL)
        risk_crit = Risk(
            id="risk-05",
            threat_id="threat-05",
            likelihood=5,
            impact=5,
            rationale="Very high likelihood and critical impact"
        )
        self.assertEqual(risk_crit.score, 25)
        self.assertEqual(risk_crit.severity, SeverityLevel.CRITICAL)

        # Negative test: Out of bounds likelihood (0 or 6)
        with self.assertRaises(ValidationError):
            Risk(
                id="risk-invalid",
                threat_id="threat-01",
                likelihood=6,
                impact=3,
                rationale="Invalid likelihood"
            )

        with self.assertRaises(ValidationError):
            Risk(
                id="risk-invalid",
                threat_id="threat-01",
                likelihood=2,
                impact=0,
                rationale="Invalid impact"
            )

    def test_attack_path_schema(self):
        """Test AttackPath graph representation."""
        node1 = AttackPathNode(
            id="node-internet",
            label="Public Internet",
            node_type=AttackPathNodeType.INTERNET,
            description="External unauthenticated access vector"
        )
        node2 = AttackPathNode(
            id="node-login",
            label="POST /api/v1/login",
            node_type=AttackPathNodeType.PUBLIC_ENDPOINT,
            reference_id="/api/v1/login"
        )
        node3 = AttackPathNode(
            id="node-auth-asset",
            label="User Accounts",
            node_type=AttackPathNodeType.PROTECTED_ASSET,
            reference_id="asset-auth-01"
        )
        edge1 = AttackPathEdge(
            source="node-internet",
            target="node-login",
            relationship="submits_unauthenticated_requests"
        )
        edge2 = AttackPathEdge(
            source="node-login",
            target="node-auth-asset",
            relationship="authenticates_against"
        )

        attack_path = AttackPath(
            id="path-01",
            name="Credential Stuffing to User Account Compromise",
            description="External actor leverages unthrottled login endpoint to test leaked credentials",
            entry_point="node-internet",
            target_asset_id="asset-auth-01",
            nodes=[node1, node2, node3],
            edges=[edge1, edge2],
            steps=[
                "Actor accesses public login route from internet",
                "Submits rapid automated requests with credential lists",
                "Attempts account takeover without rate limiting defenses"
            ],
            status=ObservationStatus.INFERRED,
            confidence=0.80,
            path_risk_score=16,
            associated_threat_ids=["threat-stride-01"]
        )

        self.assertEqual(len(attack_path.nodes), 3)
        self.assertEqual(len(attack_path.edges), 2)
        self.assertEqual(attack_path.status, ObservationStatus.INFERRED)
        self.assertEqual(attack_path.path_risk_score, 16)

    def test_full_security_analysis_serialization(self):
        """Test full SecurityAnalysis output contract creation and JSON round-trip serialization."""
        analysis = SecurityAnalysis(
            analysis_id="analysis-20260905-001",
            target_url="https://example.com",
            status="completed",
            summary=SecurityAnalysisSummary(
                total_assets_identified=1,
                total_threats_identified=1,
                highest_severity=SeverityLevel.HIGH,
                risk_distribution={
                    "CRITICAL": 0,
                    "HIGH": 1,
                    "MEDIUM": 0,
                    "LOW": 0,
                    "INFORMATIONAL": 0
                },
                key_observations=[
                    "Discovered unthrottled authentication endpoint /api/v1/login",
                    "Missing Content-Security-Policy header"
                ]
            ),
            assets=[
                Asset(
                    id="asset-auth-01",
                    name="Authentication Service",
                    type=AssetType.AUTHENTICATION_SERVICE,
                    description="User login and session management endpoint",
                    status=ObservationStatus.INFERRED,
                    confidence=0.90,
                    evidence=["POST /api/v1/login form found with password field"],
                    associated_endpoints=["/api/v1/login"]
                )
            ],
            threats=[
                Threat(
                    id="threat-stride-01",
                    title="Potential Credential Stuffing / Brute Force on Login Endpoint",
                    category=STRIDECategory.SPOOFING,
                    description="Absence of observable rate limiting or CAPTCHA may allow automated credential guessing",
                    affected_asset_ids=["asset-auth-01"],
                    status=ObservationStatus.INFERRED,
                    confidence=0.75,
                    evidence=["Discovered POST /api/v1/login with password input", "No rate-limiting headers observed"],
                    framework_mappings=FrameworkMappings(
                        owasp=[
                            OWASPReference(
                                code="A07:2021",
                                name="Identification and Authentication Failures",
                                url="https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/"
                            )
                        ],
                        cwe=[
                            CWEReference(
                                cwe_id="CWE-307",
                                name="Improper Restriction of Excessive Authentication Attempts",
                                url="https://cwe.mitre.org/data/definitions/307.html"
                            )
                        ],
                        mitre=[
                            MITREReference(
                                technique_id="T1110",
                                technique_name="Brute Force",
                                tactic="Credential Access",
                                url="https://attack.mitre.org/techniques/T1110/"
                            )
                        ]
                    ),
                    prerequisites=["Publicly reachable login endpoint", "List of candidate credentials"],
                    rag_sources=["owasp_a07_auth_failures.md", "cwe_307_brute_force.json"]
                )
            ],
            risks=[
                Risk(
                    id="risk-01",
                    threat_id="threat-stride-01",
                    likelihood=4,
                    impact=4,
                    rationale="High exposure on public login route combined with significant account takeover impact."
                )
            ],
            attack_paths=[
                AttackPath(
                    id="path-01",
                    name="External Credential Stuffing Path",
                    description="Internet -> Login Route -> Auth Service",
                    entry_point="node-internet",
                    target_asset_id="asset-auth-01",
                    nodes=[
                        AttackPathNode(id="node-internet", label="Internet", node_type=AttackPathNodeType.INTERNET),
                        AttackPathNode(id="node-login", label="/api/v1/login", node_type=AttackPathNodeType.PUBLIC_ENDPOINT),
                        AttackPathNode(id="node-auth", label="Auth Service", node_type=AttackPathNodeType.PROTECTED_ASSET)
                    ],
                    edges=[
                        AttackPathEdge(source="node-internet", target="node-login", relationship="sends_requests"),
                        AttackPathEdge(source="node-login", target="node-auth", relationship="attempts_auth")
                    ],
                    steps=["Send brute force requests", "Compromise account"],
                    confidence=0.75,
                    path_risk_score=16,
                    associated_threat_ids=["threat-stride-01"]
                )
            ],
            recommendations=[
                Recommendation(
                    id="rec-01",
                    title="Enforce IP & Account Rate Limiting and Account Lockout",
                    description="Limit rapid authentication attempts per IP address and account username to mitigate automated attacks.",
                    priority=SeverityLevel.HIGH,
                    threat_ids=["threat-stride-01"],
                    asset_ids=["asset-auth-01"],
                    remediation_guidance="Configure reverse proxy or application-level rate limiting (e.g., 5 attempts per minute).",
                    security_controls=["Rate Limiting", "Account Lockout", "WAF Rules"]
                )
            ]
        )

        # Validate JSON serialization & deserialization round trip
        json_data = analysis.model_dump_json(indent=2)
        self.assertIsInstance(json_data, str)

        reloaded = SecurityAnalysis.model_validate_json(json_data)
        self.assertEqual(reloaded.analysis_id, "analysis-20260905-001")
        self.assertEqual(len(reloaded.threats), 1)
        self.assertEqual(reloaded.risks[0].score, 16)
        self.assertEqual(reloaded.risks[0].severity, SeverityLevel.HIGH)
        self.assertEqual(reloaded.threats[0].framework_mappings.owasp[0].code, "A07:2021")


if __name__ == "__main__":
    unittest.main()
