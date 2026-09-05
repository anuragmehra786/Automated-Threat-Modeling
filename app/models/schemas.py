"""
Shared Data Contracts and Pydantic Schemas for Automated Threat Modeling.

This module defines the primary data schemas used across the application:
1. Discovery Contracts (Input from Member 2's discovery engine):
   - DiscoveryResult, Technology, Page, Endpoint, Form, SecurityIndicators
2. Security Intelligence & Threat Modeling Contracts (AI/Security layer):
   - Asset, Threat, Risk, AttackPath, Recommendation
3. Final Output Contract (Delivered to Member 3's frontend):
   - SecurityAnalysis, SecurityAnalysisSummary
4. Supporting Enums and Framework Reference Models:
   - ObservationStatus (OBSERVED, INFERRED, UNKNOWN)
   - SeverityLevel, STRIDECategory, AssetType, AttackPathNodeType
   - OWASPReference, CWEReference, MITREReference, FrameworkMappings

Design Principles:
- Explicit distinction between OBSERVED, INFERRED, and UNKNOWN status.
- Strict separation of Confidence (0.0 - 1.0) from Risk Score (1 - 25).
- Extensible schemas with backward/forward compatibility.
- Clean serialization and strong type-safety.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =====================================================================
# Core Enums
# =====================================================================

class ObservationStatus(str, Enum):
    """
    Distinguishes fact from inference and uncertainty.
    
    - OBSERVED: Directly present and confirmed in the discovery evidence (e.g., a form at /login).
    - INFERRED: Reasoned or deduced by the AI/rules engine (e.g., authentication service exists).
    - UNKNOWN: Cannot be confirmed or verified from available evidence (e.g., whether MFA is enabled).
    """
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


class SeverityLevel(str, Enum):
    """
    Standard severity ratings for risks, threats, and recommendations.
    """
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


class STRIDECategory(str, Enum):
    """
    STRIDE threat modeling categories.
    """
    SPOOFING = "Spoofing"
    TAMPERING = "Tampering"
    REPUDIATION = "Repudiation"
    INFORMATION_DISCLOSURE = "Information Disclosure"
    DENIAL_OF_SERVICE = "Denial of Service"
    ELEVATION_OF_PRIVILEGE = "Elevation of Privilege"


class AssetType(str, Enum):
    """
    Categorization of identified software and architectural assets.
    """
    PUBLIC_ENDPOINT = "public_endpoint"
    API_ENDPOINT = "api_endpoint"
    AUTHENTICATION_SERVICE = "authentication_service"
    AUTHORIZATION_SERVICE = "authorization_service"
    ADMIN_INTERFACE = "admin_interface"
    ADMIN_FUNCTIONALITY = "admin_functionality"
    USER_DATA_INTERFACE = "user_data_interface"
    USER_ACCOUNT = "user_account"
    PERSONAL_DATA = "personal_data"
    SESSION_MANAGEMENT = "session_management"
    SESSION_STORE = "session_store"
    SENSITIVE_DATA = "sensitive_data"
    DATABASE = "database"
    PAYMENT_INTERFACE = "payment_interface"
    PAYMENT_GATEWAY = "payment_gateway"
    INTERNAL_SERVICE = "internal_service"
    APPLICATION_COMPONENT = "application_component"
    TECHNOLOGY_COMPONENT = "technology_component"
    THIRD_PARTY_INTEGRATION = "third_party_integration"
    OTHER = "other"


class AttackPathNodeType(str, Enum):
    """
    Semantic types of nodes within a potential attack path graph.
    """
    INTERNET = "internet"
    PUBLIC_ENDPOINT = "public_endpoint"
    POTENTIAL_THREAT = "potential_threat"
    APPLICATION_COMPONENT = "application_component"
    PROTECTED_ASSET = "protected_asset"
    EXTERNAL_SERVICE = "external_service"


# =====================================================================
# Framework Reference Models (OWASP, CWE, MITRE ATT&CK)
# =====================================================================

class OWASPReference(BaseModel):
    """Reference to an OWASP Top 10 category or standard."""
    model_config = ConfigDict(extra="ignore")

    code: str = Field(..., description="OWASP code (e.g., 'A01:2021', 'A03:2021')", examples=["A01:2021"])
    name: str = Field(..., description="Category name (e.g., 'Broken Access Control')", examples=["Broken Access Control"])
    description: Optional[str] = Field(None, description="Summary of the OWASP category")
    url: Optional[str] = Field(None, description="Authoritative reference URL")


class CWEReference(BaseModel):
    """Reference to a Common Weakness Enumeration (CWE)."""
    model_config = ConfigDict(extra="ignore")

    cwe_id: str = Field(..., description="CWE identifier (e.g., 'CWE-79', 'CWE-89')", examples=["CWE-79"])
    name: str = Field(..., description="CWE Weakness name", examples=["Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')"])
    description: Optional[str] = Field(None, description="Summary of the CWE weakness")
    url: Optional[str] = Field(None, description="Authoritative CWE reference URL")


class MITREReference(BaseModel):
    """Reference to a MITRE ATT&CK technique."""
    model_config = ConfigDict(extra="ignore")

    technique_id: str = Field(..., description="MITRE technique ID (e.g., 'T1190', 'T1059')", examples=["T1190"])
    technique_name: str = Field(..., description="Technique name (e.g., 'Exploit Public-Facing Application')", examples=["Exploit Public-Facing Application"])
    tactic: Optional[str] = Field(None, description="ATT&CK Tactic (e.g., 'Initial Access', 'Execution')", examples=["Initial Access"])
    description: Optional[str] = Field(None, description="Summary of the technique")
    url: Optional[str] = Field(None, description="Authoritative MITRE reference URL")


class FrameworkMappings(BaseModel):
    """Container for mapped cybersecurity framework references."""
    model_config = ConfigDict(extra="ignore")

    owasp: List[OWASPReference] = Field(default_factory=list, description="Associated OWASP Top 10 categories")
    cwe: List[CWEReference] = Field(default_factory=list, description="Associated CWE weaknesses")
    mitre: List[MITREReference] = Field(default_factory=list, description="Associated MITRE ATT&CK techniques")


# =====================================================================
# Discovery Contract Models (Input from Member 2)
# =====================================================================

class Technology(BaseModel):
    """Identified technology, framework, or library."""
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Technology name (e.g., 'React', 'Django', 'Nginx')", examples=["React"])
    version: Optional[str] = Field(None, description="Detected version string if identifiable", examples=["18.2.0"])
    categories: List[str] = Field(default_factory=list, description="Categories (e.g., ['Web Framework', 'Frontend'])")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence from 0.0 to 1.0")
    evidence: Optional[str] = Field(None, description="Header, script, or DOM indicator used to identify technology")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional discovery metadata")


class FormField(BaseModel):
    """Input field within a discovered HTML form."""
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Field name or ID attribute")
    field_type: str = Field(default="text", description="HTML input type (e.g., 'text', 'password', 'hidden')")
    required: bool = Field(default=False, description="Whether field has required attribute")
    value: Optional[str] = Field(None, description="Default or preset value if observed")


class Form(BaseModel):
    """Discovered web form and input parameters."""
    model_config = ConfigDict(extra="ignore")

    form_id: Optional[str] = Field(None, description="Form identifier or name attribute")
    action: str = Field(..., description="Form action target URL or path", examples=["/api/v1/login"])
    method: str = Field(default="POST", description="HTTP method used by form (e.g., 'GET', 'POST')", examples=["POST"])
    fields: List[FormField] = Field(default_factory=list, description="Discovered input fields")
    form_type: Optional[str] = Field(None, description="Inferred form purpose (e.g., 'login', 'register', 'search', 'contact')")
    page_url: Optional[str] = Field(None, description="URL of the page where this form was discovered")


class EndpointParameter(BaseModel):
    """Query, path, body, or header parameter for an endpoint."""
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Parameter name")
    in_location: str = Field(default="query", description="Parameter location: 'query', 'path', 'body', 'header'")
    required: bool = Field(default=False, description="Whether parameter is required")
    param_type: Optional[str] = Field(None, description="Data type if inferred (e.g., 'string', 'integer')")


class Endpoint(BaseModel):
    """Discovered HTTP endpoint or API route."""
    model_config = ConfigDict(extra="ignore")

    path: str = Field(..., description="Path or relative URL of the endpoint", examples=["/api/v1/users"])
    method: str = Field(default="GET", description="HTTP method (e.g., 'GET', 'POST', 'PUT', 'DELETE')", examples=["GET"])
    parameters: List[EndpointParameter] = Field(default_factory=list, description="Accepted parameters")
    authentication_required: ObservationStatus = Field(
        default=ObservationStatus.UNKNOWN,
        description="Whether authentication appears required (OBSERVED, INFERRED, or UNKNOWN)"
    )
    headers: Dict[str, str] = Field(default_factory=dict, description="Observed headers")
    content_type: Optional[str] = Field(None, description="Response or request content type")


class Page(BaseModel):
    """Discovered webpage within the target application."""
    model_config = ConfigDict(extra="ignore")

    url: str = Field(..., description="Full URL of the discovered page")
    path: Optional[str] = Field(None, description="Relative URL path (e.g., '/dashboard')")
    status_code: Optional[int] = Field(None, description="HTTP response status code", examples=[200])
    title: Optional[str] = Field(None, description="HTML page title tag content")
    content_type: Optional[str] = Field(None, description="Content-Type header value")
    links: List[str] = Field(default_factory=list, description="Outbound or internal links discovered on this page")
    forms: List[Form] = Field(default_factory=list, description="Forms discovered on this page")
    endpoints: List[str] = Field(default_factory=list, description="API endpoints referenced on this page")


class SecurityHeaderInfo(BaseModel):
    """Analysis of a specific security response header."""
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Header name (e.g., 'Content-Security-Policy', 'Strict-Transport-Security')")
    present: bool = Field(..., description="Whether the header is present in responses")
    value: Optional[str] = Field(None, description="Observed header value")
    recommendation: Optional[str] = Field(None, description="Brief note on posture or recommended value")


class CookieInfo(BaseModel):
    """Discovered cookie and its security attributes."""
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Cookie name")
    secure: Optional[bool] = Field(None, description="Whether Secure flag is set")
    httponly: Optional[bool] = Field(None, description="Whether HttpOnly flag is set")
    samesite: Optional[str] = Field(None, description="SameSite attribute: 'Strict', 'Lax', 'None', or None")


class SecurityIndicators(BaseModel):
    """Observed security posture indicators collected during discovery."""
    model_config = ConfigDict(extra="ignore")

    headers: Dict[str, Optional[str]] = Field(default_factory=dict, description="Raw HTTP response headers")
    security_headers_summary: List[SecurityHeaderInfo] = Field(default_factory=list, description="Evaluated security headers")
    cookies: List[CookieInfo] = Field(default_factory=list, description="Observed cookies and security flags")
    https_enforced: Optional[bool] = Field(None, description="Whether HTTP automatically redirects to HTTPS")
    ssl_certificate_valid: Optional[bool] = Field(None, description="Whether TLS certificate is valid and trusted")
    robots_txt_present: Optional[bool] = Field(None, description="Whether robots.txt exists")
    sitemap_present: Optional[bool] = Field(None, description="Whether sitemap.xml exists")
    cors_wildcard: Optional[bool] = Field(None, description="Whether Access-Control-Allow-Origin is '*'")
    raw_indicators: Dict[str, Any] = Field(default_factory=dict, description="Additional raw indicators from discovery")


class DiscoveryResult(BaseModel):
    """
    Primary Input Contract from Member 2's discovery engine.
    
    Contains structured evidence gathered from authorized reconnaissance
    of a target web application.
    """
    model_config = ConfigDict(extra="ignore")

    url: str = Field(..., description="Target base URL analyzed", examples=["https://example.com"])
    scan_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp of discovery scan")
    technologies: List[Technology] = Field(default_factory=list, description="Identified technologies and frameworks")
    pages: List[Page] = Field(default_factory=list, description="Discovered pages and routes")
    endpoints: List[Endpoint] = Field(default_factory=list, description="Discovered endpoints and APIs")
    forms: List[Form] = Field(default_factory=list, description="Discovered HTML forms")
    security_indicators: SecurityIndicators = Field(default_factory=SecurityIndicators, description="Observed security indicators")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible metadata from discovery runner")


# =====================================================================
# Security Intelligence & Threat Modeling Models (My Scope)
# =====================================================================

class Asset(BaseModel):
    """
    Identified software, data, or architectural asset.
    """
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique asset identifier", examples=["asset-auth-service-01"])
    name: str = Field(..., description="Human-readable asset name", examples=["User Authentication Service"])
    type: AssetType = Field(..., description="Asset classification type", examples=[AssetType.AUTHENTICATION_SERVICE])
    description: str = Field(..., description="Clear explanation of the asset and its role")
    status: ObservationStatus = Field(
        default=ObservationStatus.INFERRED,
        description="Whether this asset was directly observed or inferred from evidence"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in asset identification (0.0 to 1.0, separate from risk)"
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Specific discovery evidence supporting the identification of this asset"
    )
    associated_endpoints: List[str] = Field(
        default_factory=list,
        description="Paths or endpoints related to this asset (e.g., ['/api/v1/login'])"
    )
    criticality: Optional[SeverityLevel] = Field(
        None,
        description="Inherent business or security criticality of this asset"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional asset metadata")


class Threat(BaseModel):
    """
    Identified potential threat relevant to the target application's observed characteristics.
    """
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique threat identifier", examples=["threat-stride-01"])
    title: str = Field(..., description="Concise threat title", examples=["Potential Credential Stuffing on Login Form"])
    category: STRIDECategory = Field(..., description="STRIDE threat category", examples=[STRIDECategory.SPOOFING])
    description: str = Field(..., description="Detailed description of the potential threat scenario")
    affected_asset_ids: List[str] = Field(
        default_factory=list,
        description="List of Asset IDs impacted by this threat"
    )
    status: ObservationStatus = Field(
        default=ObservationStatus.INFERRED,
        description="Distinguishes observed posture from inferred threat potential"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in threat relevance based on evidence (0.0 to 1.0, separate from risk)"
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Discovery evidence that justifies why this threat is relevant"
    )
    framework_mappings: FrameworkMappings = Field(
        default_factory=FrameworkMappings,
        description="Relevant OWASP, CWE, and MITRE ATT&CK references"
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Prerequisites or attacker capabilities required for this threat"
    )
    rag_sources: List[str] = Field(
        default_factory=list,
        description="Knowledge base sources and identifiers used to enrich this threat"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional threat metadata")


class Risk(BaseModel):
    """
    Deterministic risk scoring model.
    
    Risk Score = Likelihood (1-5) * Impact (1-5) -> Resulting Score: 1 to 25.
    Score is mapped to SeverityLevel via explicit deterministic thresholds:
      - 20..25 -> CRITICAL
      - 15..19 -> HIGH
      - 8..14  -> MEDIUM
      - 4..7   -> LOW
      - 1..3   -> INFORMATIONAL
    """
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique risk identifier", examples=["risk-01"])
    threat_id: str = Field(..., description="Associated Threat ID", examples=["threat-stride-01"])
    likelihood: int = Field(
        ...,
        ge=1,
        le=5,
        description="Likelihood score: 1 (Very Low), 2 (Low), 3 (Medium), 4 (High), 5 (Very High)"
    )
    impact: int = Field(
        ...,
        ge=1,
        le=5,
        description="Impact score: 1 (Very Low), 2 (Low), 3 (Medium), 4 (High), 5 (Very High)"
    )
    score: int = Field(
        default=0,
        ge=1,
        le=25,
        description="Calculated deterministic risk score (likelihood * impact, range 1-25)"
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.INFORMATIONAL,
        description="Deterministic severity mapped from calculated score"
    )
    rationale: str = Field(
        ...,
        description="Clear explanation of the justification for assigned likelihood and impact"
    )
    mitigating_factors: List[str] = Field(
        default_factory=list,
        description="Observed mitigating controls that reduce likelihood or impact"
    )

    @model_validator(mode="before")
    @classmethod
    def calculate_score_and_severity(cls, data: Any) -> Any:
        """Ensure score and severity are deterministically computed if omitted or validate consistency."""
        if isinstance(data, dict):
            likelihood = data.get("likelihood")
            impact = data.get("impact")
            if likelihood is not None and impact is not None:
                calculated_score = int(likelihood) * int(impact)
                data["score"] = calculated_score
                
                # Deterministic severity mapping
                if calculated_score >= 20:
                    data["severity"] = SeverityLevel.CRITICAL
                elif calculated_score >= 15:
                    data["severity"] = SeverityLevel.HIGH
                elif calculated_score >= 8:
                    data["severity"] = SeverityLevel.MEDIUM
                elif calculated_score >= 4:
                    data["severity"] = SeverityLevel.LOW
                else:
                    data["severity"] = SeverityLevel.INFORMATIONAL
        return data


class AttackPathNode(BaseModel):
    """Node in a potential attack path graph."""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique node identifier", examples=["node-internet", "node-login-endpoint"])
    label: str = Field(..., description="Human-readable node label", examples=["Public Internet", "POST /api/v1/login"])
    node_type: AttackPathNodeType = Field(..., description="Semantic category of this node")
    description: Optional[str] = Field(None, description="Detailed node description")
    reference_id: Optional[str] = Field(None, description="Associated asset_id, threat_id, or endpoint path if applicable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional node metadata")


class AttackPathEdge(BaseModel):
    """Directed relationship between nodes in an attack path graph."""
    model_config = ConfigDict(extra="ignore")

    source: str = Field(..., description="Source node ID", examples=["node-internet"])
    target: str = Field(..., description="Target node ID", examples=["node-login-endpoint"])
    relationship: str = Field(..., description="Relationship description", examples=["submits_unauthenticated_request"])
    description: Optional[str] = Field(None, description="Contextual explanation of the transition")


class AttackPath(BaseModel):
    """
    Representation of a potential attack path from entry point to protected asset.
    
    Represents defensive modeling of potential attack vector sequences, NOT confirmed exploits.
    """
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique attack path identifier", examples=["path-auth-bypass-01"])
    name: str = Field(..., description="Descriptive name of the potential path", examples=["External Attacker to User Account via Login Vector"])
    description: str = Field(..., description="Overview of the attack path scenario")
    entry_point: str = Field(..., description="Starting node ID or entry vector", examples=["node-internet"])
    target_asset_id: str = Field(..., description="Target Asset ID or protected node ID", examples=["asset-auth-service-01"])
    nodes: List[AttackPathNode] = Field(default_factory=list, description="Graph nodes comprising the path")
    edges: List[AttackPathEdge] = Field(default_factory=list, description="Directed edges connecting path nodes")
    steps: List[str] = Field(default_factory=list, description="Step-by-step description of the potential attack flow")
    status: ObservationStatus = Field(
        default=ObservationStatus.INFERRED,
        description="Status of the attack path (always INFERRED or UNKNOWN, never OBSERVED unless proven)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the feasibility of this potential path (0.0 to 1.0, separate from risk)"
    )
    path_risk_score: Optional[int] = Field(
        None,
        ge=1,
        le=25,
        description="Aggregated risk score for this path (1-25)"
    )
    associated_threat_ids: List[str] = Field(
        default_factory=list,
        description="Threat IDs relevant to this attack path"
    )


class Recommendation(BaseModel):
    """
    Actionable defensive security recommendation.
    """
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique recommendation identifier", examples=["rec-01"])
    title: str = Field(..., description="Concise recommendation title", examples=["Implement Rate Limiting on Login Form"])
    description: str = Field(..., description="Comprehensive explanation of the recommendation")
    priority: SeverityLevel = Field(..., description="Remediation priority level", examples=[SeverityLevel.HIGH])
    threat_ids: List[str] = Field(default_factory=list, description="Threat IDs addressed by this recommendation")
    asset_ids: List[str] = Field(default_factory=list, description="Asset IDs protected by this recommendation")
    remediation_guidance: str = Field(..., description="Concrete technical instructions for remediation")
    security_controls: List[str] = Field(
        default_factory=list,
        description="Recommended defensive security controls (e.g., ['Rate Limiting', 'CAPTCHA', 'Account Lockout'])"
    )
    framework_references: FrameworkMappings = Field(
        default_factory=FrameworkMappings,
        description="Associated framework guidance"
    )


# =====================================================================
# Final Output Contract Models (Delivered to Member 3's Frontend)
# =====================================================================

class SecurityAnalysisSummary(BaseModel):
    """
    High-level executive summary of the security analysis results.
    """
    model_config = ConfigDict(extra="ignore")

    total_assets_identified: int = Field(default=0, description="Total number of assets identified")
    total_threats_identified: int = Field(default=0, description="Total number of potential threats identified")
    highest_severity: SeverityLevel = Field(default=SeverityLevel.INFORMATIONAL, description="Highest severity level found among risks")
    risk_distribution: Dict[str, int] = Field(
        default_factory=lambda: {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "INFORMATIONAL": 0
        },
        description="Count of risks by severity level"
    )
    key_observations: List[str] = Field(
        default_factory=list,
        description="High-level narrative findings and key posture observations"
    )


class SecurityAnalysis(BaseModel):
    """
    Primary Output Contract for Member 3's Frontend Presentation Layer.
    
    Contains the complete, structured results of the AI threat modeling,
    risk scoring, knowledge base enrichment, and attack-path analysis.
    """
    model_config = ConfigDict(extra="ignore")

    analysis_id: str = Field(..., description="Unique analysis run identifier", examples=["analysis-20260905-001"])
    target_url: str = Field(..., description="Target web application URL", examples=["https://example.com"])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp when analysis was generated")
    status: str = Field(default="completed", description="Analysis execution status: 'completed', 'partial', 'failed'")
    summary: SecurityAnalysisSummary = Field(..., description="Executive summary statistics and key observations")
    assets: List[Asset] = Field(default_factory=list, description="Identified application assets")
    threats: List[Threat] = Field(default_factory=list, description="Identified potential threats")
    risks: List[Risk] = Field(default_factory=list, description="Deterministic risk calculations for identified threats")
    attack_paths: List[AttackPath] = Field(default_factory=list, description="Constructed potential attack paths")
    recommendations: List[Recommendation] = Field(default_factory=list, description="Actionable defensive recommendations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional execution and environment metadata")
