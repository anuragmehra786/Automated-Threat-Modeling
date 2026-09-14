"""
Models package for Automated Threat Modeling.

Exports shared Pydantic data contracts for input discovery evidence,
intermediate threat and risk models, and final security analysis outputs.
"""

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
)

__all__ = [
    # Enums
    "ObservationStatus",
    "SeverityLevel",
    "STRIDECategory",
    "AssetType",
    "AttackPathNodeType",
    # Framework Mappings
    "OWASPReference",
    "CWEReference",
    "MITREReference",
    "FrameworkMappings",
    # Discovery Input Contracts (Member 2)
    "Technology",
    "FormField",
    "Form",
    "EndpointParameter",
    "Endpoint",
    "Page",
    "SecurityHeaderInfo",
    "CookieInfo",
    "SecurityIndicators",
    "DiscoveryResult",
    # Security Intelligence & Threat Modeling (My Layer)
    "Asset",
    "Threat",
    "Risk",
    "AttackPathNode",
    "AttackPathEdge",
    "AttackPath",
    "Recommendation",
    # Output Contracts (Member 3)
    "SecurityAnalysisSummary",
    "SecurityAnalysis",
]
