"""
Deterministic Cybersecurity Threat Modeling Layer.

Transforms identified application assets (Asset) into modeled potential security
threats (Threat) using the STRIDE threat-modeling methodology.

Core Principles:
1. STRIDE Methodology:
   - Spoofing (S)
   - Tampering (T)
   - Repudiation (R)
   - Information Disclosure (I)
   - Denial of Service (D)
   - Elevation of Privilege (E)
2. Hypothesis, NOT Confirmed Vulnerability:
   - Modeled threats represent potential security risks inherent to asset functionality.
   - Threats are marked as INFERRED.
   - Never claims that an asset is confirmed to be vulnerable without direct exploit evidence.
3. Strict Technology Boundary:
   - Detecting a technology component (e.g., Nginx, React) does NOT automatically generate
     vulnerability claims or speculative CVEs.
4. Evidence Preservation & Traceability:
   - Every modeled threat preserves the asset evidence and documents why the STRIDE category applies.
5. Deduplication:
   - Related signals are consolidated into single, cohesive threat models per asset category.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set
from app.models.schemas import (
    Asset,
    AssetType,
    DiscoveryResult,
    FrameworkMappings,
    ObservationStatus,
    STRIDECategory,
    Threat,
)


def _sanitize_id(text: str) -> str:
    """Helper to convert names/categories into URL-safe identifier slugs."""
    clean = re.sub(r"[^a-zA-Z0-9_-]", "-", text.strip().lower())
    clean = re.sub(r"-+", "-", clean).strip("-")
    return clean or "unknown"


class ThreatModeler:
    """
    Deterministic STRIDE threat-modeling engine.
    
    Examines identified application assets and generates structured Threat objects
    representing potential security concerns requiring review and verification.
    """

    def model_threats(self, assets: List[Asset], discovery: Optional[DiscoveryResult] = None) -> List[Threat]:
        """
        Main entry point for threat modeling.
        
        Analyzes the provided list of assets and returns a consolidated list of
        modeled STRIDE threats with preserved evidence.
        """
        if not assets:
            return []

        raw_threats: List[Threat] = []

        for asset in assets:
            if asset.type == AssetType.AUTHENTICATION_SERVICE:
                raw_threats.extend(self._model_auth_threats(asset))
            elif asset.type == AssetType.ADMIN_INTERFACE:
                raw_threats.extend(self._model_admin_threats(asset))
            elif asset.type == AssetType.USER_DATA_INTERFACE:
                raw_threats.extend(self._model_user_data_threats(asset))
            elif asset.type == AssetType.SESSION_MANAGEMENT:
                raw_threats.extend(self._model_session_threats(asset))
            elif asset.type == AssetType.PAYMENT_INTERFACE:
                raw_threats.extend(self._model_payment_threats(asset))
            elif asset.type == AssetType.PUBLIC_ENDPOINT:
                raw_threats.extend(self._model_public_endpoint_threats(asset))
            elif asset.type == AssetType.TECHNOLOGY_COMPONENT:
                # Technology components do NOT automatically produce vulnerability claims
                continue

        return self._deduplicate_and_consolidate(raw_threats)

    def _model_auth_threats(self, asset: Asset) -> List[Threat]:
        """
        Generates STRIDE threats for Authentication Services:
        - Spoofing: Authentication abuse & identity spoofing.
        - Denial of Service: Resource exhaustion on authentication processing.
        """
        threats: List[Threat] = []

        # 1. Spoofing Threat
        spoofing_evidence = list(asset.evidence) + [
            "Authentication services represent an identity boundary where authentication abuse and spoofing threats are applicable.",
            "Applicable defenses such as rate limiting, MFA, account lockout, or anti-automation controls are unknown and require verification."
        ]
        threats.append(
            Threat(
                id=f"threat-auth-spoofing-{_sanitize_id(asset.id)}",
                title="Potential Authentication Abuse and Identity Spoofing",
                category=STRIDECategory.SPOOFING,
                description=(
                    "Authentication interfaces represent an identity boundary where unauthorized authentication attempts "
                    "could result in identity impersonation if appropriate verification and protection controls are insufficient. "
                    "(Applicable controls such as rate limiting, MFA, account lockout, and anti-automation are unknown and require verification)."
                ),
                affected_asset_ids=[asset.id],
                status=ObservationStatus.INFERRED,
                confidence=round(min(0.90, max(0.1, asset.confidence)), 2),
                evidence=spoofing_evidence,
                prerequisites=["Authentication endpoint is network reachable"],
                framework_mappings=FrameworkMappings(),
                metadata={"asset_type": asset.type.value}
            )
        )

        # 2. Denial of Service Threat
        dos_evidence = list(asset.evidence) + [
            "Authentication verification routines require server-side processing; applicable throttling and resource controls are unknown."
        ]
        threats.append(
            Threat(
                id=f"threat-auth-dos-{_sanitize_id(asset.id)}",
                title="Potential Resource Exhaustion on Authentication Processing",
                category=STRIDECategory.DENIAL_OF_SERVICE,
                description=(
                    "Authentication request processing can consume significant server computation; in the absence of request "
                    "throttling or concurrency limits, high request volume could degrade service responsiveness. "
                    "(Applicable rate limiting and resource governance controls are unknown)."
                ),
                affected_asset_ids=[asset.id],
                status=ObservationStatus.INFERRED,
                confidence=round(min(0.85, max(0.1, asset.confidence)), 2),
                evidence=dos_evidence,
                prerequisites=["Authentication endpoint is network reachable"],
                framework_mappings=FrameworkMappings(),
                metadata={"asset_type": asset.type.value}
            )
        )

        return threats

    def _model_admin_threats(self, asset: Asset) -> List[Threat]:
        """
        Generates STRIDE threats for Administrative Interfaces:
        - Elevation of Privilege: Unauthorized access to privileged controls.
        - Tampering: Unauthorized alteration of system configurations.
        """
        threats: List[Threat] = []

        # 1. Elevation of Privilege Threat
        elev_evidence = list(asset.evidence) + [
            "Administrative interfaces represent privileged system control surfaces.",
            "Enforcement of role-based access control (RBAC) and strong session verification is unknown."
        ]
        threats.append(
            Threat(
                id=f"threat-admin-priv-esc-{_sanitize_id(asset.id)}",
                title="Potential Unauthorized Privilege Escalation to Administrative Controls",
                category=STRIDECategory.ELEVATION_OF_PRIVILEGE,
                description=(
                    "An unauthorized actor or low-privileged user might attempt to access administrative management "
                    "functions if role-based authorization controls or access restrictions are bypassed. "
                    "(Actual authorization implementation is unknown and requires verification)."
                ),
                affected_asset_ids=[asset.id],
                status=ObservationStatus.INFERRED,
                confidence=round(min(0.90, max(0.1, asset.confidence * 0.95)), 2),
                evidence=elev_evidence,
                prerequisites=[
                    "Administrative interface is reachable",
                    "Authorization controls govern access to privileged functions"
                ],
                framework_mappings=FrameworkMappings(),
                metadata={"asset_type": asset.type.value}
            )
        )

        # 2. Tampering Threat
        tamper_evidence = list(asset.evidence) + [
            "Administrative consoles typically allow modification of application configurations and operational data."
        ]
        threats.append(
            Threat(
                id=f"threat-admin-tampering-{_sanitize_id(asset.id)}",
                title="Potential Unauthorized Modification of System Configurations",
                category=STRIDECategory.TAMPERING,
                description=(
                    "If an unauthorized actor gains access to administrative capabilities, they may modify critical "
                    "application settings, user permissions, or system configurations."
                ),
                affected_asset_ids=[asset.id],
                status=ObservationStatus.INFERRED,
                confidence=round(min(0.85, max(0.1, asset.confidence * 0.90)), 2),
                evidence=tamper_evidence,
                prerequisites=["Administrative interface is reachable"],
                framework_mappings=FrameworkMappings(),
                metadata={"asset_type": asset.type.value}
            )
        )

        return threats

    def _model_user_data_threats(self, asset: Asset) -> List[Threat]:
        """
        Generates STRIDE threats for User Data & Account Interfaces:
        - Information Disclosure: Unauthorized access to personal profile data.
        - Tampering: Unauthorized modification of user account records.
        """
        threats: List[Threat] = []

        # 1. Information Disclosure Threat
        infodisc_evidence = list(asset.evidence) + [
            "User data interfaces handle personally identifiable information (PII) and account records.",
            "Object-level authorization controls (e.g. prevention of IDOR) are unknown and require verification."
        ]
        threats.append(
            Threat(
                id=f"threat-userdata-infodisc-{_sanitize_id(asset.id)}",
                title="Potential Unauthorized Disclosure of User Profile Data",
                category=STRIDECategory.INFORMATION_DISCLOSURE,
                description=(
                    "Endpoints exposing user profile or account data may disclose sensitive personal information "
                    "if object-level access controls or authorization boundaries are not strictly enforced."
                ),
                affected_asset_ids=[asset.id],
                status=ObservationStatus.INFERRED,
                confidence=round(min(0.85, max(0.1, asset.confidence * 0.90)), 2),
                evidence=infodisc_evidence,
                prerequisites=["User data interface is reachable"],
                framework_mappings=FrameworkMappings(),
                metadata={"asset_type": asset.type.value}
            )
        )

        # 2. Tampering Threat
        tamper_evidence = list(asset.evidence) + [
            "Account endpoints accepting updates represent mutation surfaces for user profile records."
        ]
        threats.append(
            Threat(
                id=f"threat-userdata-tampering-{_sanitize_id(asset.id)}",
                title="Potential Unauthorized Modification of User Profile Information",
                category=STRIDECategory.TAMPERING,
                description=(
                    "An actor might attempt to alter another user's profile information, contact settings, or account attributes "
                    "if incoming data update requests are not verified against the caller's authorized identity."
                ),
                affected_asset_ids=[asset.id],
                status=ObservationStatus.INFERRED,
                confidence=round(min(0.80, max(0.1, asset.confidence * 0.85)), 2),
                evidence=tamper_evidence,
                prerequisites=["User data interface is reachable"],
                framework_mappings=FrameworkMappings(),
                metadata={"asset_type": asset.type.value}
            )
        )

        return threats

    def _model_session_threats(self, asset: Asset) -> List[Threat]:
        """
        Generates STRIDE threats for Session Management:
        - Spoofing: Session token interception or replay.
        - Information Disclosure: Exposure of session tokens in transit or client contexts.
        """
        threats: List[Threat] = []

        # 1. Spoofing Threat
        spoof_evidence = list(asset.evidence) + [
            "Session management mechanisms issue tokens that represent authenticated user identity.",
            "Backend token entropy, rotation policies, and invalidation mechanisms are unknown."
        ]
        threats.append(
            Threat(
                id=f"threat-session-spoofing-{_sanitize_id(asset.id)}",
                title="Potential Session Hijacking and User Impersonation",
                category=STRIDECategory.SPOOFING,
                description=(
                    "If session identifiers are intercepted, replayed, or predictable, an unauthorized party "
                    "could impersonate an established user session without knowing their primary credentials."
                ),
                affected_asset_ids=[asset.id],
                status=ObservationStatus.INFERRED,
                confidence=round(min(0.85, max(0.1, asset.confidence * 0.90)), 2),
                evidence=spoof_evidence,
                prerequisites=["Session management mechanism issues state tokens"],
                framework_mappings=FrameworkMappings(),
                metadata={"asset_type": asset.type.value}
            )
        )

        # 2. Information Disclosure Threat
        disc_evidence = list(asset.evidence) + [
            "Session identifiers stored in cookies or headers represent sensitive authorization credentials."
        ]
        threats.append(
            Threat(
                id=f"threat-session-infodisc-{_sanitize_id(asset.id)}",
                title="Potential Insecure Session Identifier Exposure",
                category=STRIDECategory.INFORMATION_DISCLOSURE,
                description=(
                    "Session identifiers could be exposed to unauthorized parties if transport security is not strictly "
                    "enforced or if client-side scripts can access session cookies."
                ),
                affected_asset_ids=[asset.id],
                status=ObservationStatus.INFERRED,
                confidence=round(min(0.80, max(0.1, asset.confidence * 0.85)), 2),
                evidence=disc_evidence,
                prerequisites=["Session tokens are transmitted across HTTP communications"],
                framework_mappings=FrameworkMappings(),
                metadata={"asset_type": asset.type.value}
            )
        )

        return threats

    def _model_payment_threats(self, asset: Asset) -> List[Threat]:
        """
        Generates STRIDE threats for Payment & Billing Interfaces:
        - Tampering: Transaction or checkout parameter manipulation.
        - Repudiation: Disputed transactions without non-repudiation audit trails.
        """
        threats: List[Threat] = []

        # 1. Tampering Threat
        tamper_evidence = list(asset.evidence) + [
            "Financial checkout endpoints accept order, pricing, or billing parameters.",
            "Server-side price verification and order integrity controls are unknown."
        ]
        threats.append(
            Threat(
                id=f"threat-payment-tampering-{_sanitize_id(asset.id)}",
                title="Potential Manipulation of Transaction or Order Parameters",
                category=STRIDECategory.TAMPERING,
                description=(
                    "If client-submitted transaction values (e.g., quantities, pricing, discount tokens) are not strictly "
                    "re-validated on the server, an actor might attempt parameter tampering during the checkout workflow."
                ),
                affected_asset_ids=[asset.id],
                status=ObservationStatus.INFERRED,
                confidence=round(min(0.85, max(0.1, asset.confidence * 0.90)), 2),
                evidence=tamper_evidence,
                prerequisites=["Payment workflow is reachable"],
                framework_mappings=FrameworkMappings(),
                metadata={"asset_type": asset.type.value}
            )
        )

        # 2. Repudiation Threat
        repudiation_evidence = list(asset.evidence) + [
            "Financial transaction interfaces require verifiable audit logs and transaction confirmation records."
        ]
        threats.append(
            Threat(
                id=f"threat-payment-repudiation-{_sanitize_id(asset.id)}",
                title="Potential Repudiation of Financial Transactions",
                category=STRIDECategory.REPUDIATION,
                description=(
                    "Without comprehensive transaction logging, timestamping, and non-repudiation controls, "
                    "disputing unauthorized or modified financial transactions becomes difficult."
                ),
                affected_asset_ids=[asset.id],
                status=ObservationStatus.INFERRED,
                confidence=round(min(0.80, max(0.1, asset.confidence * 0.80)), 2),
                evidence=repudiation_evidence,
                prerequisites=["Payment workflow is reachable"],
                framework_mappings=FrameworkMappings(),
                metadata={"asset_type": asset.type.value}
            )
        )

        return threats

    def _model_public_endpoint_threats(self, asset: Asset) -> List[Threat]:
        """
        Generates STRIDE threats for Public Endpoints:
        Only generates Tampering threats for state-changing HTTP methods (POST, PUT, DELETE, PATCH).
        Avoids generating noisy threats for simple read-only static pages.
        """
        threats: List[Threat] = []
        method = asset.metadata.get("method", "GET").upper()

        if method in ("POST", "PUT", "DELETE", "PATCH"):
            evidence_desc = list(asset.evidence) + [
                f"Observed public endpoint accepts state-changing HTTP {method} requests.",
                "Server-side input validation, sanitization, and schema enforcement are unknown."
            ]
            threats.append(
                Threat(
                    id=f"threat-endpoint-tampering-{_sanitize_id(asset.id)}",
                    title=f"Potential Input Tampering on Public {method} Endpoint",
                    category=STRIDECategory.TAMPERING,
                    description=(
                        f"Publicly accessible endpoint accepting {method} requests may be targeted with unexpected or "
                        "malformed payload structures if strict server-side validation is not applied."
                    ),
                    affected_asset_ids=[asset.id],
                    status=ObservationStatus.INFERRED,
                    confidence=round(min(0.75, max(0.1, asset.confidence * 0.75)), 2),
                    evidence=evidence_desc,
                    prerequisites=[f"Public endpoint {asset.name} is reachable"],
                    framework_mappings=FrameworkMappings(),
                    metadata={"asset_type": asset.type.value, "http_method": method}
                )
            )

        return threats

    def _deduplicate_and_consolidate(self, threats: List[Threat]) -> List[Threat]:
        """
        Consolidates threats with identical core titles and STRIDE categories for the same asset type,
        combining affected asset IDs and evidence trails without losing data.
        """
        merged_map: Dict[str, Threat] = {}

        for threat in threats:
            # Group key based on category, asset_type, and title structure to merge equivalent threats safely
            asset_type_key = threat.metadata.get("asset_type", "general")
            group_key = f"{threat.category.value}::{asset_type_key}::{threat.title}"

            if group_key not in merged_map:
                merged_map[group_key] = threat
            else:
                existing = merged_map[group_key]
                # Merge affected assets
                combined_assets = list(dict.fromkeys(existing.affected_asset_ids + threat.affected_asset_ids))
                # Merge evidence
                combined_evidence = list(dict.fromkeys(existing.evidence + threat.evidence))
                # Merge prerequisites
                combined_prereqs = list(dict.fromkeys(existing.prerequisites + threat.prerequisites))
                # Keep highest confidence
                highest_conf = max(existing.confidence, threat.confidence)

                merged_map[group_key] = Threat(
                    id=existing.id,
                    title=existing.title,
                    category=existing.category,
                    description=existing.description,
                    affected_asset_ids=combined_assets,
                    status=ObservationStatus.INFERRED,
                    confidence=highest_conf,
                    evidence=combined_evidence,
                    prerequisites=combined_prereqs,
                    framework_mappings=existing.framework_mappings,
                    rag_sources=existing.rag_sources,
                    metadata={**existing.metadata, **threat.metadata}
                )

        return list(merged_map.values())


# Module-level convenience function
def model_threats(assets: List[Asset], discovery: Optional[DiscoveryResult] = None) -> List[Threat]:
    """
    Public API function to perform deterministic STRIDE threat modeling on identified assets.
    
    Example:
        threats = model_threats(assets)
    """
    modeler = ThreatModeler()
    return modeler.model_threats(assets, discovery)
