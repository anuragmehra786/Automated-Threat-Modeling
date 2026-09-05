"""
Security Asset Identification and Reasoning Layer.

Transforms structured discovery evidence (DiscoveryResult) into security-relevant
application assets (Asset) using deterministic heuristics.

Key Principles:
1. Strict Epistemic Status:
   - OBSERVED: Directly visible in discovery evidence (e.g., specific endpoints, detected technologies).
   - INFERRED: Reasoned functional capability derived from evidence (e.g., authentication service, admin interface).
   - UNKNOWN: Kept separate; never assume unobserved internal architecture without evidence.
2. Evidence Preservation: Every asset retains the exact discovery indicators that justified its identification.
3. Conservative Confidence: Confidence is assigned between 0.0 and 1.0 based on evidentiary strength.
4. No Speculation: Does NOT invent internal components (e.g., database, Redis, private microservices) unless evidence exists.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set
from app.models.schemas import (
    Asset,
    AssetType,
    CookieInfo,
    DiscoveryResult,
    Endpoint,
    Form,
    ObservationStatus,
    Page,
    SecurityIndicators,
    SeverityLevel,
    Technology,
)


def _sanitize_id(text: str) -> str:
    """Helper to convert names/paths into clean URL-safe identifier slugs."""
    clean = re.sub(r"[^a-zA-Z0-9_-]", "-", text.strip().lower())
    clean = re.sub(r"-+", "-", clean).strip("-")
    return clean or "unknown"


class AssetIdentifier:
    """
    Deterministic reasoning engine that analyzes discovery evidence
    and extracts security-relevant application assets.

    Important Design Principles:
    - Epistemic Status: Clearly separates OBSERVED (direct evidence) from INFERRED (deduced capabilities).
    - Asset Criticality: Represents inherent business/architectural sensitivity of the asset itself
      (e.g., authentication and payment interfaces handle high-value data), NOT vulnerability presence,
      threat severity, likelihood, or risk score.
    - Public Endpoint Confidence: Set to 0.95 for observed routes to reflect high empirical discovery
      certainty while maintaining a realistic margin for dynamic routing/URL rewrites.
    - Conservative Inference: Requires concrete evidence and avoids speculative backend components.
    """

    def __init__(self) -> None:
        # Route indicators for functional inference
        self.auth_keywords = {"login", "signin", "sign-in", "auth", "authenticate", "token", "oauth"}
        self.explicit_admin_keywords = {"admin", "administrator", "wp-admin", "cpanel", "controlpanel", "console"}
        self.user_data_keywords = {"user", "users", "profile", "account", "settings", "me", "preferences", "register", "signup", "sign-up"}
        self.payment_keywords = {"checkout", "payment", "pay", "billing", "stripe", "cart", "order"}
        self.session_cookie_names = {"session", "sessionid", "session_id", "jwt", "token", "auth_token", "connect.sid", "phpsessid", "jsessionid", "aspsessionid", "csrftoken", "xsrf-token"}

    def identify_assets(self, discovery: DiscoveryResult) -> List[Asset]:
        """
        Main entry point for asset reasoning.
        
        Analyzes technologies, pages, endpoints, forms, and security indicators
        from DiscoveryResult and produces a consolidated list of Asset models.
        """
        if not discovery:
            return []

        raw_assets: List[Asset] = []

        # 1. Directly observed technology components
        raw_assets.extend(self._identify_technology_components(discovery.technologies))

        # 2. Directly observed public endpoints
        raw_assets.extend(self._identify_public_endpoints(discovery.endpoints, discovery.pages))

        # 3. Inferred authentication services
        auth_asset = self._infer_authentication_service(discovery)
        if auth_asset:
            raw_assets.append(auth_asset)

        # 4. Inferred administrative interfaces
        admin_asset = self._infer_admin_interface(discovery)
        if admin_asset:
            raw_assets.append(admin_asset)

        # 5. Inferred user data interfaces
        user_data_asset = self._infer_user_data_interface(discovery)
        if user_data_asset:
            raw_assets.append(user_data_asset)

        # 6. Inferred session management mechanisms
        session_asset = self._infer_session_management(discovery.security_indicators)
        if session_asset:
            raw_assets.append(session_asset)

        # 7. Inferred payment/billing interfaces
        payment_asset = self._infer_payment_interface(discovery)
        if payment_asset:
            raw_assets.append(payment_asset)

        # Deduplicate and consolidate evidence across identified assets
        return self._deduplicate_and_consolidate(raw_assets)

    def _identify_technology_components(self, technologies: List[Technology]) -> List[Asset]:
        """
        Creates OBSERVED technology component assets from detected stack technologies.
        Criticality is set to LOW as it represents general infrastructure/library presence.
        """
        assets: List[Asset] = []
        for tech in technologies:
            version_str = f" v{tech.version}" if tech.version else ""
            category_str = f" ({', '.join(tech.categories)})" if tech.categories else ""
            evidence_desc = f"Observed technology '{tech.name}{version_str}'"
            if tech.evidence:
                evidence_desc += f": {tech.evidence}"

            asset = Asset(
                id=f"asset-tech-{_sanitize_id(tech.name)}",
                name=f"{tech.name} Component",
                type=AssetType.TECHNOLOGY_COMPONENT,
                description=f"Directly detected software technology: {tech.name}{version_str}{category_str}.",
                status=ObservationStatus.OBSERVED,
                confidence=max(0.1, min(1.0, tech.confidence if tech.confidence is not None else 1.0)),
                evidence=[evidence_desc],
                associated_endpoints=[],
                criticality=SeverityLevel.LOW,
                metadata={"version": tech.version, "categories": tech.categories}
            )
            assets.append(asset)
        return assets

    def _identify_public_endpoints(self, endpoints: List[Endpoint], pages: List[Page]) -> List[Asset]:
        """
        Creates OBSERVED public endpoint assets for discovered entry routes.
        Confidence is 0.95 reflecting high empirical certainty of discovery observations.
        Criticality is LOW representing standard entry routes (not vulnerability severity).
        """
        assets: List[Asset] = []
        seen_paths: Set[str] = set()

        for ep in endpoints:
            if ep.path in seen_paths:
                continue
            seen_paths.add(ep.path)

            evidence_desc = f"Directly observed {ep.method} {ep.path} in discovery endpoints"
            if ep.parameters:
                param_names = [p.name for p in ep.parameters]
                evidence_desc += f" with parameters: {', '.join(param_names)}"

            asset = Asset(
                id=f"asset-endpoint-{_sanitize_id(ep.method + '-' + ep.path)}",
                name=f"Public Endpoint: {ep.method} {ep.path}",
                type=AssetType.PUBLIC_ENDPOINT,
                description=f"Publicly accessible {ep.method} endpoint located at {ep.path}.",
                status=ObservationStatus.OBSERVED,
                confidence=0.95,
                evidence=[evidence_desc],
                associated_endpoints=[ep.path],
                criticality=SeverityLevel.LOW,
                metadata={"method": ep.method, "parameters_count": len(ep.parameters)}
            )
            assets.append(asset)

        # Check pages that may not have been listed under endpoints
        for page in pages:
            path = page.path or page.url
            if path not in seen_paths and page.status_code == 200:
                seen_paths.add(path)
                asset = Asset(
                    id=f"asset-page-{_sanitize_id(path)}",
                    name=f"Public Page: {page.title or path}",
                    type=AssetType.PUBLIC_ENDPOINT,
                    description=f"Discovered webpage accessible at {page.url} (Status: {page.status_code}).",
                    status=ObservationStatus.OBSERVED,
                    confidence=0.95,
                    evidence=[f"Directly observed webpage '{page.url}' with status code {page.status_code}"],
                    associated_endpoints=[path],
                    criticality=SeverityLevel.LOW,
                    metadata={"title": page.title, "links_count": len(page.links)}
                )
                assets.append(asset)

        return assets

    def _infer_authentication_service(self, discovery: DiscoveryResult) -> Optional[Asset]:
        """
        Infers the presence of an Authentication Service if login forms, password inputs,
        or authentication routes/pages are observed.
        
        Criticality is HIGH because authentication is a sensitive identity gateway asset.
        """
        evidence: List[str] = []
        associated_endpoints: List[str] = []
        has_password_field = False
        has_login_route = False

        # 1. Check forms for password fields or login types
        for form in discovery.forms:
            form_evidence = []
            for field in form.fields:
                if field.field_type.lower() == "password" or "pass" in field.name.lower():
                    has_password_field = True
                    form_evidence.append(f"password field '{field.name}'")
                elif "user" in field.name.lower() or "email" in field.name.lower():
                    form_evidence.append(f"username/email field '{field.name}'")

            fields_desc = f" containing: {', '.join(form_evidence)}" if form_evidence else ""
            if form.form_type and form.form_type.lower() in ("login", "auth", "signin"):
                has_login_route = True
                evidence.append(f"Discovered {form.form_type} form with action '{form.action}'{fields_desc}")
            elif form_evidence:
                evidence.append(f"Discovered form targeting '{form.action}'{fields_desc}")

            if form.action:
                associated_endpoints.append(form.action)

        # 2. Check endpoints for authentication keywords
        for ep in discovery.endpoints:
            path_lower = ep.path.lower()
            if any(kw in path_lower for kw in self.auth_keywords):
                has_login_route = True
                evidence.append(f"Observed authentication-related endpoint: {ep.method} {ep.path}")
                associated_endpoints.append(ep.path)

        # 3. Check pages for login keywords
        for page in discovery.pages:
            url_lower = page.url.lower()
            title_lower = (page.title or "").lower()
            if any(kw in url_lower or kw in title_lower for kw in self.auth_keywords):
                has_login_route = True
                evidence.append(f"Discovered authentication-related page: '{page.url}' (Title: '{page.title}')")
                if page.path:
                    associated_endpoints.append(page.path)

        if not evidence:
            return None

        # Conservative confidence assignment based on evidentiary strength
        if has_password_field and has_login_route:
            confidence = 0.95
        elif has_password_field or has_login_route:
            confidence = 0.85
        else:
            confidence = 0.70

        return Asset(
            id="asset-auth-service",
            name="User Authentication Service",
            type=AssetType.AUTHENTICATION_SERVICE,
            description="Inferred service responsible for verifying user credentials, handling login requests, and issuing identity sessions.",
            status=ObservationStatus.INFERRED,
            confidence=confidence,
            evidence=list(dict.fromkeys(evidence)),  # preserve order, remove duplicates
            associated_endpoints=list(dict.fromkeys(associated_endpoints)),
            criticality=SeverityLevel.HIGH,
            metadata={"has_password_field": has_password_field, "has_login_route": has_login_route}
        )

    def _infer_admin_interface(self, discovery: DiscoveryResult) -> Optional[Asset]:
        """
        Infers the presence of an Administrative Interface ONLY when supported by explicit
        administrative paths (e.g. /admin, /wp-admin, /console) or explicit admin page titles.
        
        Ambiguous paths (like generic /dashboard or /manage) do NOT trigger an admin interface
        unless explicit administrative markers exist.
        
        Criticality is HIGH reflecting privileged management sensitivity.
        """
        evidence: List[str] = []
        associated_endpoints: List[str] = []
        has_explicit_admin_marker = False

        # 1. Check endpoints for explicit admin paths
        for ep in discovery.endpoints:
            path_lower = ep.path.lower()
            if any(kw in path_lower for kw in self.explicit_admin_keywords):
                has_explicit_admin_marker = True
                evidence.append(f"Observed explicit administrative endpoint: {ep.method} {ep.path}")
                associated_endpoints.append(ep.path)

        # 2. Check pages for explicit admin paths or explicit titles
        for page in discovery.pages:
            url_lower = page.url.lower()
            title_lower = (page.title or "").lower()
            
            is_explicit_url = any(kw in url_lower for kw in self.explicit_admin_keywords)
            is_explicit_title = any(kw in title_lower for kw in ("admin", "administrator", "control panel", "management console"))
            
            if is_explicit_url or is_explicit_title:
                has_explicit_admin_marker = True
                evidence.append(f"Discovered administrative page: '{page.url}' (Title: '{page.title}')")
                if page.path:
                    associated_endpoints.append(page.path)

        if not has_explicit_admin_marker:
            return None

        return Asset(
            id="asset-admin-interface",
            name="Administrative Management Interface",
            type=AssetType.ADMIN_INTERFACE,
            description="Inferred privileged interface providing administrative control, configuration management, or system monitoring functions.",
            status=ObservationStatus.INFERRED,
            confidence=0.90,
            evidence=list(dict.fromkeys(evidence)),
            associated_endpoints=list(dict.fromkeys(associated_endpoints)),
            criticality=SeverityLevel.HIGH,
            metadata={"explicit_admin_evidence": True}
        )

    def _infer_user_data_interface(self, discovery: DiscoveryResult) -> Optional[Asset]:
        """
        Infers User Account & Profile Data Interfaces if user management routes,
        registration forms, or profile endpoints are observed.
        
        Criticality is MEDIUM reflecting user personal data handling.
        """
        evidence: List[str] = []
        associated_endpoints: List[str] = []

        # 1. Check forms (e.g. registration, profile update)
        for form in discovery.forms:
            if form.form_type and form.form_type.lower() in ("register", "signup", "profile"):
                evidence.append(f"Discovered user data form '{form.form_type}' with action '{form.action}'")
                if form.action:
                    associated_endpoints.append(form.action)

        # 2. Check endpoints
        for ep in discovery.endpoints:
            path_lower = ep.path.lower()
            # Avoid matching login endpoints already categorized under auth
            if any(kw in path_lower for kw in self.user_data_keywords) and not any(kw in path_lower for kw in ("login", "signin")):
                evidence.append(f"Observed user data/profile endpoint: {ep.method} {ep.path}")
                associated_endpoints.append(ep.path)

        # 3. Check pages
        for page in discovery.pages:
            url_lower = page.url.lower()
            if any(kw in url_lower for kw in self.user_data_keywords) and not any(kw in url_lower for kw in ("login", "signin")):
                evidence.append(f"Discovered user account/profile page: '{page.url}'")
                if page.path:
                    associated_endpoints.append(page.path)

        if not evidence:
            return None

        return Asset(
            id="asset-user-data-interface",
            name="User Profile & Account Data Interface",
            type=AssetType.USER_DATA_INTERFACE,
            description="Inferred application interface handling user registration, personal account information, and profile settings.",
            status=ObservationStatus.INFERRED,
            confidence=0.85,
            evidence=list(dict.fromkeys(evidence)),
            associated_endpoints=list(dict.fromkeys(associated_endpoints)),
            criticality=SeverityLevel.MEDIUM,
            metadata={}
        )

    def _infer_session_management(self, indicators: SecurityIndicators) -> Optional[Asset]:
        """
        Infers the presence of Session Management mechanisms from observed cookies.
        
        Note: Indicates that state management is present based on cookies.
        Backend architecture, storage mechanisms, and security posture remain UNKNOWN.
        
        Criticality is MEDIUM representing state-tracking sensitivity.
        """
        if not indicators or not indicators.cookies:
            return None

        evidence: List[str] = []
        session_cookies_found: List[CookieInfo] = []

        for cookie in indicators.cookies:
            cookie_name_lower = cookie.name.lower()
            is_session = any(sc in cookie_name_lower for sc in self.session_cookie_names)
            if is_session:
                session_cookies_found.append(cookie)
                flag_details = f"Secure={cookie.secure}, HttpOnly={cookie.httponly}, SameSite={cookie.samesite}"
                evidence.append(
                    f"Observed session-related cookie '{cookie.name}' with flags ({flag_details}). "
                    f"Indicates state management is present; backend implementation and security posture are unknown."
                )

        if not evidence:
            return None

        return Asset(
            id="asset-session-management",
            name="Session Management Mechanism",
            type=AssetType.SESSION_MANAGEMENT,
            description=(
                "Inferred session and state management mechanism based on observed cookies. "
                "(Backend storage architecture, implementation details, and security posture are unknown)."
            ),
            status=ObservationStatus.INFERRED,
            confidence=0.90,
            evidence=evidence,
            associated_endpoints=[],
            criticality=SeverityLevel.MEDIUM,
            metadata={"session_cookies": [c.name for c in session_cookies_found]}
        )

    def _infer_payment_interface(self, discovery: DiscoveryResult) -> Optional[Asset]:
        """
        Infers Payment & Billing Interfaces if checkout/billing endpoints or payment forms are observed.
        
        Criticality is HIGH representing financial transaction data sensitivity.
        (Does NOT imply a vulnerability exists).
        """
        evidence: List[str] = []
        associated_endpoints: List[str] = []

        # 1. Check endpoints
        for ep in discovery.endpoints:
            path_lower = ep.path.lower()
            if any(kw in path_lower for kw in self.payment_keywords):
                evidence.append(f"Observed payment/billing endpoint: {ep.method} {ep.path}")
                associated_endpoints.append(ep.path)

        # 2. Check forms
        for form in discovery.forms:
            action_lower = (form.action or "").lower()
            if any(kw in action_lower for kw in self.payment_keywords) or (form.form_type and any(kw in form.form_type.lower() for kw in self.payment_keywords)):
                evidence.append(f"Discovered payment/checkout form targeting '{form.action}'")
                if form.action:
                    associated_endpoints.append(form.action)

        if not evidence:
            return None

        return Asset(
            id="asset-payment-interface",
            name="Payment & Billing Interface",
            type=AssetType.PAYMENT_INTERFACE,
            description="Inferred financial transaction or payment processing interface handling checkout and billing workflows.",
            status=ObservationStatus.INFERRED,
            confidence=0.85,
            evidence=list(dict.fromkeys(evidence)),
            associated_endpoints=list(dict.fromkeys(associated_endpoints)),
            criticality=SeverityLevel.HIGH,
            metadata={}
        )

    def _deduplicate_and_consolidate(self, assets: List[Asset]) -> List[Asset]:
        """
        Consolidates assets sharing identical identifiers or core identities,
        merging their evidence and endpoint lists without losing data.
        """
        merged_map: Dict[str, Asset] = {}

        for asset in assets:
            if asset.id not in merged_map:
                merged_map[asset.id] = asset
            else:
                existing = merged_map[asset.id]
                # Merge evidence
                combined_evidence = list(dict.fromkeys(existing.evidence + asset.evidence))
                # Merge associated endpoints
                combined_endpoints = list(dict.fromkeys(existing.associated_endpoints + asset.associated_endpoints))
                # Keep highest confidence
                highest_conf = max(existing.confidence, asset.confidence)
                # If either is OBSERVED, prioritize OBSERVED over INFERRED
                merged_status = ObservationStatus.OBSERVED if (existing.status == ObservationStatus.OBSERVED or asset.status == ObservationStatus.OBSERVED) else ObservationStatus.INFERRED

                merged_map[asset.id] = Asset(
                    id=existing.id,
                    name=existing.name,
                    type=existing.type,
                    description=existing.description,
                    status=merged_status,
                    confidence=highest_conf,
                    evidence=combined_evidence,
                    associated_endpoints=combined_endpoints,
                    criticality=existing.criticality or asset.criticality,
                    metadata={**existing.metadata, **asset.metadata}
                )

        return list(merged_map.values())


# Module-level convenience function
def identify_assets(discovery: DiscoveryResult) -> List[Asset]:
    """
    Public API function to identify and reason about security assets from a DiscoveryResult.
    
    Example:
        assets = identify_assets(discovery_result)
    """
    identifier = AssetIdentifier()
    return identifier.identify_assets(discovery)
