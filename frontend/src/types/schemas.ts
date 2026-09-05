export type AssetStatus = 'OBSERVED' | 'INFERRED' | 'UNKNOWN';
export type Severity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface Asset {
  id: string;
  name: string;
  type: string;
  status: AssetStatus;
  confidence: number;
  evidence: string[];
}

export interface Threat {
  id: string;
  threat_name: string;
  severity: Severity;
  affected_asset_id: string;
  evidence: string[];
  confidence: number;
  tags: string[];
  recommended_mitigation: string;
}

export interface Recommendation {
  id: string;
  recommendation: string;
  priority: Severity;
  related_threat_id: string;
  related_asset_id: string;
}

export interface AttackPathNode {
  id: string;
  label: string;
  type: string;
}

export interface AttackPathEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}

export interface AttackPathGraph {
  nodes: AttackPathNode[];
  edges: AttackPathEdge[];
}

export interface SecurityAnalysisSummary {
  overall_risk: Severity;
  assets_count: number;
  threats_count: number;
  critical_threats: number;
  high_threats: number;
  medium_threats: number;
  low_threats: number;
}

export interface SecurityAnalysis {
  summary: SecurityAnalysisSummary;
  assets: Asset[];
  threats: Threat[];
  attack_paths: AttackPathGraph;
  recommendations: Recommendation[];
}
