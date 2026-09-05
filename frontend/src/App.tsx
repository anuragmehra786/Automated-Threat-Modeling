import { useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
  MarkerType,
  Position,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  Shield,
  Globe,
  Search,
  AlertTriangle,
  Server,
  Bug,
  Target,
  ChevronRight,
  Download,
  FileText,
  FileJson,
  FileSpreadsheet,
  CheckCircle2,
  Eye,
  HelpCircle,
  Zap,
  BarChart3,
  Network,
  ListChecks,
} from 'lucide-react';
import './App.css';
import type { SecurityAnalysis, Severity } from './types/schemas';
import { mockAnalysis } from './data/mockData';

/* ─── Helpers ─── */

function severityClass(s: Severity) {
  return `badge badge-${s.toLowerCase()}` as string;
}

function statusClass(s: string) {
  return `badge badge-${s.toLowerCase()}` as string;
}

function getAssetNameById(analysis: SecurityAnalysis, id: string) {
  return analysis.assets.find((a) => a.id === id)?.name ?? id;
}

function getThreatNameById(analysis: SecurityAnalysis, id: string) {
  return analysis.threats.find((t) => t.id === id)?.threat_name ?? id;
}

/* ─── Attack-path Node Component ─── */

function AttackNodeComponent({ data }: { data: { label: string; nodeType: string } }) {
  return (
    <div className={`attack-node ${data.nodeType}`}>
      {data.label}
      <div className="attack-node-label">{data.nodeType.replace('_', ' ')}</div>
    </div>
  );
}

const nodeTypes = { attackNode: AttackNodeComponent };

/* ─── Tabs ─── */

type TabId = 'assets' | 'threats' | 'attack-paths' | 'recommendations';

interface TabDef {
  id: TabId;
  label: string;
  icon: React.ReactNode;
  count?: number;
}

/* ─── Main App ─── */

function App() {
  const [url, setUrl] = useState('');
  const [priorities, setPriorities] = useState<string[]>(['authentication', 'data_exposure', 'web_application']);
  const [activeTab, setActiveTab] = useState<TabId>('assets');
  const [analysisState, setAnalysisState] = useState<'idle' | 'loading' | 'done'>('idle');
  const [analysis, setAnalysis] = useState<SecurityAnalysis | null>(null);

  /* Toggle priority chip */
  const togglePriority = (p: string) => {
    setPriorities((prev) => (prev.includes(p) ? prev.filter((x) => x !== p) : [...prev, p]));
  };

  /* Simulate analysis */
  const handleAnalyze = useCallback(() => {
    if (!url.trim()) return;
    setAnalysisState('loading');
    setTimeout(() => {
      setAnalysis(mockAnalysis);
      setAnalysisState('done');
    }, 2200);
  }, [url]);

  /* Build React Flow nodes/edges from attack path data */
  const { flowNodes, flowEdges } = useMemo(() => {
    if (!analysis) return { flowNodes: [] as Node[], flowEdges: [] as Edge[] };

    const ap = analysis.attack_paths;
    const ySpacing = 100;
    const xCenter = 300;

    const flowNodes: Node[] = ap.nodes.map((n, i) => ({
      id: n.id,
      type: 'attackNode',
      position: { x: xCenter + (i % 2 === 0 ? 0 : 80), y: i * ySpacing },
      data: { label: n.label, nodeType: n.type },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    }));

    const flowEdges: Edge[] = ap.edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      label: e.label,
      animated: true,
      style: { stroke: '#6366f1', strokeWidth: 2 },
      labelStyle: { fill: '#94a3b8', fontSize: 11, fontFamily: 'Inter' },
      labelBgStyle: { fill: '#1a2035', fillOpacity: 0.9 },
      labelBgPadding: [6, 3] as [number, number],
      labelBgBorderRadius: 4,
      markerEnd: { type: MarkerType.ArrowClosed, color: '#6366f1' },
    }));

    return { flowNodes, flowEdges };
  }, [analysis]);

  /* Export */
  const handleExport = (format: string) => {
    if (!analysis) return;
    let content: string;
    let mime: string;
    let ext: string;

    if (format === 'json') {
      content = JSON.stringify(analysis, null, 2);
      mime = 'application/json';
      ext = 'json';
    } else if (format === 'csv') {
      const rows = [['Threat', 'Severity', 'Confidence', 'Affected Asset', 'Tags']];
      analysis.threats.forEach((t) => {
        rows.push([t.threat_name, t.severity, String(t.confidence), getAssetNameById(analysis, t.affected_asset_id), t.tags.join('; ')]);
      });
      content = rows.map((r) => r.map((c) => `"${c}"`).join(',')).join('\n');
      mime = 'text/csv';
      ext = 'csv';
    } else {
      /* Simple text report for PDF placeholder */
      const lines: string[] = [
        'AUTOMATED THREAT MODELING REPORT',
        '================================',
        '',
        `Overall Risk: ${analysis.summary.overall_risk}`,
        `Assets Found: ${analysis.summary.assets_count}`,
        `Threats Found: ${analysis.summary.threats_count}`,
        '',
        'THREATS',
        '-------',
      ];
      analysis.threats.forEach((t, i) => {
        lines.push(`${i + 1}. ${t.threat_name} [${t.severity}]`);
        lines.push(`   Confidence: ${(t.confidence * 100).toFixed(0)}%`);
        lines.push(`   ${t.recommended_mitigation}`);
        lines.push('');
      });
      content = lines.join('\n');
      mime = 'text/plain';
      ext = 'txt';
    }

    const blob = new Blob([content], { type: mime });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `threat-analysis.${ext}`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  /* Tab definitions */
  const tabs: TabDef[] = analysis
    ? [
        { id: 'assets', label: 'Assets', icon: <Server size={15} />, count: analysis.assets.length },
        { id: 'threats', label: 'Threats', icon: <Bug size={15} />, count: analysis.threats.length },
        { id: 'attack-paths', label: 'Attack Paths', icon: <Network size={15} /> },
        { id: 'recommendations', label: 'Recommendations', icon: <ListChecks size={15} />, count: analysis.recommendations.length },
      ]
    : [];

  /* Severity bar max for scaling */
  const severityMax = analysis
    ? Math.max(analysis.summary.critical_threats, analysis.summary.high_threats, analysis.summary.medium_threats, analysis.summary.low_threats, 1)
    : 1;

  return (
    <div className="app-container">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-brand">
          <div className="header-logo">
            <Shield size={22} />
          </div>
          <div>
            <div className="header-title">Automated Threat Modeling</div>
            <div className="header-subtitle">Attack Path Mapper & Security Intelligence</div>
          </div>
        </div>
        {analysisState === 'done' && (
          <div className="header-status">
            <span className="header-status-dot" />
            Analysis Complete
          </div>
        )}
      </header>

      {/* ── URL Submission ── */}
      <section className="submit-section">
        <div className="submit-card glass-card">
          <h2>
            <Target size={18} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 8 }} />
            Target Analysis
          </h2>
          <p>Enter an authorized URL to discover assets, model threats, and map potential attack paths.</p>

          <div className="submit-form">
            <div className="url-input-wrapper">
              <Globe size={18} className="url-input-icon" />
              <input
                id="url-input"
                className="url-input"
                type="url"
                placeholder="https://example.com"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
              />
            </div>
            <button
              id="analyze-btn"
              className={`analyze-btn ${analysisState === 'loading' ? 'loading' : ''}`}
              onClick={handleAnalyze}
              disabled={analysisState === 'loading'}
            >
              {analysisState === 'loading' ? (
                <>
                  <span className="spinner" />
                  Analyzing…
                </>
              ) : (
                <>
                  <Search size={16} />
                  Analyze
                </>
              )}
            </button>
          </div>

          <div className="priorities-row">
            {['authentication', 'data_exposure', 'web_application', 'session_management', 'access_control'].map((p) => (
              <button
                key={p}
                className={`priority-chip ${priorities.includes(p) ? 'active' : ''}`}
                onClick={() => togglePriority(p)}
              >
                <CheckCircle2 size={13} />
                {p.replace('_', ' ')}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── Results ── */}
      {analysis && analysisState === 'done' && (
        <>
          {/* Summary Grid */}
          <div className="summary-grid">
            {/* Overall Risk */}
            <div className="summary-card glass-card">
              <div className="summary-card-header">
                <span className="summary-card-label">Overall Risk</span>
                <div className="summary-card-icon" style={{ background: 'rgba(239, 68, 68, 0.12)' }}>
                  <AlertTriangle size={18} color="var(--severity-high)" />
                </div>
              </div>
              <span className={`risk-badge-large badge-${analysis.summary.overall_risk.toLowerCase()}`}>
                {analysis.summary.overall_risk}
              </span>
            </div>

            {/* Assets */}
            <div className="summary-card glass-card">
              <div className="summary-card-header">
                <span className="summary-card-label">Assets</span>
                <div className="summary-card-icon" style={{ background: 'rgba(59, 130, 246, 0.12)' }}>
                  <Server size={18} color="#3b82f6" />
                </div>
              </div>
              <div className="summary-card-value">{analysis.summary.assets_count}</div>
              <div className="summary-card-sub">Security-relevant assets identified</div>
            </div>

            {/* Threats */}
            <div className="summary-card glass-card">
              <div className="summary-card-header">
                <span className="summary-card-label">Threats</span>
                <div className="summary-card-icon" style={{ background: 'rgba(249, 115, 22, 0.12)' }}>
                  <Bug size={18} color="#f97316" />
                </div>
              </div>
              <div className="summary-card-value">{analysis.summary.threats_count}</div>
              <div className="summary-card-sub">Potential threats modeled</div>
            </div>

            {/* Severity Breakdown */}
            <div className="summary-card glass-card">
              <div className="summary-card-header">
                <span className="summary-card-label">Severity</span>
                <div className="summary-card-icon" style={{ background: 'rgba(99, 102, 241, 0.12)' }}>
                  <BarChart3 size={18} color="var(--accent-primary)" />
                </div>
              </div>
              <div className="severity-chart">
                {[
                  { label: 'CRIT', value: analysis.summary.critical_threats, color: 'var(--severity-critical)' },
                  { label: 'HIGH', value: analysis.summary.high_threats, color: 'var(--severity-high)' },
                  { label: 'MED', value: analysis.summary.medium_threats, color: 'var(--severity-medium)' },
                  { label: 'LOW', value: analysis.summary.low_threats, color: 'var(--severity-low)' },
                ].map((s) => (
                  <div key={s.label} className="severity-bar-group">
                    <span className="severity-bar-value" style={{ color: s.color }}>{s.value}</span>
                    <div
                      className="severity-bar"
                      style={{
                        height: `${Math.max((s.value / severityMax) * 40, 4)}px`,
                        background: s.color,
                      }}
                    />
                    <span className="severity-bar-label">{s.label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Attack Surface */}
            <div className="summary-card glass-card">
              <div className="summary-card-header">
                <span className="summary-card-label">Attack Surface</span>
                <div className="summary-card-icon" style={{ background: 'rgba(139, 92, 246, 0.12)' }}>
                  <Zap size={18} color="#8b5cf6" />
                </div>
              </div>
              <div className="summary-card-value">{analysis.attack_paths.nodes.length}</div>
              <div className="summary-card-sub">Nodes in attack graph</div>
            </div>

            {/* Recommendations */}
            <div className="summary-card glass-card">
              <div className="summary-card-header">
                <span className="summary-card-label">Actions</span>
                <div className="summary-card-icon" style={{ background: 'rgba(34, 197, 94, 0.12)' }}>
                  <ListChecks size={18} color="#22c55e" />
                </div>
              </div>
              <div className="summary-card-value">{analysis.recommendations.length}</div>
              <div className="summary-card-sub">Recommended mitigations</div>
            </div>
          </div>

          {/* Tabs */}
          <nav className="tab-nav">
            {tabs.map((t) => (
              <button
                key={t.id}
                id={`tab-${t.id}`}
                className={`tab-btn ${activeTab === t.id ? 'active' : ''}`}
                onClick={() => setActiveTab(t.id)}
              >
                {t.icon}
                {t.label}
                {t.count !== undefined && <span className="tab-count">{t.count}</span>}
              </button>
            ))}
          </nav>

          {/* ── Assets Tab ── */}
          <AnimatePresence mode="wait">
          {activeTab === 'assets' && (
            <motion.div
              key="assets"
              className="asset-grid"
              initial="hidden"
              animate="visible"
              exit={{ opacity: 0 }}
              variants={{ visible: { transition: { staggerChildren: 0.07 } } }}
            >
              {analysis.assets.map((asset) => (
                <motion.div
                  key={asset.id}
                  className="glass-card asset-card"
                  variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0 } }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                  whileHover={{ scale: 1.01, boxShadow: 'var(--accent-glow)' }}
                >
                  <div className="asset-card-header">
                    <div>
                      <div className="asset-card-name">{asset.name}</div>
                      <div className="asset-card-type">{asset.type.replace(/_/g, ' ')}</div>
                    </div>
                    <span className={statusClass(asset.status)}>
                      {asset.status === 'OBSERVED' && <Eye size={11} />}
                      {asset.status === 'INFERRED' && <HelpCircle size={11} />}
                      {asset.status}
                    </span>
                  </div>

                  <div className="confidence-bar-wrapper">
                    <div className="confidence-label">
                      <span>Confidence</span>
                      <span>{(asset.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="confidence-bar">
                      <motion.div
                        className="confidence-bar-fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${asset.confidence * 100}%` }}
                        transition={{ duration: 0.6, ease: 'easeOut', delay: 0.2 }}
                      />
                    </div>
                  </div>

                  <ul className="evidence-list">
                    {asset.evidence.map((ev, j) => (
                      <li key={j} className="evidence-item">
                        <ChevronRight size={12} className="evidence-icon" />
                        {ev}
                      </li>
                    ))}
                  </ul>
                </motion.div>
              ))}
            </motion.div>
          )}
          </AnimatePresence>

          {/* ── Threats Tab ── */}
          <AnimatePresence mode="wait">
          {activeTab === 'threats' && (
            <motion.div
              key="threats"
              className="threat-list"
              initial="hidden"
              animate="visible"
              exit={{ opacity: 0 }}
              variants={{ visible: { transition: { staggerChildren: 0.06 } } }}
            >
              {analysis.threats.map((threat) => (
                <motion.div
                  key={threat.id}
                  className="glass-card threat-card"
                  data-severity={threat.severity}
                  variants={{ hidden: { opacity: 0, x: -16 }, visible: { opacity: 1, x: 0 } }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                  whileHover={{ scale: 1.005 }}
                >
                  <div className="threat-card-header">
                    <span className="threat-card-name">{threat.threat_name}</span>
                    <span className={severityClass(threat.severity)}>{threat.severity}</span>
                  </div>

                  <div className="threat-meta">
                    <span>
                      <Server size={12} />
                      {getAssetNameById(analysis, threat.affected_asset_id)}
                    </span>
                    <span>
                      <Target size={12} />
                      Confidence: {(threat.confidence * 100).toFixed(0)}%
                    </span>
                  </div>

                  <div className="threat-tags">
                    {threat.tags.map((tag) => (
                      <span key={tag} className="threat-tag">{tag}</span>
                    ))}
                  </div>

                  <div className="threat-evidence-section">
                    <div className="threat-evidence-title">Evidence</div>
                    <ul className="evidence-list">
                      {threat.evidence.map((ev, j) => (
                        <li key={j} className="evidence-item">
                          <ChevronRight size={12} className="evidence-icon" />
                          {ev}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="threat-mitigation">
                    <strong>Recommended Mitigation</strong>
                    {threat.recommended_mitigation}
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
          </AnimatePresence>

          {/* ── Attack Paths Tab ── */}
          <AnimatePresence mode="wait">
          {activeTab === 'attack-paths' && (
            <motion.div
              key="attack-paths"
              className="attack-path-container"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <ReactFlow
                nodes={flowNodes}
                edges={flowEdges}
                nodeTypes={nodeTypes}
                fitView
                proOptions={{ hideAttribution: true }}
                style={{ background: 'var(--bg-secondary)' }}
              >
                <Background color="#1e293b" gap={24} size={1} />
                <Controls
                  style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)', borderRadius: 8 }}
                />
                <MiniMap
                  nodeColor={() => '#6366f1'}
                  maskColor="rgba(10, 14, 26, 0.8)"
                  style={{ background: 'var(--bg-card)', borderRadius: 8, border: '1px solid var(--border-subtle)' }}
                />
              </ReactFlow>
            </motion.div>
          )}
          </AnimatePresence>

          {/* ── Recommendations Tab ── */}
          <AnimatePresence mode="wait">
          {activeTab === 'recommendations' && (
            <motion.div
              key="recommendations"
              className="recommendation-list"
              initial="hidden"
              animate="visible"
              exit={{ opacity: 0 }}
              variants={{ visible: { transition: { staggerChildren: 0.08 } } }}
            >
              {analysis.recommendations.map((rec, i) => (
                <motion.div
                  key={rec.id}
                  className="glass-card recommendation-card"
                  variants={{ hidden: { opacity: 0, y: 12 }, visible: { opacity: 1, y: 0 } }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                  whileHover={{ scale: 1.005 }}
                >
                  <div className="recommendation-number">{i + 1}</div>
                  <div className="recommendation-content">
                    <div className="recommendation-text">{rec.recommendation}</div>
                    <div className="recommendation-meta">
                      <span className={severityClass(rec.priority)}>Priority: {rec.priority}</span>
                      <span>Threat: {getThreatNameById(analysis, rec.related_threat_id)}</span>
                      <span>Asset: {getAssetNameById(analysis, rec.related_asset_id)}</span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
          </AnimatePresence>


          {/* ── Reports ── */}
          <div className="report-section glass-card">
            <h3>
              <Download size={18} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 8 }} />
              Export Report
            </h3>
            <div className="report-buttons">
              <button id="export-txt" className="report-btn" onClick={() => handleExport('txt')}>
                <FileText size={16} />
                Download TXT Report
              </button>
              <button id="export-json" className="report-btn" onClick={() => handleExport('json')}>
                <FileJson size={16} />
                Download JSON
              </button>
              <button id="export-csv" className="report-btn" onClick={() => handleExport('csv')}>
                <FileSpreadsheet size={16} />
                Download CSV
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
