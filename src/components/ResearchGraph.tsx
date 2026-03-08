import { useState, useMemo, useRef, useEffect } from "react";
import { Network, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";

interface GraphProps {
  projects: any[];
  experiments: any[];
  models: any[];
}

interface GraphNode {
  id: string;
  label: string;
  type: "project" | "experiment" | "model" | "paper" | "dataset";
  x: number;
  y: number;
}

interface GraphEdge {
  from: string;
  to: string;
}

export default function ResearchGraph({ projects, experiments, models }: GraphProps) {
  const [zoom, setZoom] = useState(1);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const { nodes, edges } = useMemo(() => {
    const n: GraphNode[] = [];
    const e: GraphEdge[] = [];
    const W = 900, H = 600;
    const cx = W / 2, cy = H / 2;

    // Place projects in inner ring
    projects.forEach((p, i) => {
      const angle = (2 * Math.PI * i) / Math.max(projects.length, 1) - Math.PI / 2;
      const r = 140;
      n.push({ id: `p-${p.id}`, label: p.name, type: "project", x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) });

      // Extract papers from project
      const papers = Array.isArray(p.papers) ? p.papers : [];
      papers.slice(0, 3).forEach((paper: any, j: number) => {
        const pAngle = angle + ((j - 1) * 0.3);
        const pr = 260;
        const paperId = `paper-${p.id}-${j}`;
        n.push({ id: paperId, label: paper.title?.slice(0, 30) || `Paper ${j + 1}`, type: "paper", x: cx + pr * Math.cos(pAngle), y: cy + pr * Math.sin(pAngle) });
        e.push({ from: `p-${p.id}`, to: paperId });
      });
    });

    // Place experiments around projects
    experiments.forEach((exp, i) => {
      const proj = projects.findIndex(p => p.id === exp.project_id);
      const baseAngle = proj >= 0 ? (2 * Math.PI * proj) / Math.max(projects.length, 1) - Math.PI / 2 : (2 * Math.PI * i) / Math.max(experiments.length, 1);
      const offset = (i % 3 - 1) * 0.25;
      const r = 200;
      n.push({ id: `e-${exp.id}`, label: exp.title?.slice(0, 25) || "Experiment", type: "experiment", x: cx + r * Math.cos(baseAngle + offset), y: cy + r * Math.sin(baseAngle + offset) });
      if (exp.project_id) e.push({ from: `p-${exp.project_id}`, to: `e-${exp.id}` });
    });

    // Place models in outer ring
    models.forEach((m, i) => {
      const angle = (2 * Math.PI * i) / Math.max(models.length, 1) + Math.PI / 4;
      const r = 250;
      n.push({ id: `m-${m.id}`, label: m.name?.slice(0, 25) || "Model", type: "model", x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) });
      if (m.project_id) e.push({ from: `p-${m.project_id}`, to: `m-${m.id}` });
    });

    return { nodes: n, edges: e };
  }, [projects, experiments, models]);

  const typeColors: Record<string, { fill: string; stroke: string; text: string }> = {
    project: { fill: "hsl(var(--primary) / 0.15)", stroke: "hsl(var(--primary) / 0.5)", text: "hsl(var(--primary))" },
    experiment: { fill: "hsl(45 93% 47% / 0.15)", stroke: "hsl(45 93% 47% / 0.5)", text: "hsl(45 93% 47%)" },
    model: { fill: "hsl(142 71% 45% / 0.15)", stroke: "hsl(142 71% 45% / 0.5)", text: "hsl(142 71% 45%)" },
    paper: { fill: "hsl(var(--muted) / 0.5)", stroke: "hsl(var(--border))", text: "hsl(var(--muted-foreground))" },
    dataset: { fill: "hsl(280 68% 60% / 0.15)", stroke: "hsl(280 68% 60% / 0.5)", text: "hsl(280 68% 60%)" },
  };

  if (nodes.length === 0) {
    return (
      <div className="text-center py-16">
        <Network className="h-10 w-10 text-muted-foreground/30 mx-auto mb-4" />
        <h3 className="text-sm font-semibold text-foreground mb-1">Research Graph Empty</h3>
        <p className="text-xs text-muted-foreground">Create projects and analyze papers to build your knowledge graph.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
          <Network className="h-4 w-4 text-primary" /> Research Knowledge Graph
        </h3>
        <div className="flex items-center gap-1">
          <button onClick={() => setZoom(z => Math.max(0.5, z - 0.1))} className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground"><ZoomOut className="h-3.5 w-3.5" /></button>
          <span className="text-[10px] text-muted-foreground w-10 text-center">{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom(z => Math.min(2, z + 0.1))} className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground"><ZoomIn className="h-3.5 w-3.5" /></button>
          <button onClick={() => setZoom(1)} className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground"><Maximize2 className="h-3.5 w-3.5" /></button>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-[10px] text-muted-foreground">
        {[
          { type: "project", label: "Project", color: "bg-primary/30" },
          { type: "experiment", label: "Experiment", color: "bg-yellow-500/30" },
          { type: "model", label: "Model", color: "bg-green-500/30" },
          { type: "paper", label: "Paper", color: "bg-muted" },
        ].map(l => (
          <span key={l.type} className="flex items-center gap-1.5">
            <span className={`h-2.5 w-2.5 rounded-full ${l.color}`} />
            {l.label}
          </span>
        ))}
      </div>

      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <svg ref={svgRef} viewBox="0 0 900 600" className="w-full" style={{ height: "500px" }}>
          <g transform={`scale(${zoom})`} style={{ transformOrigin: "center" }}>
            {/* Edges */}
            {edges.map((edge, i) => {
              const from = nodes.find(n => n.id === edge.from);
              const to = nodes.find(n => n.id === edge.to);
              if (!from || !to) return null;
              const isHovered = hoveredNode === edge.from || hoveredNode === edge.to;
              return (
                <line key={i} x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                  stroke={isHovered ? "hsl(var(--primary) / 0.6)" : "hsl(var(--border))"}
                  strokeWidth={isHovered ? 2 : 1} strokeDasharray={isHovered ? "" : "4 2"} />
              );
            })}
            {/* Nodes */}
            {nodes.map(node => {
              const c = typeColors[node.type] || typeColors.paper;
              const isHovered = hoveredNode === node.id;
              const r = node.type === "project" ? 28 : node.type === "paper" ? 18 : 22;
              return (
                <g key={node.id} onMouseEnter={() => setHoveredNode(node.id)} onMouseLeave={() => setHoveredNode(null)}
                  style={{ cursor: "pointer" }}>
                  <circle cx={node.x} cy={node.y} r={isHovered ? r + 4 : r}
                    fill={c.fill} stroke={c.stroke} strokeWidth={isHovered ? 2.5 : 1.5}
                    style={{ transition: "all 0.2s" }} />
                  <text x={node.x} y={node.y + r + 14} textAnchor="middle"
                    fontSize={node.type === "project" ? 11 : 9}
                    fontWeight={node.type === "project" ? 600 : 400}
                    fill={isHovered ? c.text : "hsl(var(--muted-foreground))"}
                    style={{ transition: "fill 0.2s" }}>
                    {node.label.length > 20 ? node.label.slice(0, 18) + "…" : node.label}
                  </text>
                  {/* Type icon letter */}
                  <text x={node.x} y={node.y + 4} textAnchor="middle" fontSize={r > 22 ? 12 : 9} fontWeight={700} fill={c.text}>
                    {node.type[0].toUpperCase()}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>
    </div>
  );
}
