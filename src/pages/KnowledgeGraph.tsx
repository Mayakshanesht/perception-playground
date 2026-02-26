import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, ZoomIn, ZoomOut } from "lucide-react";
import { Link } from "react-router-dom";

interface GraphNode {
  id: string;
  label: string;
  type: "paper" | "architecture" | "task" | "dataset";
  x: number;
  y: number;
}

interface GraphEdge {
  from: string;
  to: string;
}

const nodes: GraphNode[] = [
  // Tasks
  { id: "t1", label: "Classification", type: "task", x: 100, y: 80 },
  { id: "t2", label: "Detection", type: "task", x: 300, y: 60 },
  { id: "t3", label: "Segmentation", type: "task", x: 500, y: 70 },
  { id: "t4", label: "Depth", type: "task", x: 700, y: 80 },
  { id: "t5", label: "Pose", type: "task", x: 840, y: 140 },
  { id: "t6", label: "Action", type: "task", x: 850, y: 300 },
  // Architectures
  { id: "a1", label: "ResNet", type: "architecture", x: 150, y: 220 },
  { id: "a2", label: "VGG", type: "architecture", x: 50, y: 200 },
  { id: "a3", label: "YOLO", type: "architecture", x: 310, y: 190 },
  { id: "a4", label: "Faster R-CNN", type: "architecture", x: 420, y: 220 },
  { id: "a5", label: "U-Net", type: "architecture", x: 540, y: 200 },
  { id: "a6", label: "Mask R-CNN", type: "architecture", x: 660, y: 220 },
  { id: "a7", label: "DPT", type: "architecture", x: 760, y: 210 },
  { id: "a8", label: "ViTPose", type: "architecture", x: 840, y: 210 },
  { id: "a9", label: "VideoMAE", type: "architecture", x: 840, y: 370 },
  { id: "a10", label: "SAM", type: "architecture", x: 620, y: 290 },
  // Datasets
  { id: "d1", label: "ImageNet", type: "dataset", x: 100, y: 380 },
  { id: "d2", label: "COCO", type: "dataset", x: 360, y: 380 },
  { id: "d3", label: "Pascal VOC", type: "dataset", x: 220, y: 410 },
  { id: "d4", label: "NYU Depth v2", type: "dataset", x: 640, y: 390 },
  { id: "d5", label: "Kinetics-400", type: "dataset", x: 840, y: 440 },
  // Papers
  { id: "p1", label: "He 2016", type: "paper", x: 200, y: 320 },
  { id: "p2", label: "Redmon 2016", type: "paper", x: 320, y: 310 },
  { id: "p3", label: "Ren 2015", type: "paper", x: 430, y: 330 },
  { id: "p4", label: "Ronneberger 2015", type: "paper", x: 540, y: 330 },
  { id: "p5", label: "Ranftl 2021", type: "paper", x: 740, y: 320 },
  { id: "p6", label: "Xu 2022", type: "paper", x: 840, y: 270 },
  { id: "p7", label: "Tong 2022", type: "paper", x: 840, y: 520 },
  { id: "p8", label: "Kirillov 2023", type: "paper", x: 620, y: 370 },
];

const edges: GraphEdge[] = [
  { from: "t1", to: "a1" }, { from: "t1", to: "a2" },
  { from: "t2", to: "a3" }, { from: "t2", to: "a4" },
  { from: "t3", to: "a5" }, { from: "t3", to: "a6" }, { from: "t3", to: "a10" },
  { from: "t4", to: "a7" }, { from: "t5", to: "a8" }, { from: "t6", to: "a9" },
  { from: "a1", to: "p1" }, { from: "a3", to: "p2" },
  { from: "a4", to: "p3" }, { from: "a5", to: "p4" },
  { from: "a7", to: "p5" }, { from: "a8", to: "p6" }, { from: "a9", to: "p7" }, { from: "a10", to: "p8" },
  { from: "a1", to: "d1" }, { from: "a2", to: "d1" },
  { from: "a3", to: "d2" }, { from: "a4", to: "d2" },
  { from: "a3", to: "d3" }, { from: "a4", to: "d3" },
  { from: "a5", to: "d2" }, { from: "a6", to: "d2" }, { from: "a10", to: "d2" },
  { from: "a7", to: "d4" }, { from: "a8", to: "d2" }, { from: "a9", to: "d5" },
  { from: "t2", to: "t3" }, { from: "t3", to: "t5" }, { from: "t5", to: "t6" }, { from: "t3", to: "t4" },
];

const typeConfig: Record<string, { color: string; label: string }> = {
  task: { color: "187, 85%, 53%", label: "Task" },
  architecture: { color: "265, 70%, 60%", label: "Architecture" },
  dataset: { color: "32, 95%, 55%", label: "Dataset" },
  paper: { color: "340, 75%, 55%", label: "Paper" },
};

export default function KnowledgeGraph() {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);

  const connectedNodes = useMemo(() => {
    if (!hoveredNode) return new Set<string>();
    const connected = new Set<string>([hoveredNode]);
    edges.forEach(e => {
      if (e.from === hoveredNode) connected.add(e.to);
      if (e.to === hoveredNode) connected.add(e.from);
    });
    return connected;
  }, [hoveredNode]);

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      <h1 className="text-2xl font-bold text-foreground tracking-tight mb-2">Knowledge Graph</h1>
      <p className="text-sm text-muted-foreground mb-6">Explore links between tasks, architectures, papers, and datasets. Hover a node to highlight its local neighborhood.</p>

      {/* Legend */}
      <div className="flex gap-4 mb-4">
        {Object.entries(typeConfig).map(([key, val]) => (
          <div key={key} className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full" style={{ backgroundColor: `hsl(${val.color})` }} />
            <span className="text-xs text-muted-foreground">{val.label}</span>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="flex gap-2 mb-4">
        <button onClick={() => setZoom(z => Math.min(z + 0.2, 2))} className="p-2 rounded-md border border-border bg-card text-muted-foreground hover:text-foreground">
          <ZoomIn className="h-4 w-4" />
        </button>
        <button onClick={() => setZoom(z => Math.max(z - 0.2, 0.5))} className="p-2 rounded-md border border-border bg-card text-muted-foreground hover:text-foreground">
          <ZoomOut className="h-4 w-4" />
        </button>
      </div>

      {/* Graph */}
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <div className="grid-pattern relative" style={{ height: 620 }}>
          <svg
            viewBox="0 0 920 600"
            className="w-full h-full"
            style={{ transform: `scale(${zoom})`, transformOrigin: "center" }}
          >
            {/* Edges */}
            {edges.map((edge, i) => {
              const from = nodes.find(n => n.id === edge.from)!;
              const to = nodes.find(n => n.id === edge.to)!;
              const isHighlighted = hoveredNode && connectedNodes.has(edge.from) && connectedNodes.has(edge.to);
              const isDimmed = hoveredNode && !isHighlighted;
              return (
                <line
                  key={i}
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  stroke={isHighlighted ? "hsl(187 85% 53%)" : "hsl(222 30% 18%)"}
                  strokeWidth={isHighlighted ? 2 : 1}
                  opacity={isDimmed ? 0.15 : 0.6}
                  className="transition-all duration-200"
                />
              );
            })}
            {/* Nodes */}
            {nodes.map((node) => {
              const config = typeConfig[node.type];
              const isDimmed = hoveredNode && !connectedNodes.has(node.id);
              return (
                <g
                  key={node.id}
                  onMouseEnter={() => setHoveredNode(node.id)}
                  onMouseLeave={() => setHoveredNode(null)}
                  className="cursor-pointer"
                  opacity={isDimmed ? 0.2 : 1}
                  style={{ transition: "opacity 0.2s" }}
                >
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={node.type === "task" ? 28 : 22}
                    fill={`hsl(${config.color} / 0.15)`}
                    stroke={`hsl(${config.color})`}
                    strokeWidth={hoveredNode === node.id ? 2 : 1}
                  />
                  <text
                    x={node.x}
                    y={node.y + 1}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill={`hsl(${config.color})`}
                    fontSize={9}
                    fontFamily="Inter, sans-serif"
                    fontWeight={500}
                  >
                    {node.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    </div>
  );
}
