import { useState, useMemo } from "react";
import { ArrowLeft, ZoomIn, ZoomOut, Maximize2, Filter } from "lucide-react";
import { Link } from "react-router-dom";

interface GraphNode {
  id: string;
  label: string;
  type: "paper" | "architecture" | "task" | "dataset" | "module";
  x: number;
  y: number;
}

interface GraphEdge {
  from: string;
  to: string;
}

const nodes: GraphNode[] = [
  // Modules (top row)
  { id: "mod-cam", label: "Camera", type: "module", x: 80, y: 40 },
  { id: "mod-sem", label: "Semantic", type: "module", x: 200, y: 40 },
  { id: "mod-geo", label: "Geometric", type: "module", x: 350, y: 40 },
  { id: "mod-mot", label: "Motion", type: "module", x: 500, y: 40 },
  { id: "mod-rec", label: "Reconstruction", type: "module", x: 650, y: 40 },
  { id: "mod-nlp", label: "Agentic AI", type: "module", x: 800, y: 40 },
  { id: "mod-rea", label: "Scene Reasoning", type: "module", x: 950, y: 40 },
  { id: "mod-gen", label: "Generative", type: "module", x: 1100, y: 40 },

  // Tasks (row 2)
  { id: "t1", label: "Classification", type: "task", x: 120, y: 140 },
  { id: "t2", label: "Detection", type: "task", x: 270, y: 130 },
  { id: "t3", label: "Segmentation", type: "task", x: 400, y: 140 },
  { id: "t4", label: "Depth Estimation", type: "task", x: 530, y: 130 },
  { id: "t5", label: "Pose Estimation", type: "task", x: 440, y: 210 },
  { id: "t6", label: "Optical Flow", type: "task", x: 650, y: 140 },
  { id: "t7", label: "Video Action", type: "task", x: 680, y: 220 },
  { id: "t8", label: "SfM / SLAM", type: "task", x: 800, y: 140 },
  { id: "t9", label: "Novel View Synth", type: "task", x: 900, y: 140 },
  { id: "t10", label: "VQA", type: "task", x: 1000, y: 140 },
  { id: "t11", label: "Image Captioning", type: "task", x: 1050, y: 220 },
  { id: "t12", label: "Stereo Matching", type: "task", x: 530, y: 210 },
  { id: "t13", label: "3D Detection", type: "task", x: 320, y: 210 },
  // NLP tasks
  { id: "t14", label: "Tokenization", type: "task", x: 750, y: 130 },
  { id: "t15", label: "Text Generation", type: "task", x: 830, y: 140 },
  { id: "t16", label: "Agent Planning", type: "task", x: 870, y: 220 },
  { id: "t19", label: "Tool Use", type: "task", x: 750, y: 220 },
  { id: "t20", label: "Multi-Agent", type: "task", x: 920, y: 220 },
  // Generative tasks
  { id: "t17", label: "Image Generation", type: "task", x: 1100, y: 140 },
  { id: "t18", label: "Image Editing", type: "task", x: 1150, y: 220 },

  // Architectures (row 3)
  { id: "a1", label: "ResNet", type: "architecture", x: 100, y: 300 },
  { id: "a2", label: "VGG", type: "architecture", x: 50, y: 360 },
  { id: "a3", label: "YOLO", type: "architecture", x: 220, y: 300 },
  { id: "a4", label: "Faster R-CNN", type: "architecture", x: 330, y: 300 },
  { id: "a5", label: "U-Net", type: "architecture", x: 400, y: 310 },
  { id: "a6", label: "Mask R-CNN", type: "architecture", x: 480, y: 300 },
  { id: "a7", label: "DPT / MiDaS", type: "architecture", x: 560, y: 310 },
  { id: "a8", label: "ViTPose", type: "architecture", x: 440, y: 370 },
  { id: "a9", label: "RAFT", type: "architecture", x: 660, y: 310 },
  { id: "a10", label: "SAM", type: "architecture", x: 480, y: 370 },
  { id: "a11", label: "NeRF", type: "architecture", x: 850, y: 300 },
  { id: "a12", label: "3D Gaussians", type: "architecture", x: 920, y: 310 },
  { id: "a13", label: "COLMAP", type: "architecture", x: 780, y: 300 },
  { id: "a14", label: "CLIP", type: "architecture", x: 1000, y: 300 },
  { id: "a15", label: "GPT-4V", type: "architecture", x: 1060, y: 310 },
  { id: "a16", label: "DETR", type: "architecture", x: 280, y: 360 },
  { id: "a17", label: "DeepLab", type: "architecture", x: 400, y: 370 },
  { id: "a18", label: "VideoMAE", type: "architecture", x: 700, y: 370 },
  { id: "a19", label: "Depth Anything", type: "architecture", x: 600, y: 370 },
  { id: "a20", label: "BEVDet", type: "architecture", x: 330, y: 370 },
  { id: "a21", label: "FlowNet", type: "architecture", x: 720, y: 310 },
  { id: "a22", label: "ORB-SLAM", type: "architecture", x: 790, y: 370 },
  // Agentic AI architectures
  { id: "a23", label: "Transformer", type: "architecture", x: 780, y: 300 },
  { id: "a24", label: "BERT", type: "architecture", x: 830, y: 370 },
  { id: "a25", label: "GPT", type: "architecture", x: 870, y: 300 },
  { id: "a26", label: "LLaMA", type: "architecture", x: 900, y: 370 },
  { id: "a33", label: "ReAct", type: "architecture", x: 760, y: 370 },
  { id: "a34", label: "LangGraph", type: "architecture", x: 940, y: 370 },
  { id: "a35", label: "MCP", type: "architecture", x: 820, y: 300 },
  // Generative architectures
  { id: "a27", label: "Stable Diffusion", type: "architecture", x: 1080, y: 300 },
  { id: "a28", label: "DALL-E", type: "architecture", x: 1140, y: 310 },
  { id: "a29", label: "StyleGAN", type: "architecture", x: 1180, y: 370 },
  { id: "a30", label: "VAE", type: "architecture", x: 1060, y: 370 },
  // VLM architectures
  { id: "a31", label: "ViT", type: "architecture", x: 940, y: 300 },
  { id: "a32", label: "LLaVA", type: "architecture", x: 980, y: 370 },

  // Papers (row 4)
  { id: "p1", label: "He 2016", type: "paper", x: 80, y: 440 },
  { id: "p2", label: "Redmon 2016", type: "paper", x: 200, y: 440 },
  { id: "p3", label: "Ren 2015", type: "paper", x: 310, y: 440 },
  { id: "p4", label: "Ronneberger 2015", type: "paper", x: 400, y: 440 },
  { id: "p5", label: "Ranftl 2021", type: "paper", x: 560, y: 440 },
  { id: "p6", label: "Teed 2020", type: "paper", x: 660, y: 440 },
  { id: "p7", label: "Mildenhall 2020", type: "paper", x: 850, y: 440 },
  { id: "p8", label: "Kirillov 2023", type: "paper", x: 480, y: 440 },
  { id: "p9", label: "Radford 2021", type: "paper", x: 1000, y: 440 },
  { id: "p10", label: "Carion 2020", type: "paper", x: 270, y: 440 },
  { id: "p11", label: "Chen 2018", type: "paper", x: 400, y: 490 },
  { id: "p12", label: "Kerbl 2023", type: "paper", x: 920, y: 440 },
  { id: "p13", label: "Yang 2024", type: "paper", x: 600, y: 490 },
  { id: "p14", label: "Xu 2022", type: "paper", x: 440, y: 490 },
  { id: "p15", label: "Dosovitskiy 2015", type: "paper", x: 720, y: 440 },
  { id: "p16", label: "Mur-Artal 2015", type: "paper", x: 790, y: 490 },

  // Datasets (row 5)
  { id: "d1", label: "ImageNet", type: "dataset", x: 80, y: 540 },
  { id: "d2", label: "COCO", type: "dataset", x: 300, y: 540 },
  { id: "d3", label: "Pascal VOC", type: "dataset", x: 180, y: 540 },
  { id: "d4", label: "NYU Depth v2", type: "dataset", x: 550, y: 540 },
  { id: "d5", label: "Cityscapes", type: "dataset", x: 420, y: 540 },
  { id: "d6", label: "Sintel", type: "dataset", x: 660, y: 540 },
  { id: "d7", label: "Kinetics-400", type: "dataset", x: 740, y: 540 },
  { id: "d8", label: "nuScenes", type: "dataset", x: 340, y: 580 },
  { id: "d9", label: "KITTI", type: "dataset", x: 500, y: 580 },
  { id: "d10", label: "ScanNet", type: "dataset", x: 860, y: 540 },
  { id: "d11", label: "LAION-5B", type: "dataset", x: 1000, y: 540 },
  { id: "d12", label: "Common Crawl", type: "dataset", x: 780, y: 580 },
  { id: "d13", label: "FFHQ", type: "dataset", x: 1140, y: 540 },
];

const edges: GraphEdge[] = [
  // Module → Task
  { from: "mod-cam", to: "t1" },
  { from: "mod-sem", to: "t1" }, { from: "mod-sem", to: "t2" }, { from: "mod-sem", to: "t3" }, { from: "mod-sem", to: "t13" },
  { from: "mod-geo", to: "t4" }, { from: "mod-geo", to: "t5" }, { from: "mod-geo", to: "t12" },
  { from: "mod-mot", to: "t6" }, { from: "mod-mot", to: "t7" },
  { from: "mod-rec", to: "t8" }, { from: "mod-rec", to: "t9" },
  { from: "mod-nlp", to: "t14" }, { from: "mod-nlp", to: "t15" }, { from: "mod-nlp", to: "t16" },
  { from: "mod-rea", to: "t10" }, { from: "mod-rea", to: "t11" },
  { from: "mod-gen", to: "t17" }, { from: "mod-gen", to: "t18" },

  // Task → Architecture
  { from: "t1", to: "a1" }, { from: "t1", to: "a2" },
  { from: "t2", to: "a3" }, { from: "t2", to: "a4" }, { from: "t2", to: "a16" },
  { from: "t3", to: "a5" }, { from: "t3", to: "a6" }, { from: "t3", to: "a10" }, { from: "t3", to: "a17" },
  { from: "t4", to: "a7" }, { from: "t4", to: "a19" },
  { from: "t5", to: "a8" },
  { from: "t6", to: "a9" }, { from: "t6", to: "a21" },
  { from: "t7", to: "a18" },
  { from: "t8", to: "a13" }, { from: "t8", to: "a22" },
  { from: "t9", to: "a11" }, { from: "t9", to: "a12" },
  { from: "t10", to: "a14" }, { from: "t10", to: "a15" },
  { from: "t11", to: "a15" },
  { from: "t12", to: "a9" },
  { from: "t13", to: "a20" },
  { from: "t14", to: "a23" }, { from: "t15", to: "a25" }, { from: "t15", to: "a26" }, { from: "t16", to: "a25" },
  { from: "t10", to: "a14" }, { from: "t10", to: "a15" }, { from: "t10", to: "a31" }, { from: "t10", to: "a32" },
  { from: "t17", to: "a27" }, { from: "t17", to: "a28" }, { from: "t17", to: "a29" }, { from: "t17", to: "a30" },
  { from: "t18", to: "a27" },

  // Architecture → Paper
  { from: "a1", to: "p1" }, { from: "a3", to: "p2" }, { from: "a4", to: "p3" },
  { from: "a5", to: "p4" }, { from: "a7", to: "p5" }, { from: "a9", to: "p6" },
  { from: "a11", to: "p7" }, { from: "a10", to: "p8" }, { from: "a14", to: "p9" },
  { from: "a16", to: "p10" }, { from: "a17", to: "p11" }, { from: "a12", to: "p12" },
  { from: "a19", to: "p13" }, { from: "a8", to: "p14" }, { from: "a21", to: "p15" },
  { from: "a22", to: "p16" },

  // Architecture → Dataset
  { from: "a1", to: "d1" }, { from: "a2", to: "d1" },
  { from: "a3", to: "d2" }, { from: "a4", to: "d2" }, { from: "a3", to: "d3" },
  { from: "a5", to: "d5" }, { from: "a6", to: "d2" }, { from: "a10", to: "d2" },
  { from: "a7", to: "d4" }, { from: "a17", to: "d5" },
  { from: "a9", to: "d6" }, { from: "a21", to: "d6" },
  { from: "a18", to: "d7" },
  { from: "a11", to: "d10" }, { from: "a13", to: "d10" },
  { from: "a14", to: "d11" },
  { from: "a20", to: "d8" }, { from: "a19", to: "d9" },
  { from: "a22", to: "d9" },
  { from: "a25", to: "d12" }, { from: "a26", to: "d12" }, { from: "a24", to: "d12" },
  { from: "a27", to: "d11" }, { from: "a29", to: "d13" },
  { from: "a31", to: "d1" }, { from: "a31", to: "d11" },

  // Cross-task relationships
  { from: "t2", to: "t3" }, { from: "t3", to: "t5" }, { from: "t3", to: "t4" },
  { from: "t4", to: "t12" }, { from: "t6", to: "t7" },
  { from: "t8", to: "t9" }, { from: "t4", to: "t8" },
  { from: "t2", to: "t13" },
];

const typeConfig: Record<string, { color: string; label: string; radius: number }> = {
  module:       { color: "142, 71%, 45%", label: "Module", radius: 32 },
  task:         { color: "187, 85%, 53%", label: "Task", radius: 28 },
  architecture: { color: "265, 70%, 60%", label: "Architecture", radius: 24 },
  paper:        { color: "340, 75%, 55%", label: "Paper", radius: 20 },
  dataset:      { color: "32, 95%, 55%", label: "Dataset", radius: 20 },
};

export default function KnowledgeGraph() {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [zoom, setZoom] = useState(0.85);
  const [filterType, setFilterType] = useState<string | null>(null);

  const connectedNodes = useMemo(() => {
    if (!hoveredNode) return new Set<string>();
    const connected = new Set<string>([hoveredNode]);
    edges.forEach(e => {
      if (e.from === hoveredNode) connected.add(e.to);
      if (e.to === hoveredNode) connected.add(e.from);
    });
    return connected;
  }, [hoveredNode]);

  const filteredNodes = filterType ? nodes.filter(n => n.type === filterType) : nodes;
  const filteredNodeIds = new Set(filteredNodes.map(n => n.id));

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      <h1 className="text-2xl font-bold text-foreground tracking-tight mb-2">Knowledge Graph</h1>
      <p className="text-sm text-muted-foreground mb-6">
        Explore the relationships between learning modules, perception tasks, model architectures, seminal papers, and benchmark datasets.
        Hover any node to highlight its connections.
      </p>

      {/* Legend & Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
        <div className="flex flex-wrap gap-3">
          {Object.entries(typeConfig).map(([key, val]) => (
            <button
              key={key}
              onClick={() => setFilterType(filterType === key ? null : key)}
              className={`flex items-center gap-2 px-2 py-1 rounded-lg transition-all text-xs ${
                filterType === key ? "bg-muted border border-border" : "hover:bg-muted/50"
              }`}
            >
              <div className="h-3 w-3 rounded-full" style={{ backgroundColor: `hsl(${val.color})` }} />
              <span className="text-muted-foreground">{val.label}</span>
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1">
          <button onClick={() => setZoom(z => Math.max(0.4, z - 0.1))} className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground">
            <ZoomOut className="h-3.5 w-3.5" />
          </button>
          <span className="text-[10px] text-muted-foreground w-10 text-center">{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom(z => Math.min(2, z + 0.1))} className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground">
            <ZoomIn className="h-3.5 w-3.5" />
          </button>
          <button onClick={() => setZoom(0.85)} className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground">
            <Maximize2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Graph */}
      <div className="rounded-xl border border-border bg-card overflow-auto">
        <div className="relative" style={{ height: 650, minWidth: 900 }}>
          <svg
            viewBox="0 0 1120 620"
            className="w-full h-full"
            style={{ transform: `scale(${zoom})`, transformOrigin: "center" }}
          >
            {/* Edges */}
            {edges.map((edge, i) => {
              const from = nodes.find(n => n.id === edge.from);
              const to = nodes.find(n => n.id === edge.to);
              if (!from || !to) return null;
              if (filterType && !filteredNodeIds.has(from.id) && !filteredNodeIds.has(to.id)) return null;
              const isHighlighted = hoveredNode && connectedNodes.has(edge.from) && connectedNodes.has(edge.to);
              const isDimmed = hoveredNode && !isHighlighted;
              return (
                <line key={i} x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                  stroke={isHighlighted ? "hsl(187 85% 53%)" : "hsl(var(--border))"}
                  strokeWidth={isHighlighted ? 2 : 0.8}
                  opacity={isDimmed ? 0.08 : 0.4}
                  className="transition-all duration-200"
                />
              );
            })}
            {/* Nodes */}
            {nodes.map((node) => {
              const config = typeConfig[node.type];
              const isDimmed = hoveredNode && !connectedNodes.has(node.id);
              const isFiltered = filterType && node.type !== filterType;
              if (isFiltered) return null;
              const r = config.radius;
              const isHovered = hoveredNode === node.id;
              return (
                <g key={node.id}
                  onMouseEnter={() => setHoveredNode(node.id)}
                  onMouseLeave={() => setHoveredNode(null)}
                  className="cursor-pointer"
                  opacity={isDimmed ? 0.15 : 1}
                  style={{ transition: "opacity 0.2s" }}
                >
                  <circle cx={node.x} cy={node.y} r={isHovered ? r + 3 : r}
                    fill={`hsl(${config.color} / 0.12)`}
                    stroke={`hsl(${config.color} / ${isHovered ? 0.8 : 0.4})`}
                    strokeWidth={isHovered ? 2 : 1}
                    style={{ transition: "all 0.15s" }}
                  />
                  <text x={node.x} y={node.y + 1} textAnchor="middle" dominantBaseline="middle"
                    fill={`hsl(${config.color})`}
                    fontSize={node.type === "module" ? 10 : node.type === "task" ? 9 : 8}
                    fontWeight={node.type === "module" ? 700 : node.type === "task" ? 600 : 500}
                  >
                    {node.label.length > 16 ? node.label.slice(0, 14) + "…" : node.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Hovered node info */}
      {hoveredNode && (() => {
        const node = nodes.find(n => n.id === hoveredNode);
        if (!node) return null;
        const connections = edges.filter(e => e.from === hoveredNode || e.to === hoveredNode);
        const connectedLabels = connections.map(e => {
          const otherId = e.from === hoveredNode ? e.to : e.from;
          return nodes.find(n => n.id === otherId)?.label;
        }).filter(Boolean);
        return (
          <div className="mt-3 rounded-lg border border-border bg-card/80 p-3 text-xs">
            <span className="font-semibold text-foreground">{node.label}</span>
            <span className="text-muted-foreground ml-2">({typeConfig[node.type].label})</span>
            <span className="text-muted-foreground ml-2">— Connected to: {connectedLabels.join(", ")}</span>
          </div>
        );
      })()}
    </div>
  );
}
