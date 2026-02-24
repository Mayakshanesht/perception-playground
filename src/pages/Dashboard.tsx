import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Brain, Scan, Layers, Mountain, Move3D, Box, Users, Eye, Video, GitBranch,
  ArrowRight, Sparkles
} from "lucide-react";

const modules = [
  {
    name: "Image Classification",
    desc: "AlexNet, VGG, ResNet, Vision Transformers",
    icon: Brain,
    path: "/module/classification",
    color: "187 85% 53%",
  },
  {
    name: "Object Detection",
    desc: "R-CNN family, YOLO, SSD — one vs two stage",
    icon: Scan,
    path: "/module/detection",
    color: "160 84% 39%",
  },
  {
    name: "Segmentation",
    desc: "Semantic, Instance & Panoptic segmentation",
    icon: Layers,
    path: "/module/segmentation",
    color: "265 70% 60%",
  },
  {
    name: "Depth Estimation",
    desc: "Monocular depth, stereo vision, disparity maps",
    icon: Mountain,
    path: "/module/depth",
    color: "32 95% 55%",
  },
  {
    name: "Structure from Motion",
    desc: "Feature matching, epipolar geometry, bundle adjustment",
    icon: Move3D,
    path: "/module/sfm",
    color: "340 75% 55%",
  },
  {
    name: "Neural Rendering",
    desc: "NeRF, Gaussian Splatting, volume rendering",
    icon: Box,
    path: "/module/nerf",
    color: "200 80% 55%",
  },
  {
    name: "Pose Estimation",
    desc: "2D/3D human pose, keypoint detection",
    icon: Users,
    path: "/module/pose",
    color: "50 90% 50%",
  },
  {
    name: "Multi-Object Tracking",
    desc: "SORT, DeepSORT, tracking by detection",
    icon: Eye,
    path: "/module/tracking",
    color: "280 70% 55%",
  },
  {
    name: "Action Recognition",
    desc: "2D CNN+LSTM, 3D CNN, Video Transformers",
    icon: Video,
    path: "/module/action",
    color: "15 85% 55%",
  },
];

const stats = [
  { label: "Modules", value: "9" },
  { label: "Architectures", value: "25+" },
  { label: "Key Papers", value: "60+" },
  { label: "Interactive Demos", value: "15+" },
];

export default function Dashboard() {
  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Hero */}
      <div className="mb-10">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="h-4 w-4 text-primary" />
          <span className="text-xs font-mono text-primary uppercase tracking-widest">Autonomous Vision Studio</span>
        </div>
        <h1 className="text-4xl font-bold text-foreground tracking-tight mb-3">
          Perception Lab
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl leading-relaxed">
          Explore computer vision architectures through interactive visualizations, 
          step-by-step algorithm flows, and experiment sandboxes.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
        {stats.map((s, i) => (
          <motion.div
            key={s.label}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08 }}
            className="rounded-lg border border-border bg-card p-4"
          >
            <p className="text-2xl font-bold text-foreground font-mono">{s.value}</p>
            <p className="text-xs text-muted-foreground mt-1">{s.label}</p>
          </motion.div>
        ))}
      </div>

      {/* Module Grid */}
      <h2 className="text-lg font-semibold text-foreground mb-4">Learning Modules</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
        {modules.map((mod, i) => (
          <motion.div
            key={mod.path}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 * i, duration: 0.35 }}
          >
            <Link
              to={mod.path}
              className="block rounded-xl border border-border bg-card p-5 group hover:border-primary/40 transition-all duration-200"
              style={{
                boxShadow: `0 0 0px hsl(${mod.color} / 0)`,
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.boxShadow = `0 0 24px -8px hsl(${mod.color} / 0.3)`;
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.boxShadow = `0 0 0px hsl(${mod.color} / 0)`;
              }}
            >
              <div className="flex items-start justify-between mb-3">
                <div
                  className="h-10 w-10 rounded-lg flex items-center justify-center"
                  style={{ backgroundColor: `hsl(${mod.color} / 0.12)` }}
                >
                  <mod.icon className="h-5 w-5" style={{ color: `hsl(${mod.color})` }} />
                </div>
                <ArrowRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity -translate-x-1 group-hover:translate-x-0 duration-200" />
              </div>
              <h3 className="font-semibold text-foreground text-sm mb-1">{mod.name}</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">{mod.desc}</p>
            </Link>
          </motion.div>
        ))}
      </div>

      {/* Knowledge Graph link */}
      <Link
        to="/knowledge-graph"
        className="flex items-center gap-4 rounded-xl border border-border bg-card p-5 hover:border-primary/40 transition-all duration-200 glow-primary/0 hover:glow-primary"
      >
        <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
          <GitBranch className="h-6 w-6 text-primary" />
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-foreground">Knowledge Graph</h3>
          <p className="text-sm text-muted-foreground">Explore connections between papers, architectures, tasks & datasets</p>
        </div>
        <ArrowRight className="h-5 w-5 text-muted-foreground" />
      </Link>
    </div>
  );
}
