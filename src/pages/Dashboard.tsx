import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Brain,
  Scan,
  Layers,
  Mountain,
  Move3D,
  Box,
  Users,
  Eye,
  Video,
  GitBranch,
  ArrowRight,
  Sparkles,
  WandSparkles,
} from "lucide-react";
import Playground from "@/components/Playground";

const modules = [
  { name: "Multi-Object Tracking", desc: "SORT, DeepSORT, data association and temporal identity", icon: Eye, path: "/module/tracking", color: "280 70% 55%" },
  { name: "Image Classification", desc: "AlexNet, VGG, ResNet, Vision Transformers", icon: Brain, path: "/module/classification", color: "187 85% 53%" },
  { name: "Object Detection", desc: "R-CNN family, YOLO, SSD", icon: Scan, path: "/module/detection", color: "160 84% 39%" },
  { name: "Segmentation", desc: "Semantic, instance and panoptic segmentation", icon: Layers, path: "/module/segmentation", color: "265 70% 60%" },
  { name: "Depth Estimation", desc: "Monocular depth, stereo vision, disparity", icon: Mountain, path: "/module/depth", color: "32 95% 55%" },
  { name: "Pose Estimation", desc: "2D/3D human pose and keypoint detection", icon: Users, path: "/module/pose", color: "50 90% 50%" },
  { name: "Action Recognition", desc: "3D CNNs and video transformers", icon: Video, path: "/module/action", color: "15 85% 55%" },
  { name: "Optical Flow", desc: "Motion fields, RAFT, Lucas-Kanade", icon: GitBranch, path: "/module/opticalflow", color: "170 80% 45%" },
  { name: "Structure from Motion", desc: "Feature matching, epipolar geometry, bundle adjustment", icon: Move3D, path: "/module/sfm", color: "340 75% 55%" },
  { name: "Neural Rendering", desc: "NeRF and Gaussian Splatting", icon: Box, path: "/module/nerf", color: "200 80% 55%" },
];

export default function Dashboard() {
  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto aurora-bg rounded-2xl">
      <div className="mb-10">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="h-4 w-4 text-primary" />
          <span className="text-xs font-mono text-primary uppercase tracking-widest">Perception Concept Studio</span>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-foreground tracking-tight mb-3">
          Promptable Vision Lab
        </h1>
        <p className="text-base md:text-lg text-muted-foreground max-w-3xl leading-relaxed">
          Explore modern perception pipelines with interactive detection, segmentation, pose, depth, and SAM 2 segmentation.
        </p>
      </div>

      <section className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <WandSparkles className="h-4 w-4 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">SAM 2 Segmentation Playground</h2>
        </div>
        <Playground
          title="SAM 2 Promptable Segmentation"
          description="Segment objects with SAM 2 prompts (points/boxes/masks) or run segment-everything on image/video input."
          taskType="sam2-segmentation"
          acceptVideo
          acceptImage
          modelName="sam2.1_b.pt"
          learningFocus="Try point/box prompts and compare mask stability across frames."
        />
      </section>

      <h2 className="text-lg font-semibold text-foreground mb-4">All Learning Modules</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
        {modules.map((mod, i) => (
          <motion.div
            key={mod.path}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.04 * i, duration: 0.32 }}
          >
            <Link
              to={mod.path}
              className="block rounded-xl border border-border/80 bg-card/90 backdrop-blur-sm p-5 group hover:border-primary/60 transition-all duration-200"
              style={{ boxShadow: `0 0 0px hsl(${mod.color} / 0)` }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.boxShadow = `0 0 28px -10px hsl(${mod.color} / 0.55)`;
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.boxShadow = `0 0 0px hsl(${mod.color} / 0)`;
              }}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="h-10 w-10 rounded-lg flex items-center justify-center" style={{ backgroundColor: `hsl(${mod.color} / 0.18)` }}>
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

      <Link
        to="/knowledge-graph"
        className="flex items-center gap-4 rounded-xl border border-border bg-card/90 backdrop-blur-sm p-5 hover:border-primary/50 transition-all duration-200"
      >
        <div className="h-12 w-12 rounded-lg bg-primary/15 flex items-center justify-center glow-primary">
          <GitBranch className="h-6 w-6 text-primary" />
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-foreground">Knowledge Graph</h3>
          <p className="text-sm text-muted-foreground">Explore links among tasks, datasets, architectures, and papers</p>
        </div>
        <ArrowRight className="h-5 w-5 text-muted-foreground" />
      </Link>
    </div>
  );
}

