import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Camera, Layers, Mountain, Activity, Box, MessageSquare,
  ArrowRight, Sparkles, WandSparkles, BookOpen,
} from "lucide-react";
import Playground from "@/components/Playground";

const pipelineModules = [
  {
    name: "Camera Image Generation",
    desc: "Pinhole model, calibration, lens distortion, image formation",
    icon: Camera,
    path: "/module/camera",
    color: "220 70% 55%",
    step: 1,
  },
  {
    name: "Semantic Information",
    desc: "Classification, object detection, and segmentation",
    icon: Layers,
    path: "/module/semantic",
    color: "187 85% 53%",
    step: 2,
  },
  {
    name: "Geometric Information",
    desc: "Depth estimation and pose estimation",
    icon: Mountain,
    path: "/module/geometric",
    color: "32 95% 55%",
    step: 3,
  },
  {
    name: "Motion Estimation",
    desc: "Tracking, optical flow, action recognition, velocity",
    icon: Activity,
    path: "/module/motion",
    color: "280 70% 55%",
    step: 4,
  },
  {
    name: "3D Reconstruction & Rendering",
    desc: "Structure from Motion, NeRF, Gaussian Splatting",
    icon: Box,
    path: "/module/reconstruction",
    color: "340 75% 55%",
    step: 5,
  },
  {
    name: "Scene Reasoning",
    desc: "Multimodal LLMs, visual grounding, Florence-2",
    icon: MessageSquare,
    path: "/module/scene-reasoning",
    color: "290 70% 55%",
    step: 6,
  },
];

export default function Dashboard() {
  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto aurora-bg rounded-2xl">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="h-4 w-4 text-primary" />
          <span className="text-xs font-mono text-primary uppercase tracking-widest">Perception Lab</span>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-foreground tracking-tight mb-3">
          Computer Vision Learning Platform
        </h1>
        <p className="text-base md:text-lg text-muted-foreground max-w-3xl leading-relaxed">
          Master the complete perception pipeline — from how cameras form images to how AI reasons about visual scenes.
        </p>
      </div>

      {/* Pipeline Overview */}
      <section className="mb-10">
        <div className="rounded-xl border border-border bg-card/80 backdrop-blur-sm p-6">
          <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider mb-3">The Perception Pipeline</h2>
          <p className="text-sm text-muted-foreground leading-relaxed mb-5">
            Computer vision follows a structured pipeline: a <strong className="text-foreground">Camera</strong> captures the 3D world as a 2D image.
            <strong className="text-foreground"> Semantic</strong> analysis extracts what objects are present.
            <strong className="text-foreground"> Geometric</strong> reasoning recovers 3D structure (depth, pose).
            <strong className="text-foreground"> Motion</strong> estimation tracks changes across frames.
            <strong className="text-foreground"> 3D Reconstruction</strong> builds complete scene models.
            Finally, <strong className="text-foreground"> Scene Reasoning</strong> with multimodal LLMs enables open-ended visual understanding.
          </p>
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
            {["Camera", "→", "Semantics", "→", "Geometry", "→", "Motion", "→", "Reconstruction", "→", "Reasoning"].map((item, i) => (
              item === "→" ? (
                <ArrowRight key={i} className="h-3 w-3 text-primary/60" />
              ) : (
                <span key={i} className="px-2.5 py-1 rounded-md bg-primary/10 text-primary font-medium">{item}</span>
              )
            ))}
          </div>
        </div>
      </section>

      {/* SAM2 Playground */}
      <section className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <WandSparkles className="h-4 w-4 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">SAM 2 Segmentation Playground</h2>
        </div>
        <Playground
          title="SAM 2 Segmentation"
          description="Segment objects with SAM 2 prompts (points/boxes/masks) or segment-everything on image input."
          taskType="sam2-segmentation"
          acceptVideo={false}
          acceptImage
          modelName="sam2.1_b.pt"
          learningFocus="Try point/box prompts and compare mask stability across frames."
        />
      </section>

      {/* Module Cards */}
      <h2 className="text-lg font-semibold text-foreground mb-4">Learning Modules</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
        {pipelineModules.map((mod, i) => (
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
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-lg flex items-center justify-center" style={{ backgroundColor: `hsl(${mod.color} / 0.18)` }}>
                    <mod.icon className="h-5 w-5" style={{ color: `hsl(${mod.color})` }} />
                  </div>
                  <span className="text-[10px] font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded">Step {mod.step}</span>
                </div>
                <ArrowRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
              <h3 className="font-semibold text-foreground text-sm mb-1">{mod.name}</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">{mod.desc}</p>
            </Link>
          </motion.div>
        ))}
      </div>

      {/* Tutorials Link */}
      <Link
        to="/tutorials"
        className="flex items-center gap-4 rounded-xl border border-border bg-card/90 backdrop-blur-sm p-5 hover:border-primary/50 transition-all duration-200"
      >
        <div className="h-12 w-12 rounded-lg bg-primary/15 flex items-center justify-center glow-primary">
          <BookOpen className="h-6 w-6 text-primary" />
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-foreground">Hands-On Tutorials</h3>
          <p className="text-sm text-muted-foreground">Guided Colab notebooks for detection, segmentation, depth, and more</p>
        </div>
        <ArrowRight className="h-5 w-5 text-muted-foreground" />
      </Link>
    </div>
  );
}
