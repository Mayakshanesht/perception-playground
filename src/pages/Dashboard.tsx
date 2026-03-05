import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Camera, Layers, Mountain, Activity, Box, MessageSquare,
  ArrowRight, Sparkles, WandSparkles, BookOpen, Cpu, Zap, GraduationCap,
} from "lucide-react";
import Playground from "@/components/Playground";

const pipelineModules = [
  {
    name: "Camera Image Generation",
    desc: "Pinhole model, calibration, lens distortion, image formation",
    icon: Camera,
    path: "/module/camera",
    color: "var(--module-camera)",
    step: 1,
  },
  {
    name: "Semantic Information",
    desc: "Classification, object detection, and segmentation",
    icon: Layers,
    path: "/module/semantic",
    color: "var(--module-semantic)",
    step: 2,
  },
  {
    name: "Geometric Information",
    desc: "Depth estimation and pose estimation",
    icon: Mountain,
    path: "/module/geometric",
    color: "var(--module-geometric)",
    step: 3,
  },
  {
    name: "Motion Estimation",
    desc: "Tracking, optical flow, action recognition, velocity",
    icon: Activity,
    path: "/module/motion",
    color: "var(--module-motion)",
    step: 4,
  },
  {
    name: "3D Reconstruction & Rendering",
    desc: "Structure from Motion, NeRF, Gaussian Splatting",
    icon: Box,
    path: "/module/reconstruction",
    color: "var(--module-reconstruction)",
    step: 5,
  },
  {
    name: "Scene Reasoning",
    desc: "Multimodal LLMs, visual grounding, Florence-2",
    icon: MessageSquare,
    path: "/module/scene-reasoning",
    color: "var(--module-reasoning)",
    step: 6,
  },
];

const pipelineSteps = [
  { label: "Camera", color: "var(--module-camera)" },
  { label: "Semantics", color: "var(--module-semantic)" },
  { label: "Geometry", color: "var(--module-geometric)" },
  { label: "Motion", color: "var(--module-motion)" },
  { label: "Reconstruction", color: "var(--module-reconstruction)" },
  { label: "Reasoning", color: "var(--module-reasoning)" },
];

const playgrounds = [
  { name: "Object Detection", module: "Semantic", path: "/module/semantic" },
  { name: "Segmentation", module: "Semantic", path: "/module/semantic" },
  { name: "Depth Estimation", module: "Geometric", path: "/module/geometric" },
  { name: "Pose Estimation", module: "Geometric", path: "/module/geometric" },
  { name: "Speed Estimation", module: "Motion", path: "/module/motion" },
];

export default function Dashboard() {
  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto aurora-bg rounded-2xl">
      {/* Header */}
      <div className="mb-10">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="h-4 w-4 text-primary" />
          <span className="text-xs font-mono text-primary uppercase tracking-widest">KnowGraph</span>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4">
          <span className="gradient-text">Perception Lab</span>
        </h1>
        <p className="text-base md:text-lg text-muted-foreground max-w-3xl leading-relaxed">
          An interactive environment for learning computer vision and perception systems through theory, equations, and real model experimentation. 
          Progress through the perception pipeline from camera physics to multimodal reasoning.
        </p>
      </div>

      {/* Pipeline Diagram */}
      <section className="mb-10">
        <div className="rounded-xl border border-border bg-card/80 backdrop-blur-sm p-6">
          <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider mb-4">The Perception Pipeline</h2>
          <p className="text-sm text-muted-foreground leading-relaxed mb-6">
            Computer vision follows a structured pipeline: a <strong className="text-foreground">Camera</strong> captures the 3D world as a 2D image.
            <strong className="text-foreground"> Semantic</strong> analysis extracts what objects are present.
            <strong className="text-foreground"> Geometric</strong> reasoning recovers 3D structure.
            <strong className="text-foreground"> Motion</strong> estimation tracks changes across frames.
            <strong className="text-foreground"> 3D Reconstruction</strong> builds complete scene models.
            Finally, <strong className="text-foreground"> Scene Reasoning</strong> with multimodal LLMs enables open-ended visual understanding.
          </p>
          
          {/* Visual pipeline */}
          <div className="flex flex-wrap items-center gap-2">
            {pipelineSteps.map((step, i) => (
              <div key={step.label} className="flex items-center gap-2">
                <Link
                  to={pipelineModules[i].path}
                  className="px-3 py-2 rounded-lg border border-border bg-muted/30 text-xs font-medium text-foreground hover:border-primary/40 hover:bg-primary/5 transition-all"
                  style={{ borderColor: `hsl(${step.color} / 0.3)` }}
                >
                  <span style={{ color: `hsl(${step.color})` }}>{step.label}</span>
                </Link>
                {i < pipelineSteps.length - 1 && (
                  <ArrowRight className="h-3 w-3 text-muted-foreground/40 shrink-0" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Continue Learning */}
      <section className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <GraduationCap className="h-4 w-4 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">Continue Learning</h2>
        </div>
        <Link
          to="/module/camera"
          className="block rounded-xl border border-border bg-card/90 backdrop-blur-sm p-5 hover:border-primary/50 transition-all duration-200 group"
        >
          <div className="flex items-center gap-4">
            <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center">
              <Camera className="h-6 w-6 text-primary" />
            </div>
            <div className="flex-1">
              <p className="text-xs text-muted-foreground mb-0.5">Start with the foundation</p>
              <h3 className="font-semibold text-foreground">Camera Image Generation</h3>
              <p className="text-sm text-muted-foreground">Understand how cameras capture the 3D world as 2D images</p>
            </div>
            <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-primary transition-colors" />
          </div>
        </Link>
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

      {/* Available Playgrounds */}
      <section className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <Cpu className="h-4 w-4 text-accent" />
          <h2 className="text-lg font-semibold text-foreground">Available Playgrounds</h2>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          {playgrounds.map((pg) => (
            <Link
              key={pg.name}
              to={pg.path}
              className="rounded-xl border border-border bg-card/80 p-4 text-center hover:border-primary/40 hover:bg-primary/5 transition-all group"
            >
              <p className="text-sm font-medium text-foreground group-hover:text-primary transition-colors">{pg.name}</p>
              <p className="text-[10px] text-muted-foreground mt-1">{pg.module} Module</p>
            </Link>
          ))}
        </div>
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
              className="block rounded-xl border border-border/80 bg-card/90 backdrop-blur-sm p-5 group hover:border-primary/40 transition-all duration-200"
              style={{ boxShadow: `0 0 0px hsl(${mod.color} / 0)` }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.boxShadow = `0 0 28px -10px hsl(${mod.color} / 0.4)`;
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.boxShadow = `0 0 0px hsl(${mod.color} / 0)`;
              }}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-lg flex items-center justify-center" style={{ backgroundColor: `hsl(${mod.color} / 0.12)` }}>
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

      {/* GPU Info Banner */}
      <section className="mb-10">
        <div className="rounded-xl border border-accent/20 bg-accent/5 p-5">
          <div className="flex items-start gap-3">
            <Zap className="h-5 w-5 text-accent shrink-0 mt-0.5" />
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-1">GPU Playground Info</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Playgrounds run real computer vision models on GPU infrastructure. Learning modules and tutorials remain free. Playground runs may require GPU credits.
              </p>
            </div>
          </div>
        </div>
      </section>

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
