import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Camera, Layers, Mountain, Activity, Box, MessageSquare,
  ArrowRight, Sparkles, WandSparkles, BookOpen, Cpu, Zap, GraduationCap,
  Brain, Network, FlaskConical, FileText, Crown, Rocket, GitBranch, BarChart3, Lightbulb,
} from "lucide-react";
import Playground from "@/components/Playground";
import { useSubscription } from "@/hooks/useSubscription";

const pipelineModules = [
  {
    name: "Camera Image Formation",
    desc: "Pinhole model, calibration, lens distortion, image formation, coordinate systems",
    icon: Camera,
    path: "/module/camera",
    color: "var(--module-camera)",
    step: 1,
  },
  {
    name: "Semantic Information",
    desc: "Classification, object detection, segmentation — understanding what's in the scene",
    icon: Layers,
    path: "/module/semantic",
    color: "var(--module-semantic)",
    step: 2,
  },
  {
    name: "Geometric Information",
    desc: "Depth estimation, stereo vision, pose estimation — recovering 3D structure",
    icon: Mountain,
    path: "/module/geometric",
    color: "var(--module-geometric)",
    step: 3,
  },
  {
    name: "Motion Estimation",
    desc: "Optical flow, tracking, action recognition, velocity estimation",
    icon: Activity,
    path: "/module/motion",
    color: "var(--module-motion)",
    step: 4,
  },
  {
    name: "3D Reconstruction & Rendering",
    desc: "SfM, Multi-View Stereo, NeRF, 3D Gaussian Splatting",
    icon: Box,
    path: "/module/reconstruction",
    color: "var(--module-reconstruction)",
    step: 5,
  },
  {
    name: "Scene Reasoning",
    desc: "CLIP, Florence-2, multimodal LLMs, visual grounding",
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

const copilotWorkflow = [
  { step: "1", label: "Create Project", desc: "Define your research question", icon: Lightbulb },
  { step: "2", label: "AI Analysis", desc: "Get papers, repos & insights", icon: Brain },
  { step: "3", label: "Hypotheses", desc: "Choose from AI-generated hypotheses", icon: GitBranch },
  { step: "4", label: "Notebook", desc: "Export runnable Jupyter notebooks", icon: Rocket },
];

export default function Dashboard() {
  const { isSubscribed } = useSubscription();
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
        <p className="text-xs font-mono text-accent uppercase tracking-widest mb-3">Interactive Computer Vision Learning Lab for Autonomous Systems</p>
        <p className="text-base md:text-lg text-muted-foreground max-w-3xl leading-relaxed">
          A complete computer vision curriculum with lecture-grade modules, conceptual labs, perception system design studios, 
          and AI-assisted research learning. Progress through the perception pipeline from camera physics to multimodal reasoning.
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
              <h3 className="font-semibold text-foreground">Camera Image Formation</h3>
              <p className="text-sm text-muted-foreground">Understand how cameras capture the 3D world as 2D images</p>
            </div>
            <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-primary transition-colors" />
          </div>
        </Link>
      </section>

      {/* Module Cards */}
      <section className="mb-10">
        <h2 className="text-lg font-semibold text-foreground mb-4">Learning Modules</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
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

      {/* Research Copilot — Premium Feature Showcase */}
      <section className="mb-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="relative rounded-2xl border-2 border-accent/30 bg-gradient-to-br from-accent/5 via-card/95 to-primary/5 backdrop-blur-sm p-6 md:p-8 overflow-hidden"
        >
          {/* Decorative glow */}
          <div className="absolute top-0 right-0 w-64 h-64 bg-accent/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-48 h-48 bg-primary/10 rounded-full blur-3xl translate-y-1/2 -translate-x-1/2 pointer-events-none" />

          <div className="relative z-10">
            {/* Header */}
            <div className="flex items-center gap-3 mb-2">
              <div className="h-10 w-10 rounded-xl bg-accent/15 flex items-center justify-center ring-2 ring-accent/20">
                <Sparkles className="h-5 w-5 text-accent" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-xl font-bold text-foreground">Research Copilot</h2>
                  <span className="text-[9px] px-2 py-0.5 rounded-full bg-accent/15 text-accent font-bold uppercase tracking-wider">Pro Feature</span>
                </div>
                <p className="text-xs text-muted-foreground">AI-powered research workflow for computer vision</p>
              </div>
            </div>

            {/* Description */}
            <p className="text-sm text-muted-foreground leading-relaxed mt-4 mb-6 max-w-2xl">
              Go from a research question to runnable experiments in minutes. The Research Copilot finds relevant papers, 
              generates hypotheses, and exports complete Jupyter notebooks with PyTorch code — all powered by AI.
            </p>

            {/* Workflow Steps */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
              {copilotWorkflow.map((item, i) => (
                <motion.div
                  key={item.label}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + i * 0.1, duration: 0.35 }}
                  className="relative rounded-xl border border-border/60 bg-card/80 p-4 text-center group"
                >
                  <div className="text-[10px] font-mono text-accent mb-2 font-bold">Step {item.step}</div>
                  <div className="h-9 w-9 rounded-lg bg-accent/10 flex items-center justify-center mx-auto mb-2">
                    <item.icon className="h-4.5 w-4.5 text-accent" />
                  </div>
                  <h4 className="text-xs font-semibold text-foreground mb-0.5">{item.label}</h4>
                  <p className="text-[10px] text-muted-foreground leading-snug">{item.desc}</p>
                  {i < copilotWorkflow.length - 1 && (
                    <ArrowRight className="hidden md:block absolute -right-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-accent/40 z-10" />
                  )}
                </motion.div>
              ))}
            </div>

            {/* Key Features */}
            <div className="flex flex-wrap gap-2 mb-6">
              {[
                { icon: FileText, label: "Paper Discovery" },
                { icon: BarChart3, label: "Metrics Tracking" },
                { icon: GitBranch, label: "Hypothesis Engine" },
                { icon: Rocket, label: "Notebook Export" },
              ].map((feat) => (
                <div key={feat.label} className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-border/50 bg-muted/30 text-[11px] text-muted-foreground">
                  <feat.icon className="h-3 w-3 text-accent/70" />
                  {feat.label}
                </div>
              ))}
            </div>

            {/* CTA */}
            {isSubscribed ? (
              <Link
                to="/research-copilot"
                className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg bg-accent text-accent-foreground text-sm font-semibold hover:bg-accent/90 transition-colors shadow-lg shadow-accent/20"
              >
                <Sparkles className="h-4 w-4" />
                Launch Research Copilot
                <ArrowRight className="h-4 w-4" />
              </Link>
            ) : (
              <Link
                to="/pricing"
                className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg bg-accent text-accent-foreground text-sm font-semibold hover:bg-accent/90 transition-colors shadow-lg shadow-accent/20"
              >
                <Crown className="h-4 w-4" />
                Upgrade to Pro to Unlock
                <ArrowRight className="h-4 w-4" />
              </Link>
            )}
          </div>
        </motion.div>
      </section>

      {/* Tutorials & Studios */}
      <section className="mb-10 grid grid-cols-1 md:grid-cols-2 gap-4">
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

        <Link
          to="/studios"
          className="flex items-center gap-4 rounded-xl border border-border bg-card/90 backdrop-blur-sm p-5 hover:border-accent/50 transition-all duration-200"
        >
          <div className="h-12 w-12 rounded-lg bg-accent/15 flex items-center justify-center glow-accent">
            <FlaskConical className="h-6 w-6 text-accent" />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-foreground">Perception Studios</h3>
            <p className="text-sm text-muted-foreground">Design complete perception pipelines in capstone-style exercises</p>
          </div>
          <ArrowRight className="h-5 w-5 text-muted-foreground" />
        </Link>
      </section>

      {/* GPU Info Banner */}
      <section className="mb-10">
        <div className="rounded-xl border border-accent/20 bg-accent/5 p-5">
          <div className="flex items-start gap-3">
            <Zap className="h-5 w-5 text-accent shrink-0 mt-0.5" />
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-1">GPU Playground Info</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Playgrounds run real computer vision models on GPU infrastructure. Learning modules, tutorials, and conceptual labs remain free. Playground runs may require GPU credits.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
