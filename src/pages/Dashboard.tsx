import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Camera, Layers, Mountain, Activity, Box, MessageSquare,
  ArrowRight, Sparkles, WandSparkles, BookOpen, Cpu, Zap, GraduationCap,
  Brain, Network, FlaskConical, FileText, Crown, Rocket, GitBranch, BarChart3, Lightbulb,
  Eye, ChevronRight, Type, Paintbrush,
} from "lucide-react";
import Playground from "@/components/Playground";
import { useSubscription } from "@/hooks/useSubscription";

const pipelineModules = [
  { name: "Camera Image Formation", desc: "Pinhole model, calibration, lens distortion, coordinate systems", icon: Camera, path: "/module/camera", color: "var(--module-camera)", step: 1 },
  { name: "Semantic Information", desc: "Classification, detection, segmentation — what's in the scene", icon: Layers, path: "/module/semantic", color: "var(--module-semantic)", step: 2 },
  { name: "Geometric Information", desc: "Depth estimation, stereo vision, pose — recovering 3D structure", icon: Mountain, path: "/module/geometric", color: "var(--module-geometric)", step: 3 },
  { name: "Motion Estimation", desc: "Optical flow, tracking, action recognition, velocity", icon: Activity, path: "/module/motion", color: "var(--module-motion)", step: 4 },
  { name: "3D Reconstruction", desc: "SfM, Multi-View Stereo, NeRF, Gaussian Splatting", icon: Box, path: "/module/reconstruction", color: "var(--module-reconstruction)", step: 5 },
  { name: "Scene Reasoning", desc: "CLIP, Florence-2, multimodal LLMs, visual grounding", icon: MessageSquare, path: "/module/scene-reasoning", color: "var(--module-reasoning)", step: 6 },
  { name: "NLP & Large Language Models", desc: "Tokenization, transformers, BERT/GPT, RLHF, agents, LoRA", icon: Type, path: "/module/nlp-llm", color: "var(--module-nlp)", step: 7 },
  { name: "Generative Vision", desc: "VAEs, GANs, diffusion models, Stable Diffusion, ControlNet", icon: Paintbrush, path: "/module/generative-vision", color: "var(--module-generative)", step: 8 },
];

const pipelineSteps = [
  { label: "Camera", color: "var(--module-camera)" },
  { label: "Semantics", color: "var(--module-semantic)" },
  { label: "Geometry", color: "var(--module-geometric)" },
  { label: "Motion", color: "var(--module-motion)" },
  { label: "Reconstruction", color: "var(--module-reconstruction)" },
  { label: "Reasoning", color: "var(--module-reasoning)" },
  { label: "NLP & LLMs", color: "var(--module-nlp)" },
  { label: "Generative", color: "var(--module-generative)" },
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
    <div className="p-5 md:p-8 max-w-7xl mx-auto">
      {/* Hero Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="mb-8"
      >
        <div className="flex items-center gap-2 mb-2">
          <div className="h-7 w-7 rounded-lg bg-primary/15 flex items-center justify-center">
            <Eye className="h-3.5 w-3.5 text-primary" />
          </div>
          <span className="text-[11px] font-mono text-primary uppercase tracking-[0.2em] font-medium">KnowGraph Perception Lab</span>
        </div>
        <h1 className="text-3xl md:text-4xl font-bold tracking-tight mb-2">
          <span className="gradient-text">Computer Vision</span>
          <span className="text-foreground"> Learning Lab</span>
        </h1>
        <p className="text-sm text-muted-foreground max-w-2xl leading-relaxed">
          Master the full perception pipeline — from camera physics to multimodal reasoning. 
          Lecture-grade theory, interactive playgrounds, and AI-assisted research tools.
        </p>
      </motion.div>

      {/* Pipeline Overview — compact */}
      <section className="mb-8">
        <div className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-4 md:p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xs font-semibold text-foreground uppercase tracking-wider">Perception Pipeline</h2>
            <Link to="/module/camera" className="text-[10px] text-primary hover:underline flex items-center gap-1">
              Start learning <ChevronRight className="h-3 w-3" />
            </Link>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            {pipelineSteps.map((step, i) => (
              <div key={step.label} className="flex items-center gap-1.5">
                <Link
                  to={pipelineModules[i].path}
                  className="px-2.5 py-1.5 rounded-md border border-border/60 bg-muted/20 text-[11px] font-medium hover:bg-primary/5 hover:border-primary/30 transition-all"
                  style={{ borderColor: `hsl(${step.color} / 0.25)` }}
                >
                  <span style={{ color: `hsl(${step.color})` }}>{step.label}</span>
                </Link>
                {i < pipelineSteps.length - 1 && (
                  <ArrowRight className="h-2.5 w-2.5 text-muted-foreground/30 shrink-0" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ★ Research Copilot — Premium Showcase (moved up) */}
      <section className="mb-8">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="relative rounded-2xl border border-accent/25 bg-gradient-to-br from-accent/[0.06] via-card/90 to-primary/[0.04] backdrop-blur-sm overflow-hidden"
        >
          {/* Decorative */}
          <div className="absolute top-0 right-0 w-72 h-72 bg-accent/8 rounded-full blur-[80px] -translate-y-1/3 translate-x-1/4 pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-56 h-56 bg-primary/8 rounded-full blur-[60px] translate-y-1/3 -translate-x-1/4 pointer-events-none" />

          <div className="relative z-10 p-5 md:p-7">
            {/* Top row */}
            <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-5">
              <div className="flex items-start gap-3">
                <div className="h-11 w-11 rounded-xl bg-accent/12 flex items-center justify-center ring-1 ring-accent/20 shrink-0 mt-0.5">
                  <Sparkles className="h-5 w-5 text-accent" />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <h2 className="text-lg font-bold text-foreground">Research Copilot</h2>
                    <span className="text-[8px] px-1.5 py-0.5 rounded-full bg-accent/15 text-accent font-bold uppercase tracking-wider leading-none">Pro</span>
                  </div>
                  <p className="text-xs text-muted-foreground max-w-md leading-relaxed">
                    Go from research question to runnable experiments. AI finds papers, generates hypotheses, and exports PyTorch notebooks.
                  </p>
                </div>
              </div>
              {isSubscribed ? (
                <Link
                  to="/research-copilot"
                  className="inline-flex items-center gap-2 px-5 py-2 rounded-lg bg-accent text-accent-foreground text-xs font-semibold hover:bg-accent/90 transition-colors shadow-md shadow-accent/15 shrink-0 self-start"
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  Launch Copilot
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              ) : (
                <Link
                  to="/pricing"
                  className="inline-flex items-center gap-2 px-5 py-2 rounded-lg bg-accent text-accent-foreground text-xs font-semibold hover:bg-accent/90 transition-colors shadow-md shadow-accent/15 shrink-0 self-start"
                >
                  <Crown className="h-3.5 w-3.5" />
                  Upgrade to Unlock
                </Link>
              )}
            </div>

            {/* Workflow Steps */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5 mb-4">
              {copilotWorkflow.map((item, i) => (
                <motion.div
                  key={item.label}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.25 + i * 0.08, duration: 0.3 }}
                  className="relative rounded-lg border border-border/50 bg-card/70 p-3 text-center"
                >
                  <div className="text-[9px] font-mono text-accent/70 mb-1.5 font-semibold">Step {item.step}</div>
                  <div className="h-8 w-8 rounded-md bg-accent/10 flex items-center justify-center mx-auto mb-1.5">
                    <item.icon className="h-4 w-4 text-accent" />
                  </div>
                  <h4 className="text-[11px] font-semibold text-foreground">{item.label}</h4>
                  <p className="text-[9px] text-muted-foreground leading-snug mt-0.5">{item.desc}</p>
                  {i < copilotWorkflow.length - 1 && (
                    <ArrowRight className="hidden md:block absolute -right-2 top-1/2 -translate-y-1/2 h-3 w-3 text-accent/30 z-10" />
                  )}
                </motion.div>
              ))}
            </div>

            {/* Feature tags */}
            <div className="flex flex-wrap gap-1.5">
              {[
                { icon: FileText, label: "Paper Discovery" },
                { icon: BarChart3, label: "Metrics Tracking" },
                { icon: GitBranch, label: "Hypothesis Engine" },
                { icon: Rocket, label: "Notebook Export" },
              ].map((feat) => (
                <div key={feat.label} className="flex items-center gap-1 px-2 py-1 rounded-md border border-border/40 bg-muted/20 text-[10px] text-muted-foreground">
                  <feat.icon className="h-2.5 w-2.5 text-accent/60" />
                  {feat.label}
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </section>

      {/* Quick Start + Tutorials & Studios row */}
      <section className="mb-8 grid grid-cols-1 md:grid-cols-3 gap-3">
        <Link
          to="/module/camera"
          className="flex items-center gap-3 rounded-xl border border-border bg-card/80 backdrop-blur-sm p-4 hover:border-primary/40 transition-all group"
        >
          <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
            <GraduationCap className="h-5 w-5 text-primary" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">Start Learning</p>
            <h3 className="text-sm font-semibold text-foreground truncate">Camera Module</h3>
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors shrink-0" />
        </Link>

        <Link
          to="/tutorials"
          className="flex items-center gap-3 rounded-xl border border-border bg-card/80 backdrop-blur-sm p-4 hover:border-primary/40 transition-all group"
        >
          <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
            <BookOpen className="h-5 w-5 text-primary" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5">
              <h3 className="text-sm font-semibold text-foreground">Tutorials</h3>
              <Crown className="h-3 w-3 text-accent" />
            </div>
            <p className="text-[10px] text-muted-foreground">Guided Colab notebooks</p>
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors shrink-0" />
        </Link>

        <Link
          to="/studios"
          className="flex items-center gap-3 rounded-xl border border-border bg-card/80 backdrop-blur-sm p-4 hover:border-accent/40 transition-all group"
        >
          <div className="h-10 w-10 rounded-lg bg-accent/10 flex items-center justify-center shrink-0">
            <FlaskConical className="h-5 w-5 text-accent" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-foreground">Perception Studios</h3>
            <p className="text-[10px] text-muted-foreground">Pipeline design exercises</p>
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-accent transition-colors shrink-0" />
        </Link>
      </section>

      {/* Module Cards */}
      <section className="mb-8">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">Learning Modules</h2>
          <span className="text-[10px] text-muted-foreground font-mono">8 modules</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {pipelineModules.map((mod, i) => (
            <motion.div
              key={mod.path}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.03 * i, duration: 0.3 }}
            >
              <Link
                to={mod.path}
                className="block rounded-xl border border-border/70 bg-card/80 backdrop-blur-sm p-4 group hover:border-primary/30 transition-all duration-200"
                style={{ boxShadow: `0 0 0px hsl(${mod.color} / 0)` }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.boxShadow = `0 0 24px -8px hsl(${mod.color} / 0.35)`;
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.boxShadow = `0 0 0px hsl(${mod.color} / 0)`;
                }}
              >
                <div className="flex items-center gap-3 mb-2">
                  <div className="h-9 w-9 rounded-lg flex items-center justify-center shrink-0" style={{ backgroundColor: `hsl(${mod.color} / 0.1)` }}>
                    <mod.icon className="h-4.5 w-4.5" style={{ color: `hsl(${mod.color})` }} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-foreground text-sm truncate">{mod.name}</h3>
                    <span className="text-[9px] font-mono text-muted-foreground">Step {mod.step}</span>
                  </div>
                  <ArrowRight className="h-3.5 w-3.5 text-muted-foreground/40 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                </div>
                <p className="text-[11px] text-muted-foreground leading-relaxed pl-12">{mod.desc}</p>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Playgrounds Grid */}
      <section className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <Cpu className="h-3.5 w-3.5 text-accent" />
          <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">GPU Playgrounds</h2>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2.5">
          {playgrounds.map((pg) => (
            <Link
              key={pg.name}
              to={pg.path}
              className="rounded-lg border border-border/60 bg-card/70 p-3 text-center hover:border-primary/30 hover:bg-primary/5 transition-all group"
            >
              <p className="text-xs font-medium text-foreground group-hover:text-primary transition-colors">{pg.name}</p>
              <p className="text-[9px] text-muted-foreground mt-0.5">{pg.module}</p>
            </Link>
          ))}
        </div>
      </section>

      {/* SAM2 Playground */}
      <section className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <WandSparkles className="h-3.5 w-3.5 text-primary" />
          <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">SAM 2 Playground</h2>
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

      {/* GPU Info — minimal */}
      <section className="mb-6">
        <div className="rounded-lg border border-accent/15 bg-accent/[0.03] px-4 py-3 flex items-center gap-3">
          <Zap className="h-4 w-4 text-accent shrink-0" />
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            <span className="text-foreground font-medium">GPU Playgrounds</span> run real CV models on GPU infrastructure. Learning modules remain free. Playground runs may require GPU credits.
          </p>
        </div>
      </section>
    </div>
  );
}
