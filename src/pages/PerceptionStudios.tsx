import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft, ArrowRight, Sparkles, FlaskConical, Rocket,
  Search, FileText, Lightbulb, Code2, ChevronDown, BookOpen,
  Brain, Target, Layers, Zap, CheckCircle2
} from "lucide-react";
import { Link } from "react-router-dom";

/* ─── Two showcase problem statements ─── */
const showcases = [
  {
    id: "real-time-panoptic",
    title: "Real-Time Panoptic Segmentation for Autonomous Driving",
    question: "How can we achieve real-time panoptic segmentation (>30 FPS) on edge GPUs while maintaining competitive PQ scores on Cityscapes?",
    tags: ["Panoptic Segmentation", "Real-Time Inference", "Edge Deployment", "Autonomous Driving"],
    stages: {
      analysis: {
        title: "AI Paper Analysis",
        icon: Search,
        papers: [
          { title: "Panoptic-DeepLab: A Simple, Strong, and Fast Baseline", authors: "Cheng et al., CVPR 2020", insight: "Bottom-up approach eliminates instance branch overhead, enabling single-shot panoptic prediction." },
          { title: "Real-Time Panoptic Segmentation from Dense Detections", authors: "Hou et al., CVPR 2020", insight: "Dense detection + stuff segmentation merged via efficient fusion module at 30+ FPS." },
          { title: "EfficientPS: Efficient Panoptic Segmentation", authors: "Mohan & Valada, IJCV 2021", insight: "Shared backbone with 2-way FPN achieves state-of-the-art PQ with 2× fewer FLOPs." },
        ],
        repos: [
          { name: "detectron2/panoptic-deeplab", stars: "29.5k", url: "https://github.com/facebookresearch/detectron2" },
          { name: "robot-learning-freiburg/EfficientPS", stars: "420", url: "https://github.com/robot-learning-freiburg/EfficientPS" },
        ],
      },
      hypotheses: [
        {
          label: "A",
          title: "Knowledge Distillation from Heavy Teacher",
          description: "Train a lightweight MobileNetV3 student using panoptic distillation from a Swin-L teacher, targeting TensorRT INT8 on Jetson Orin.",
          metrics: "Target: PQ 58+ on Cityscapes, 35 FPS on Jetson Orin Nano",
        },
        {
          label: "B",
          title: "Unified Query-Based Architecture",
          description: "Adapt Mask2Former with a pruned backbone and dynamic query selection to reduce computation while preserving mask quality.",
          metrics: "Target: PQ 62+ on Cityscapes, 25 FPS on RTX 3060",
        },
        {
          label: "C",
          title: "Temporal Fusion with Video Context",
          description: "Extend single-frame panoptic with lightweight temporal attention across 3 frames for consistent video panoptic segmentation.",
          metrics: "Target: VPQ 55+ on Cityscapes-VPS, 20 FPS",
        },
      ],
      notebook: {
        title: "Generated Research Notebook",
        sections: [
          "Environment Setup (PyTorch, MMDet, Cityscapes API)",
          "Dataset Loading with Panoptic Annotations",
          "MobileNetV3 + Panoptic-DeepLab Architecture",
          "Knowledge Distillation Training Loop",
          "PQ/SQ/RQ Evaluation Pipeline",
          "TensorRT Export & Latency Benchmarking",
          "W&B Experiment Tracking Integration",
        ],
      },
    },
  },
  {
    id: "monocular-3d-detection",
    title: "Monocular 3D Object Detection from Single Camera",
    question: "Can we achieve LiDAR-competitive 3D detection (NDS > 0.45 on nuScenes) using only monocular camera images?",
    tags: ["3D Detection", "Monocular Depth", "BEV Perception", "nuScenes"],
    stages: {
      analysis: {
        title: "AI Paper Analysis",
        icon: Search,
        papers: [
          { title: "BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye-View", authors: "Huang et al., 2022", insight: "Lift-Splat-Shoot paradigm projects 2D features into 3D BEV space for detection." },
          { title: "MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection", authors: "Zhang et al., ICCV 2023", insight: "Depth-aware cross-attention in transformer decoder improves 3D localization from single images." },
          { title: "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data", authors: "Yang et al., CVPR 2024", insight: "Foundation depth model trained on 62M images provides strong monocular depth priors." },
        ],
        repos: [
          { name: "HuangJunJie2017/BEVDet", stars: "1.8k", url: "https://github.com/HuangJunJie2017/BEVDet" },
          { name: "DepthAnything/Depth-Anything-V2", stars: "3.2k", url: "https://github.com/DepthAnything/Depth-Anything-V2" },
        ],
      },
      hypotheses: [
        {
          label: "A",
          title: "Depth Foundation Model + BEV Lifting",
          description: "Use frozen Depth Anything V2 as depth backbone, lift image features to BEV using predicted depth distributions, then apply CenterPoint head.",
          metrics: "Target: NDS 0.42+ on nuScenes val, single camera",
        },
        {
          label: "B",
          title: "Geometric-Aware Transformer Decoder",
          description: "Extend MonoDETR with camera-intrinsic positional encoding and depth-bin classification for more accurate 3D box regression.",
          metrics: "Target: mAP 0.35+ on nuScenes val",
        },
        {
          label: "C",
          title: "Pseudo-LiDAR with Temporal Stereo",
          description: "Generate pseudo-LiDAR point clouds from consecutive frames using structure-from-motion depth, then apply PointPillars detector.",
          metrics: "Target: NDS 0.38+ with 2-frame input",
        },
      ],
      notebook: {
        title: "Generated Research Notebook",
        sections: [
          "Environment Setup (PyTorch, MMDet3D, nuScenes devkit)",
          "nuScenes Dataset Loading & Preprocessing",
          "Depth Anything V2 Feature Extraction",
          "BEV Feature Lifting Module",
          "CenterPoint Detection Head",
          "NDS/mAP Evaluation on nuScenes",
          "Qualitative Visualization (3D Boxes on Images)",
        ],
      },
    },
  },
];

type Stage = "overview" | "analysis" | "hypotheses" | "notebook";

function StageIndicator({ current, stages }: { current: Stage; stages: { key: Stage; label: string; icon: any }[] }) {
  const currentIdx = stages.findIndex(s => s.key === current);
  return (
    <div className="flex items-center gap-1">
      {stages.map((s, i) => {
        const done = i < currentIdx;
        const active = i === currentIdx;
        return (
          <div key={s.key} className="flex items-center gap-1">
            <div className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[10px] font-medium transition-all ${
              active ? "bg-primary/15 text-primary border border-primary/30" :
              done ? "bg-accent/10 text-accent" : "text-muted-foreground/50"
            }`}>
              {done ? <CheckCircle2 className="h-3 w-3" /> : <s.icon className="h-3 w-3" />}
              <span className="hidden sm:inline">{s.label}</span>
            </div>
            {i < stages.length - 1 && <ArrowRight className="h-3 w-3 text-muted-foreground/30" />}
          </div>
        );
      })}
    </div>
  );
}

const stageList: { key: Stage; label: string; icon: any }[] = [
  { key: "overview", label: "Problem", icon: Target },
  { key: "analysis", label: "AI Analysis", icon: Search },
  { key: "hypotheses", label: "Hypotheses", icon: Lightbulb },
  { key: "notebook", label: "Notebook", icon: Code2 },
];

function ShowcaseCard({ showcase }: { showcase: typeof showcases[0] }) {
  const [stage, setStage] = useState<Stage>("overview");
  const [expanded, setExpanded] = useState(true);

  const nextStage = () => {
    const idx = stageList.findIndex(s => s.key === stage);
    if (idx < stageList.length - 1) setStage(stageList[idx + 1].key);
  };

  const prevStage = () => {
    const idx = stageList.findIndex(s => s.key === stage);
    if (idx > 0) setStage(stageList[idx - 1].key);
  };

  return (
    <div className="rounded-xl border border-border bg-card/80 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-4 p-5 hover:bg-muted/20 transition-colors text-left"
      >
        <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
          <Brain className="h-5 w-5 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-foreground text-sm">{showcase.title}</h3>
          <p className="text-xs text-muted-foreground mt-0.5 truncate">{showcase.question}</p>
        </div>
        <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform shrink-0 ${expanded ? "rotate-180" : ""}`} />
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 space-y-4">
              {/* Stage indicator */}
              <StageIndicator current={stage} stages={stageList} />

              {/* Stage content */}
              <div className="min-h-[240px]">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={stage}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.2 }}
                  >
                    {stage === "overview" && (
                      <div className="space-y-4">
                        <div className="rounded-lg bg-primary/5 border border-primary/20 p-4">
                          <h4 className="text-xs font-semibold text-primary uppercase tracking-wider mb-2 flex items-center gap-2">
                            <Target className="h-3.5 w-3.5" /> Research Question
                          </h4>
                          <p className="text-sm text-foreground leading-relaxed italic">"{showcase.question}"</p>
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {showcase.tags.map(t => (
                            <span key={t} className="text-[10px] px-2 py-1 rounded-full bg-accent/10 text-accent font-medium">{t}</span>
                          ))}
                        </div>
                        <p className="text-xs text-muted-foreground">
                          The Research Copilot analyzes this question across arxiv papers, GitHub repos, and benchmarks to produce actionable research hypotheses and a ready-to-run notebook.
                        </p>
                      </div>
                    )}

                    {stage === "analysis" && (
                      <div className="space-y-4">
                        <h4 className="text-xs font-semibold text-foreground uppercase tracking-wider flex items-center gap-2">
                          <Search className="h-3.5 w-3.5 text-primary" /> Found Papers & Repositories
                        </h4>
                        <div className="space-y-2">
                          {showcase.stages.analysis.papers.map((p, i) => (
                            <div key={i} className="rounded-lg bg-muted/30 border border-border p-3">
                              <div className="flex items-start gap-2">
                                <FileText className="h-3.5 w-3.5 text-primary mt-0.5 shrink-0" />
                                <div>
                                  <p className="text-xs font-semibold text-foreground">{p.title}</p>
                                  <p className="text-[10px] text-muted-foreground mt-0.5">{p.authors}</p>
                                  <p className="text-[11px] text-muted-foreground mt-1 leading-relaxed">💡 {p.insight}</p>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {showcase.stages.analysis.repos.map((r, i) => (
                            <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted/20 border border-border text-xs">
                              <Code2 className="h-3 w-3 text-accent" />
                              <span className="text-foreground font-medium">{r.name}</span>
                              <span className="text-muted-foreground">⭐ {r.stars}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {stage === "hypotheses" && (
                      <div className="space-y-3">
                        <h4 className="text-xs font-semibold text-foreground uppercase tracking-wider flex items-center gap-2">
                          <Lightbulb className="h-3.5 w-3.5 text-accent" /> Generated Research Hypotheses
                        </h4>
                        {showcase.stages.hypotheses.map((h) => (
                          <div key={h.label} className="rounded-lg border border-border bg-muted/20 p-3.5">
                            <div className="flex items-center gap-2 mb-1.5">
                              <span className="h-5 w-5 rounded-md bg-primary/15 text-primary text-[10px] font-bold flex items-center justify-center">{h.label}</span>
                              <span className="text-xs font-semibold text-foreground">{h.title}</span>
                            </div>
                            <p className="text-[11px] text-muted-foreground leading-relaxed mb-2">{h.description}</p>
                            <div className="flex items-center gap-1.5 text-[10px] text-accent font-medium">
                              <Zap className="h-3 w-3" />
                              {h.metrics}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {stage === "notebook" && (
                      <div className="space-y-4">
                        <h4 className="text-xs font-semibold text-foreground uppercase tracking-wider flex items-center gap-2">
                          <Code2 className="h-3.5 w-3.5 text-accent" /> {showcase.stages.notebook.title}
                        </h4>
                        <div className="rounded-lg bg-muted/40 border border-border overflow-hidden">
                          <div className="px-3 py-2 border-b border-border bg-muted/60 flex items-center gap-2">
                            <div className="flex gap-1">
                              <div className="h-2 w-2 rounded-full bg-destructive/60" />
                              <div className="h-2 w-2 rounded-full bg-accent/60" />
                              <div className="h-2 w-2 rounded-full bg-primary/60" />
                            </div>
                            <span className="text-[10px] text-muted-foreground font-mono">research_notebook.ipynb</span>
                          </div>
                          <div className="p-3 space-y-1.5">
                            {showcase.stages.notebook.sections.map((s, i) => (
                              <div key={i} className="flex items-center gap-2 text-xs">
                                <span className="text-muted-foreground/50 font-mono w-4 text-right text-[10px]">[{i + 1}]</span>
                                <CheckCircle2 className="h-3 w-3 text-accent/60" />
                                <span className="text-foreground/80">{s}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                        <p className="text-[11px] text-muted-foreground">
                          The Research Copilot generates a complete, runnable Jupyter notebook with PyTorch code, dataset loaders, training loops, and evaluation — ready for Google Colab or Kaggle.
                        </p>
                      </div>
                    )}
                  </motion.div>
                </AnimatePresence>
              </div>

              {/* Navigation */}
              <div className="flex items-center justify-between pt-2 border-t border-border">
                <button
                  onClick={prevStage}
                  disabled={stage === "overview"}
                  className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ArrowLeft className="h-3 w-3" /> Previous
                </button>
                {stage !== "notebook" ? (
                  <button
                    onClick={nextStage}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary/10 text-primary text-xs font-medium hover:bg-primary/20 transition-colors"
                  >
                    Next Stage <ArrowRight className="h-3 w-3" />
                  </button>
                ) : (
                  <Link
                    to="/research-copilot"
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 transition-colors"
                  >
                    <Sparkles className="h-3 w-3" /> Try Research Copilot
                  </Link>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function PerceptionStudios() {
  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      <div className="mb-8">
        <div className="flex items-center gap-3 mb-3">
          <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center">
            <FlaskConical className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-foreground tracking-tight">Perception Studios</h1>
            <p className="text-sm text-muted-foreground">See how the Research Copilot tackles real CV research problems</p>
          </div>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed max-w-3xl">
          Walk through two real computer vision research problems and see exactly what the Research Copilot produces at each stage — 
          from paper discovery and analysis, to hypothesis generation, to a ready-to-run research notebook. 
          Then try it yourself with your own research question.
        </p>
      </div>

      {/* Copilot pipeline overview */}
      <div className="rounded-xl border border-border bg-muted/20 p-4 mb-6">
        <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3">Research Copilot Pipeline</h3>
        <div className="flex flex-wrap items-center gap-2">
          {[
            { icon: Target, label: "Research Question", color: "text-primary" },
            { icon: Search, label: "AI Paper Analysis", color: "text-primary" },
            { icon: Lightbulb, label: "Hypothesis Generation", color: "text-accent" },
            { icon: Code2, label: "Notebook Generation", color: "text-accent" },
            { icon: Rocket, label: "Run & Iterate", color: "text-primary" },
          ].map((s, i) => (
            <div key={s.label} className="flex items-center gap-2">
              <div className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-card border border-border">
                <s.icon className={`h-3.5 w-3.5 ${s.color}`} />
                <span className="text-[11px] font-medium text-foreground">{s.label}</span>
              </div>
              {i < 4 && <ArrowRight className="h-3 w-3 text-muted-foreground/40 shrink-0" />}
            </div>
          ))}
        </div>
      </div>

      {/* Showcase cards */}
      <div className="space-y-4 mb-8">
        {showcases.map(s => (
          <ShowcaseCard key={s.id} showcase={s} />
        ))}
      </div>

      {/* CTA */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="rounded-xl border border-primary/30 bg-primary/5 p-6 text-center"
      >
        <Sparkles className="h-8 w-8 text-primary mx-auto mb-3" />
        <h3 className="text-lg font-bold text-foreground mb-2">Ready to start your own research?</h3>
        <p className="text-sm text-muted-foreground mb-4 max-w-lg mx-auto">
          The Research Copilot analyzes papers, generates hypotheses, and creates runnable notebooks for any computer vision research question.
        </p>
        <Link
          to="/research-copilot"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-colors"
        >
          <Rocket className="h-4 w-4" /> Launch Research Copilot
        </Link>
      </motion.div>
    </div>
  );
}
