import ModulePage from "@/components/ModulePage";
import { ModuleContent, PlaygroundConfig } from "@/data/moduleContent";
import { moduleContents } from "@/data/moduleContent";
import { MathEquation } from "@/components/MathBlock";
import AITutor from "@/components/AITutor";
import { OpticalFlowCanvas, TrackingCanvas, Tracking3DCanvas, ActionRecognitionCanvas } from "@/components/MotionCanvasAnimations";
import { ArrowLeft, GraduationCap, Waves, Eye, Box, Clapperboard, Layers } from "lucide-react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useSectionObserver } from "@/hooks/useSectionObserver";
import { Progress } from "@/components/ui/progress";

const velocityPlayground: PlaygroundConfig = {
  title: "Speed Estimation Playground",
  description: "Upload a video to estimate object speeds using YOLO tracking with velocity estimation.",
  taskType: "velocity-estimation",
  acceptVideo: true,
  acceptImage: false,
  modelName: "yolo26n.pt (tracking + speed estimation)",
  learningFocus: "Compare estimated speeds across different objects and analyze how tracking quality affects velocity accuracy.",
};

// Merge tracking + action + optical flow + add 3D tracking
const motionModule: ModuleContent = {
  id: "motion",
  title: "Motion Estimation",
  subtitle: "Understand visual motion — track objects across frames, recognize actions, and estimate optical flow and velocity.",
  color: "280 70% 55%",
  theory: [
    ...moduleContents.tracking.theory,
    ...moduleContents.opticalflow.theory,
    ...moduleContents.action.theory,
    // 3D Object Tracking theory
    {
      title: "3D Multi-Object Tracking",
      content: "3D MOT extends 2D tracking to 3D space using LiDAR point clouds or multi-camera systems. Each object's state is represented as [x, y, z, θ, l, w, h, vx, vy] in world coordinates. 3D detectors like CenterPoint or PointPillars produce 3D bounding boxes, which are associated across frames using 3D IoU or center distance. The Kalman filter operates in 3D with a constant-velocity motion model. This is critical for autonomous driving where metric 3D positions and velocities are needed for planning.",
      equations: [
        {
          label: "3D IoU",
          tex: "\\text{IoU}_{3D}(A, B) = \\frac{\\text{Vol}(A \\cap B)}{\\text{Vol}(A \\cup B)}",
        },
        {
          label: "3D State Vector",
          tex: "\\mathbf{s} = [x, y, z, \\theta, l, w, h, v_x, v_y]^T",
        },
      ],
    },
    {
      title: "Point Cloud Tracking",
      content: "Point cloud-based tracking directly operates on LiDAR scans without projecting to 2D. Methods like PointTrack segment individual object point clouds and match them across frames using shape descriptors. Scene flow estimation computes per-point 3D motion vectors — the 3D analog of optical flow. Self-supervised approaches like PointPWC-Net learn 3D scene flow from unlabeled data.",
      equations: [
        {
          label: "Scene Flow",
          tex: "\\mathbf{SF}(\\mathbf{p}) = (\\Delta x, \\Delta y, \\Delta z) \\in \\mathbb{R}^3 \\quad \\forall \\mathbf{p} \\in \\mathcal{P}",
        },
        {
          label: "Chamfer Distance",
          tex: "d_{CD}(P, Q) = \\frac{1}{|P|} \\sum_{p \\in P} \\min_{q \\in Q} \\|p - q\\|^2 + \\frac{1}{|Q|} \\sum_{q \\in Q} \\min_{p \\in P} \\|p - q\\|^2",
        },
      ],
    },
  ],
  algorithms: [
    ...moduleContents.tracking.algorithms,
    ...moduleContents.opticalflow.algorithms,
    ...moduleContents.action.algorithms,
    {
      name: "AB3DMOT — 3D Multi-Object Tracking",
      steps: [
        { step: "3D Detection", detail: "CenterPoint / PointPillars produces 3D bounding boxes from LiDAR" },
        { step: "Kalman Predict (3D)", detail: "Constant-velocity model predicts next 3D state for each track" },
        { step: "3D IoU Association", detail: "Compute 3D IoU between predicted and detected boxes; run Hungarian algorithm" },
        { step: "State Update", detail: "Kalman update with matched detections in 3D space" },
        { step: "Track Management", detail: "Birth: new tracks from unmatched detections; Death: remove after T_lost frames" },
      ],
    },
  ],
  papers: [
    ...moduleContents.tracking.papers,
    ...moduleContents.opticalflow.papers,
    ...moduleContents.action.papers,
    { year: 2020, title: "AB3DMOT", authors: "Weng & Kitani", venue: "IROS", summary: "Simple 3D MOT baseline: 3D Kalman filter + 3D IoU matching. Strong and fast." },
    { year: 2021, title: "CenterPoint", authors: "Yin et al.", venue: "CVPR", summary: "Center-based 3D detection and tracking from point clouds. State-of-the-art on nuScenes." },
    { year: 2022, title: "SimpleTrack", authors: "Pang et al.", venue: "ICLR", summary: "Improved 3D tracking with GNN-based association and multi-frame aggregation." },
    { year: 2023, title: "PointPWC-Net", authors: "Wu et al.", venue: "CVPR", summary: "Self-supervised 3D scene flow estimation from point clouds." },
  ].sort((a, b) => a.year - b.year),
  playgrounds: [velocityPlayground],
};

const color = motionModule.color;

const theoryByTitle: Record<string, typeof motionModule.theory[0]> = {};
motionModule.theory.forEach(s => { theoryByTitle[s.title] = s; });

function TheoryInline({ title }: { title: string }) {
  const section = theoryByTitle[title];
  if (!section) return null;
  return (
    <div className="concept-card">
      <div className="flex items-center flex-wrap gap-y-1 mb-3">
        <h3 className="font-semibold text-foreground text-sm">{section.title}</h3>
        <AITutor conceptTitle={section.title} conceptContent={section.content} moduleName="Motion Estimation" />
      </div>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">{section.content}</p>
      {section.equations?.map((eq) => (
        <div key={eq.label} className="mb-3">
          <MathEquation tex={eq.tex} label={eq.label} />
          {eq.variables && eq.variables.length > 0 && (
            <div className="mt-1.5 rounded-lg bg-muted/30 border border-border p-3">
              <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">Where</p>
              <div className="space-y-0.5">
                {eq.variables.map((v: any) => (
                  <p key={v.symbol} className="text-xs text-muted-foreground">
                    <span className="font-mono text-foreground">{v.symbol}</span> = {v.meaning}
                  </p>
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function ContentCard({ title, children, accent }: { title: string; children: React.ReactNode; accent?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card/50 p-4">
      <p className="text-[10px] font-semibold uppercase tracking-wider mb-2" style={{ color: accent || `hsl(${color})` }}>{title}</p>
      <div className="text-sm text-muted-foreground leading-relaxed">{children}</div>
    </div>
  );
}

function SectionHeader({ icon: Icon, title, number, subtitle }: { icon: any; title: string; number: number; subtitle?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="flex items-start gap-4 mb-6"
    >
      <div
        className="h-10 w-10 rounded-xl flex items-center justify-center shrink-0 mt-1"
        style={{ backgroundColor: `hsl(${color} / 0.12)` }}
      >
        <Icon className="h-5 w-5" style={{ color: `hsl(${color})` }} />
      </div>
      <div>
        <p className="text-[10px] font-mono font-bold uppercase tracking-widest mb-1" style={{ color: `hsl(${color})` }}>Part {number}</p>
        <h2 className="text-xl font-bold text-foreground tracking-tight">{title}</h2>
        {subtitle && <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{subtitle}</p>}
      </div>
    </motion.div>
  );
}

export default function MotionModule() {
  const progressPct = useSectionObserver("motion", ['optical-flow', 'tracking', 'tracking-3d', 'action', 'review']);

  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      <div className="flex items-start gap-4 mb-8">
        <div className="h-12 w-12 rounded-xl flex items-center justify-center shrink-0" style={{ backgroundColor: `hsl(${color} / 0.12)` }}>
          <GraduationCap className="h-6 w-6" style={{ color: `hsl(${color})` }} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground tracking-tight">{motionModule.title}</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{motionModule.subtitle}</p>
        </div>
      </div>

      {/* Learning flow nav */}
      <div className="rounded-xl border border-border bg-muted/30 p-4 mb-8">
        <h2 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3">Structured Learning Flow</h2>
        <div className="grid sm:grid-cols-3 lg:grid-cols-5 gap-2">
          {[
            { id: "optical-flow", icon: "🌊", label: "Optical Flow" },
            { id: "tracking", icon: "🎯", label: "2D Tracking (MOT)" },
            { id: "tracking-3d", icon: "📦", label: "3D Object Tracking" },
            { id: "action", icon: "🎬", label: "Action Recognition" },
            { id: "review", icon: "📚", label: "Papers & Practice" },
          ].map((item) => (
            <a
              key={item.id}
              href={`#${item.id}`}
              className="rounded-lg border border-border bg-card p-2.5 hover:border-primary/40 transition-colors text-center"
            >
              <p className="text-sm mb-0.5">{item.icon}</p>
              <p className="text-xs text-foreground font-medium">{item.label}</p>
            </a>
          ))}
        </div>
      </div>

      <div className="space-y-12">

        {/* ═══ Part 1: Optical Flow ═══ */}
        <section id="optical-flow">
          <SectionHeader
            icon={Waves}
            title="Optical Flow"
            number={1}
            subtitle="Estimate per-pixel motion between consecutive frames — the fundamental representation of visual motion. One equation, two unknowns: the aperture problem."
          />
          <div className="space-y-4">
            <OpticalFlowCanvas />

            <TheoryInline title="What is Optical Flow?" />
            <TheoryInline title="Brightness Constancy Assumption" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Aperture Problem" accent="#e040fb">
                The optical flow equation Iₓu + Iᵧv + Iₜ = 0 gives one constraint for two unknowns (u, v). Through a small window, you can only measure flow perpendicular to edges — not the full 2D motion.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  Lucas-Kanade: assumes flow constant in local window<br />
                  Horn-Schunck: global smoothness regularization
                </div>
              </ContentCard>
              <ContentCard title="RAFT: Modern Optical Flow" accent="#00bcd4">
                RAFT builds a 4D all-pairs correlation volume, then iteratively refines flow with a GRU. 12 iterations produce state-of-the-art results with strong generalization.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  Correlate all pixel pairs → GRU refine ×12
                </div>
              </ContentCard>
            </div>

            <TheoryInline title="Lucas-Kanade Method" />
            <TheoryInline title="RAFT: Recurrent All-Pairs Field Transforms" />
          </div>
        </section>

        {/* ═══ Part 2: Multi-Object Tracking ═══ */}
        <section id="tracking">
          <SectionHeader
            icon={Eye}
            title="Multi-Object Tracking (2D)"
            number={2}
            subtitle="Follow multiple objects across video frames while maintaining consistent identities. The tracking-by-detection paradigm: detect independently, associate across time."
          />
          <div className="space-y-4">
            <TrackingCanvas />

            <TheoryInline title="Tracking by Detection" />
            <TheoryInline title="Kalman Filter for Motion Prediction" />

            <div className="grid md:grid-cols-3 gap-4">
              <ContentCard title="SORT Pipeline" accent="#e040fb">
                Simple Online Realtime Tracking: Kalman filter + Hungarian algorithm + IoU matching. Fast baseline (~260 FPS).
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  State: [x, y, a, h, ẋ, ẏ, ȧ, ḣ]
                </div>
              </ContentCard>
              <ContentCard title="DeepSORT" accent="#00bcd4">
                Adds CNN appearance features (128-d Re-ID). Combined cost: motion (Mahalanobis) + appearance (cosine). Handles long occlusions.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  cᵢⱼ = λ·d_mahal + (1-λ)·d_cosine
                </div>
              </ContentCard>
              <ContentCard title="ByteTrack" accent="#ff9800">
                Associates every detection — including low-confidence ones. Two-stage matching recovers tracks lost by high-threshold methods.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  1st pass: high conf → tracks<br />
                  2nd pass: low conf → remaining tracks
                </div>
              </ContentCard>
            </div>

            <TheoryInline title="Hungarian Algorithm for Assignment" />
            <TheoryInline title="Appearance Features (DeepSORT)" />
          </div>
        </section>

        {/* ═══ Part 3: 3D Object Tracking ═══ */}
        <section id="tracking-3d">
          <SectionHeader
            icon={Box}
            title="3D Object Tracking"
            number={3}
            subtitle="Extend tracking to 3D using LiDAR or multi-camera systems. Track objects in metric world coordinates with 3D bounding boxes, velocities, and orientations — essential for autonomous driving."
          />
          <div className="space-y-4">
            <Tracking3DCanvas />

            <TheoryInline title="3D Multi-Object Tracking" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="3D Detection → 3D Tracking" accent="#e040fb">
                3D detectors (CenterPoint, PointPillars) produce 7-DoF boxes: (x, y, z, l, w, h, θ). Kalman filter with constant-velocity model predicts next state. Association via 3D IoU or center distance.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  IoU₃D = Vol(A∩B) / Vol(A∪B)
                </div>
              </ContentCard>
              <ContentCard title="Scene Flow: 3D Optical Flow" accent="#00bcd4">
                Per-point 3D motion vectors — the analog of optical flow but in 3D space. Captures both object motion and ego-motion. Self-supervised via Chamfer distance.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  SF(p) = (Δx, Δy, Δz) ∀ p ∈ point cloud
                </div>
              </ContentCard>
            </div>

            <TheoryInline title="Point Cloud Tracking" />

            <ContentCard title="Evaluation: nuScenes Tracking" accent="#ff9800">
              nuScenes tracking benchmark evaluates 3D MOT with AMOTA (Average Multi-Object Tracking Accuracy), combining detection quality and association accuracy. Methods must track across LiDAR sweeps at 2Hz with camera images at 12Hz.
            </ContentCard>
          </div>
        </section>

        {/* ═══ Part 4: Action Recognition ═══ */}
        <section id="action">
          <SectionHeader
            icon={Clapperboard}
            title="Video Action Recognition"
            number={4}
            subtitle="Classify actions in video by learning temporal dynamics. A 'jumping' action looks like 'standing' in any single frame — temporal reasoning is essential."
          />
          <div className="space-y-4">
            <ActionRecognitionCanvas />

            <TheoryInline title="Temporal Modeling Challenges" />
            <TheoryInline title="Two-Stream Networks" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="3D Convolutions (I3D)" accent="#e040fb">
                Extend 2D convolutions to time: 3×3×3 kernels learn spatiotemporal features. I3D inflates ImageNet-pretrained 2D weights by repeating along temporal dimension.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  W₃D(i,j,t) = (1/T)·W₂D(i,j) ∀t ∈ [1,T]
                </div>
              </ContentCard>
              <ContentCard title="SlowFast Networks" accent="#00bcd4">
                Two pathways: Slow (low frame rate, high channels → spatial semantics) and Fast (high frame rate, low channels → temporal dynamics). Lateral connections fuse both.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  Slow: 4 frames, 64 ch<br />
                  Fast: 32 frames, 8 ch (β=1/8)
                </div>
              </ContentCard>
            </div>

            <TheoryInline title="3D Convolutions (C3D, I3D)" />
            <TheoryInline title="Video Transformers" />
          </div>
        </section>

        {/* ═══ Part 5: Consolidated Review ═══ */}
        <section id="review">
          <SectionHeader
            icon={Layers}
            title="Algorithms, Papers & Practice"
            number={5}
            subtitle="Consolidated algorithms, key research papers, interactive playgrounds, and quizzes."
          />
          <ModulePage content={motionModule} hideHeader hideTheory />
        </section>
      </div>
    </div>
  );
}
