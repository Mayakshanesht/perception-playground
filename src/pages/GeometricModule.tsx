import ModulePage from "@/components/ModulePage";
import { ModuleContent } from "@/data/moduleContent";
import { moduleContents } from "@/data/moduleContent";
import { MathEquation } from "@/components/MathBlock";
import AITutor from "@/components/AITutor";
import { MonocularDepthCanvas, StereoVisionCanvas, Pose2DCanvas, Pose3DCanvas, SelfSupervisedDepthCanvas } from "@/components/GeometricCanvasAnimations";
import { ArrowLeft, GraduationCap, Eye, Scan, PersonStanding, Rotate3D, RefreshCw } from "lucide-react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useSectionObserver } from "@/hooks/useSectionObserver";
import { Progress } from "@/components/ui/progress";

// Merge depth + pose into geometric information
const geometricModule: ModuleContent = {
  id: "geometric",
  title: "Geometric Information",
  subtitle: "Recover 3D geometry from images — estimate depth, detect human poses, and understand spatial structure.",
  color: "32 95% 55%",
  theory: [
    ...moduleContents.depth.theory,
    ...moduleContents.pose.theory,
  ],
  algorithms: [
    ...moduleContents.depth.algorithms,
    ...moduleContents.pose.algorithms,
  ],
  papers: [
    ...moduleContents.depth.papers,
    ...moduleContents.pose.papers,
  ].sort((a, b) => a.year - b.year),
  playgrounds: [
    ...(moduleContents.depth.playground ? [moduleContents.depth.playground] : []),
    ...(moduleContents.pose.playground ? [moduleContents.pose.playground] : []),
  ],
};

const color = geometricModule.color;

const theoryByTitle: Record<string, typeof geometricModule.theory[0]> = {};
geometricModule.theory.forEach(s => { theoryByTitle[s.title] = s; });

function TheoryInline({ title }: { title: string }) {
  const section = theoryByTitle[title];
  if (!section) return null;
  return (
    <div className="concept-card">
      <div className="flex items-center flex-wrap gap-y-1 mb-3">
        <h3 className="font-semibold text-foreground text-sm">{section.title}</h3>
        <AITutor conceptTitle={section.title} conceptContent={section.content} moduleName="Geometric Information" />
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

export default function GeometricModule() {
  const progressPct = useSectionObserver("geometric", ['stereo', 'mono-depth', 'pose-2d', 'pose-3d', 'self-sup', 'review']);

  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto">
      {/* Back link */}
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      {/* Header */}
      <div className="flex items-start gap-4 mb-8">
        <div
          className="h-12 w-12 rounded-xl flex items-center justify-center shrink-0"
          style={{ backgroundColor: `hsl(${color} / 0.12)` }}
        >
          <GraduationCap className="h-6 w-6" style={{ color: `hsl(${color})` }} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground tracking-tight">{geometricModule.title}</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{geometricModule.subtitle}</p>
        </div>
      </div>

      {/* Learning flow nav */}
      <div className="rounded-xl border border-border bg-muted/30 p-4 mb-8">
        <h2 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3">Structured Learning Flow</h2>
        <div className="grid sm:grid-cols-3 lg:grid-cols-5 gap-2">
          {[
            { id: "stereo", icon: "👀", label: "Stereo Vision" },
            { id: "mono-depth", icon: "🔭", label: "Monocular Depth" },
            { id: "pose-2d", icon: "🦴", label: "2D Pose" },
            { id: "pose-3d", icon: "🧊", label: "3D Pose Lifting" },
            { id: "self-sup", icon: "🔄", label: "Self-Supervised" },
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
        {/* ═══════ Part 1: Stereo Vision ═══════ */}
        <section id="stereo">
          <SectionHeader
            icon={Scan}
            title="Stereo Vision & Disparity"
            number={1}
            subtitle="Two cameras, known baseline. Depth from triangulation: find the same point in both images, measure the pixel shift (disparity), recover metric depth via Z = fB/d."
          />
          <div className="space-y-4">
            <StereoVisionCanvas />

            <TheoryInline title="Stereo Vision & Disparity" />

            <div className="grid md:grid-cols-3 gap-4">
              <ContentCard title="Core Formula" accent="#e8b84b">
                Depth Z is inversely proportional to disparity d. Large disparity = nearby object. Near-zero disparity makes depth blow up — the stereo horizon.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  Z = f · B / d<br />d = xL − xR
                </div>
              </ContentCard>
              <ContentCard title="Epipolar Constraint" accent="#2dd4bf">
                A 3D point's image must lie on a known line in the other camera — the epipolar line. This reduces matching from 2D search to 1D search.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  p_R^T · F · p_L = 0
                </div>
              </ContentCard>
              <ContentCard title="Cost Volume" accent="#fb7185">
                For each pixel and disparity candidate, compute a matching cost. The 3D cost volume C[x,y,d] is regularized (SGM) or processed by a 3D CNN to find the best disparity.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  C[x,y,d] = cost(IL(x,y), IR(x-d,y))
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══════ Part 2: Monocular Depth ═══════ */}
        <section id="mono-depth">
          <SectionHeader
            icon={Eye}
            title="Monocular Depth Estimation"
            number={2}
            subtitle="From a single image, predict the depth of every pixel — an ill-posed problem solved by learning perceptual cues: occlusion, relative size, texture gradients, and perspective convergence."
          />
          <div className="space-y-4">
            <MonocularDepthCanvas />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Ill-Posed Problem" accent="#e8b84b">
                Infinite 3D scenes can project to the same 2D image (scale-depth ambiguity). A small near object looks identical to a large far object. Networks learn priors from training data, not pure geometry.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  f: ℝ^(H×W×3) → ℝ^(H×W) — one depth per pixel
                </div>
              </ContentCard>
              <ContentCard title="Architecture: DPT / MiDaS" accent="#38bdf8">
                Encoder (ResNet / ViT) captures global semantics. Decoder upsamples with skip connections to preserve edges. DPT uses ViT reassembly blocks for long-range receptive fields — crucial for depth consistency.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  ViT tokens → Reassemble → Fuse → Depth
                </div>
              </ContentCard>
            </div>

            <TheoryInline title="Monocular Depth Estimation" />
            <TheoryInline title="Depth with Transformers (DPT, MiDaS)" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Scale-Invariant Loss" accent="#2dd4bf">
                Standard MSE penalizes absolute error — unfair since depth ranges vary wildly. The scale-invariant loss (Eigen et al.) measures relative structure, ignoring global scale offset.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  dᵢ = log ŷᵢ − log yᵢ<br />
                  L_SI = (1/n)Σdᵢ² − (λ/n²)(Σdᵢ)²
                </div>
              </ContentCard>
              <ContentCard title="BerHu Loss" accent="#fb7185">
                L1 for small errors (robust to outliers), L2 for large errors (stronger gradient). The adaptive threshold c = 0.2 × max(|error|). Widely used in indoor depth benchmarks.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  B(x) = |x| if |x| ≤ c<br />
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= (x²+c²)/2c if |x| &gt; c
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══════ Part 3: 2D Pose ═══════ */}
        <section id="pose-2d">
          <SectionHeader
            icon={PersonStanding}
            title="2D Pose Estimation"
            number={3}
            subtitle="Detect 17 anatomical keypoints (joints) in pixel coordinates. Two paradigms: top-down (detect person then joints) and bottom-up (detect all joints then group)."
          />
          <div className="space-y-4">
            <Pose2DCanvas />

            <TheoryInline title="2D Pose Estimation" />
            <TheoryInline title="High-Resolution Networks (HRNet)" />
            <TheoryInline title="Part Affinity Fields (OpenPose)" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Heatmap Prediction" accent="#e8b84b">
                For each of K keypoints, predict a Gaussian heatmap peaked at the joint location. Soft localization is more robust than direct coordinate regression.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  Ĥk(x,y) = exp(−((x−xk)²+(y−yk)²) / 2σ²)
                </div>
              </ContentCard>
              <ContentCard title="Top-Down vs Bottom-Up" accent="#2dd4bf">
                <strong className="text-foreground">Top-down:</strong> Detect persons → crop → estimate joints per person. Cost linear in person count.<br /><br />
                <strong className="text-foreground">Bottom-up:</strong> Detect all joints globally → group via PAFs. Fixed cost regardless of crowd size.
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══════ Part 4: 3D Pose Lifting ═══════ */}
        <section id="pose-3d">
          <SectionHeader
            icon={Rotate3D}
            title="3D Pose Lifting"
            number={4}
            subtitle="From 2D pixel coordinates, recover the full 3D skeleton in camera space. The lifting approach chains a 2D detector with a learned 2D→3D mapping."
          />
          <div className="space-y-4">
            <Pose3DCanvas />

            <TheoryInline title="3D Pose Estimation (Lifting)" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="MPJPE Metric" accent="#2dd4bf">
                Mean Per Joint Position Error — average Euclidean distance (mm) between predicted and ground-truth 3D joints. PA-MPJPE aligns via Procrustes first for pure structural accuracy.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  MPJPE = (1/K) Σk ‖X̂k − Xk‖₂
                </div>
              </ContentCard>
              <ContentCard title="Depth Ambiguity" accent="#fb7185">
                Multiple 3D poses project to the same 2D skeleton (e.g., leg raised forward vs backward look identical from front). Networks resolve this via temporal context and bone-length priors.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  Projection: x2D = P · X3D (non-invertible)
                </div>
              </ContentCard>
            </div>

            <ContentCard title="Ground Truth: MoCap Systems" accent="#38bdf8">
              3D ground truth comes from Motion Capture — reflective markers on body, tracked by multiple IR cameras. Precise to sub-millimeter. Human3.6M (3.6M frames) and MPI-INF-3DHP are standard benchmarks.
            </ContentCard>
          </div>
        </section>

        {/* ═══════ Part 5: Self-Supervised Depth ═══════ */}
        <section id="self-sup">
          <SectionHeader
            icon={RefreshCw}
            title="Self-Supervised Depth"
            number={5}
            subtitle="Train depth estimation without any ground-truth depth labels. Use consecutive video frames: predict depth + ego-motion, synthesize novel views, minimize photometric error."
          />
          <div className="space-y-4">
            <SelfSupervisedDepthCanvas />

            <TheoryInline title="Self-Supervised Depth Learning" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="View Synthesis as Supervision" accent="#e8b84b">
                If depth and ego-motion are correct, we can project pixels from frame t to synthesize frame t±1. The photometric error is the loss. No depth labels needed — just unlabeled video.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  ps ~ K · T(t→s) · Dt(pt) · K⁻¹ · pt
                </div>
              </ContentCard>
              <ContentCard title="SSIM + L1 Loss" accent="#2dd4bf">
                Pure L1/L2 pixel loss is sensitive to lighting changes. SSIM captures structural similarity. Monodepth2 combines both with α=0.85 for SSIM, 0.15 for L1.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  Lp = α·(1−SSIM)/2 + (1−α)·‖Ia−Ib‖₁
                </div>
              </ContentCard>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Zero-Shot Depth: MiDaS / DPT" accent="#a78bfa">
                Train on 12 diverse datasets simultaneously with scale-and-shift invariant loss. The result: a model that generalizes to any image domain without fine-tuning.
              </ContentCard>
              <ContentCard title="Failure Mode: Moving Objects" accent="#fb7185">
                The photometric loss assumes a static scene. Moving objects (cars, people) violate this. Solutions: auto-mask stationary pixels, predict per-object motion, or use semantic segmentation to ignore movers.
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══════ Part 6: Consolidated Review ═══════ */}
        <section id="review">
          <SectionHeader
            icon={GraduationCap}
            title="Algorithms, Papers & Practice"
            number={6}
            subtitle="Consolidated algorithms, key research papers, interactive playgrounds, and quizzes."
          />
          <ModulePage content={geometricModule} hideHeader hideTheory />
        </section>
      </div>
    </div>
  );
}
