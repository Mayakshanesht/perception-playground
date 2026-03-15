import ModulePage from "@/components/ModulePage";
import { ModuleContent } from "@/data/moduleContent";
import { moduleContents } from "@/data/moduleContent";
import { MathEquation } from "@/components/MathBlock";
import AITutor from "@/components/AITutor";
import {
  FeatureMatchingCanvas,
  EpipolarGeometryCanvas,
  SfMPipelineCanvas,
  MVSCanvas,
  NeRFCanvas,
  GaussianSplattingCanvas,
} from "@/components/ReconstructionCanvasAnimations";
import { ArrowLeft, GraduationCap, Search, Layers, Box, Mountain, Sparkles, Blocks } from "lucide-react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

// ── Merge SfM + NeRF + add MVS into 3D reconstruction
const mvsTheory = [
  {
    title: "Multi-View Stereo (MVS)",
    content:
      "After SfM produces calibrated cameras and a sparse point cloud, Multi-View Stereo (MVS) computes dense geometry. MVS estimates depth maps from multiple calibrated views, performs multi-view consistency checks, and fuses per-view depth maps into a dense 3D representation (point cloud, mesh, or voxel grid). Methods range from traditional plane sweeping and PatchMatch to learned approaches like MVSNet.",
    equations: [
      {
        label: "Photo-Consistency Cost",
        tex: "C(\\mathbf{p}, d) = \\frac{1}{N-1} \\sum_{i=2}^{N} \\rho(I_1(\\mathbf{p}), I_i(\\mathcal{H}_i(\\mathbf{p}, d)))",
      },
      {
        label: "Depth Map Fusion",
        tex: "D_{\\text{fused}}(\\mathbf{p}) = \\text{median}\\{D_i(\\pi_i(\\mathbf{X}_\\mathbf{p}))\\}_{i=1}^{N}",
      },
    ],
  },
];

const featureMatchTheory = [
  {
    title: "Feature Detection & Matching (SIFT)",
    content:
      "SIFT detects scale-invariant keypoints by finding extrema in the Difference-of-Gaussians (DoG) scale space. Each keypoint is described by a 128-D gradient histogram vector (4×4 spatial bins × 8 orientations). Matching uses Lowe's ratio test: accept match only if distance to nearest neighbor is much smaller than to second-nearest, rejecting ~90% of false matches.",
    equations: [
      {
        label: "Scale Space",
        tex: "L(x,y,\\sigma) = G(x,y,\\sigma) * I(x,y)",
      },
      {
        label: "DoG Approximation",
        tex: "D(x,y,\\sigma) = L(x,y,k\\sigma) - L(x,y,\\sigma)",
      },
      {
        label: "Ratio Test",
        tex: "\\frac{\\|d_1 - d_2\\|}{\\|d_1 - d_3\\|} < \\tau, \\quad \\tau \\approx 0.75",
      },
    ],
  },
];

const epipolarTheory = [
  {
    title: "Epipolar Geometry",
    content:
      "The fundamental matrix F is a 3×3 rank-2 matrix (7 DOF) that encodes the epipolar constraint between two uncalibrated views. It maps a point x in one image to an epipolar line l′=Fx in the other, constraining where its match can lie. The essential matrix E = K′ᵀFK works in calibrated coordinates and encodes relative pose [R|t]. SVD decomposition of E yields 4 candidate (R,t) pairs, disambiguated by checking positive-depth triangulation.",
    equations: [
      {
        label: "Epipolar Constraint",
        tex: "\\mathbf{x}'^T \\mathbf{F} \\mathbf{x} = 0",
      },
      {
        label: "Essential Matrix",
        tex: "\\mathbf{E} = [\\mathbf{t}]_\\times \\mathbf{R} = \\mathbf{K}'^T \\mathbf{F} \\mathbf{K}",
      },
      {
        label: "RANSAC Iterations",
        tex: "N = \\frac{\\log(1-p)}{\\log(1-(1-\\varepsilon)^s)}",
      },
    ],
  },
];

const gaussianSplattingTheory = [
  {
    title: "3D Gaussian Splatting",
    content:
      "3DGS represents scenes as millions of 3D Gaussians, each parameterized by position μ ∈ ℝ³, rotation quaternion q ∈ ℝ⁴, scale s ∈ ℝ³, opacity α ∈ ℝ, and spherical harmonic coefficients for view-dependent color (degree 3 SH = 48 floats). Rendering is via differentiable tile-based rasterization: project Gaussians into 2D ellipses, sort by depth, alpha-composite front-to-back. Trains in ~30min and renders at 100+ FPS.",
    equations: [
      {
        label: "3D Gaussian",
        tex: "G(\\mathbf{x}) = \\exp\\left(-\\frac{1}{2}(\\mathbf{x}-\\boldsymbol{\\mu})^T \\boldsymbol{\\Sigma}^{-1} (\\mathbf{x}-\\boldsymbol{\\mu})\\right)",
      },
      {
        label: "Covariance Decomposition",
        tex: "\\boldsymbol{\\Sigma} = \\mathbf{R} \\mathbf{S} \\mathbf{S}^T \\mathbf{R}^T",
      },
      {
        label: "2D Projection",
        tex: "\\boldsymbol{\\Sigma}' = \\mathbf{J} \\mathbf{W} \\boldsymbol{\\Sigma} \\mathbf{W}^T \\mathbf{J}^T",
      },
      {
        label: "Alpha Compositing",
        tex: "C = \\sum_i c_i \\alpha_i \\prod_{j<i} (1 - \\alpha_j)",
      },
    ],
  },
];

const reconstructionModule: ModuleContent = {
  id: "reconstruction",
  title: "3D Reconstruction & Rendering",
  subtitle:
    "Reconstruct 3D scenes from images and render novel views — from classical Structure from Motion through Multi-View Stereo to neural radiance fields and Gaussian Splatting.",
  color: "340 75% 55%",
  theory: [
    ...featureMatchTheory,
    ...epipolarTheory,
    ...moduleContents.sfm.theory,
    ...mvsTheory,
    ...moduleContents.nerf.theory,
    ...gaussianSplattingTheory,
  ],
  algorithms: [
    ...moduleContents.sfm.algorithms,
    {
      name: "MVS Pipeline (COLMAP Dense)",
      steps: [
        { step: "Input from SfM", detail: "Calibrated cameras, sparse 3D points from incremental SfM" },
        { step: "Stereo Matching", detail: "PatchMatch stereo computes depth and normal maps for each view" },
        { step: "Consistency Filtering", detail: "Check geometric and photometric consistency across views" },
        { step: "Depth Map Fusion", detail: "Merge per-view depth maps into a unified dense point cloud" },
        { step: "Surface Reconstruction", detail: "Poisson or Delaunay meshing from dense point cloud" },
      ],
    },
    ...moduleContents.nerf.algorithms,
  ],
  papers: [
    ...moduleContents.sfm.papers,
    { year: 2004, title: "SIFT", authors: "Lowe", venue: "IJCV", summary: "Scale-Invariant Feature Transform with DoG keypoint detection and 128-D descriptor." },
    { year: 2010, title: "PMVS2", authors: "Furukawa & Ponce", venue: "TPAMI", summary: "Patch-based Multi-View Stereo for dense reconstruction with expansion and filtering." },
    { year: 2018, title: "MVSNet", authors: "Yao et al.", venue: "ECCV", summary: "Learned multi-view stereo with differentiable homography warping and 3D cost volume." },
    { year: 2023, title: "3D Gaussian Splatting", authors: "Kerbl et al.", venue: "SIGGRAPH", summary: "Real-time radiance field rendering with 3D Gaussians and differentiable rasterization." },
    ...moduleContents.nerf.papers,
  ].sort((a, b) => a.year - b.year),
};

const color = reconstructionModule.color;

const theoryByTitle: Record<string, (typeof reconstructionModule.theory)[0]> = {};
reconstructionModule.theory.forEach((s) => {
  theoryByTitle[s.title] = s;
});

function TheoryInline({ title }: { title: string }) {
  const section = theoryByTitle[title];
  if (!section) return null;
  return (
    <div className="concept-card">
      <div className="flex items-center flex-wrap gap-y-1 mb-3">
        <h3 className="font-semibold text-foreground text-sm">{section.title}</h3>
        <AITutor conceptTitle={section.title} conceptContent={section.content} moduleName="3D Reconstruction & Rendering" />
      </div>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">{section.content}</p>
      {section.equations?.map((eq) => (
        <div key={eq.label} className="mb-3">
          <MathEquation tex={eq.tex} label={eq.label} />
        </div>
      ))}
    </div>
  );
}

function ContentCard({ title, children, accent }: { title: string; children: React.ReactNode; accent?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card/50 p-4">
      <p className="text-[10px] font-semibold uppercase tracking-wider mb-2" style={{ color: accent || `hsl(${color})` }}>
        {title}
      </p>
      <div className="text-sm text-muted-foreground leading-relaxed">{children}</div>
    </div>
  );
}

function SectionHeader({ icon: Icon, title, number, subtitle }: { icon: any; title: string; number: number; subtitle?: string }) {
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="flex items-start gap-4 mb-6">
      <div className="h-10 w-10 rounded-xl flex items-center justify-center shrink-0 mt-1" style={{ backgroundColor: `hsl(${color} / 0.12)` }}>
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

export default function ReconstructionModule() {
  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      {/* Header */}
      <div className="flex items-start gap-4 mb-8">
        <div className="h-12 w-12 rounded-xl flex items-center justify-center shrink-0" style={{ backgroundColor: `hsl(${color} / 0.12)` }}>
          <GraduationCap className="h-6 w-6" style={{ color: `hsl(${color})` }} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground tracking-tight">{reconstructionModule.title}</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{reconstructionModule.subtitle}</p>
        </div>
      </div>

      {/* Learning flow nav */}
      <div className="rounded-xl border border-border bg-muted/30 p-4 mb-8">
        <h2 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3">Structured Learning Flow</h2>
        <div className="grid sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {[
            { id: "features", icon: "🔍", label: "Feature Matching" },
            { id: "epipolar", icon: "📐", label: "Epipolar Geometry" },
            { id: "sfm", icon: "📷", label: "SfM Pipeline" },
            { id: "mvs", icon: "🏔️", label: "Multi-View Stereo" },
            { id: "nerf", icon: "🌈", label: "NeRF" },
            { id: "gs", icon: "💎", label: "Gaussian Splatting" },
          ].map((item) => (
            <a key={item.id} href={`#${item.id}`} className="rounded-lg border border-border bg-card p-2.5 hover:border-primary/40 transition-colors text-center">
              <p className="text-sm mb-0.5">{item.icon}</p>
              <p className="text-xs text-foreground font-medium">{item.label}</p>
            </a>
          ))}
        </div>
      </div>

      <div className="space-y-12">
        {/* ═══════ Part 1: Feature Matching ═══════ */}
        <section id="features">
          <SectionHeader
            icon={Search}
            title="Feature Detection & Matching"
            number={1}
            subtitle="Find distinctive keypoints invariant to scale, rotation, and illumination. Match descriptor vectors across image pairs using Lowe's ratio test, then filter outliers with RANSAC."
          />
          <div className="space-y-4">
            <FeatureMatchingCanvas />
            <TheoryInline title="Feature Detection & Matching (SIFT)" />

            <div className="grid md:grid-cols-3 gap-4">
              <ContentCard title="Scale-Space Extrema" accent="hsl(var(--primary))">
                SIFT builds a Gaussian pyramid, computes DoG at each octave. Keypoints are 3D extrema in (x,y,σ) space — stable under scale change.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  DoG extremum: local max/min<br />across 26 neighbors in (x,y,σ) cube
                </div>
              </ContentCard>
              <ContentCard title="128-D Descriptor" accent="hsla(40, 95%, 50%, 1)">
                16×16 patch → 4×4 cells → 8-bin gradient orientation histogram each. Concatenated and L2-normalized → 128-D vector invariant to illumination.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  f ∈ ℝ¹²⁸, ‖f‖₂ = 1<br />clamped at 0.2, re-normalized
                </div>
              </ContentCard>
              <ContentCard title="Ratio Test + RANSAC" accent="hsla(270, 60%, 55%, 1)">
                Accept match only if nearest/second-nearest distance ratio &lt; 0.75. RANSAC iteratively estimates F from minimal samples, keeping best inlier set.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  N = log(1−p) / log(1−(1−ε)^s)<br />p=0.99, s=8 (8-point alg.)
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══════ Part 2: Epipolar Geometry ═══════ */}
        <section id="epipolar">
          <SectionHeader
            icon={Layers}
            title="Epipolar Geometry"
            number={2}
            subtitle="The fundamental matrix constrains where a point's match lies in the second image. Essential matrix adds calibration. Reduces stereo matching from 2D to 1D search."
          />
          <div className="space-y-4">
            <EpipolarGeometryCanvas />
            <TheoryInline title="Epipolar Geometry" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Fundamental vs Essential Matrix" accent="hsl(var(--primary))">
                <strong className="text-foreground">F</strong> (uncalibrated): 3×3 rank-2, 7 DOF. Maps point → epipolar line.<br /><br />
                <strong className="text-foreground">E = K′ᵀFK</strong> (calibrated): encodes [R|t]. SVD → 4 candidate poses, disambiguated by positive-depth check.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  E = [t]ₓ R, ‖t‖ = 1<br />4 solutions: (R₁,±t), (R₂,±t)
                </div>
              </ContentCard>
              <ContentCard title="Rectification" accent="hsl(160, 80%, 55%)">
                Apply homographies H, H′ to both images so epipolar lines become horizontal. After rectification, matching is a 1D scan along each row — used in all stereo algorithms.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  H, H′: epipoles → ∞<br />aligned rows ⟹ 1D disparity search
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══════ Part 3: SfM Pipeline ═══════ */}
        <section id="sfm">
          <SectionHeader
            icon={Box}
            title="Structure from Motion Pipeline"
            number={3}
            subtitle="Incrementally reconstruct cameras and 3D points from an unordered image collection. Bundle adjustment jointly refines all estimates to minimize reprojection error."
          />
          <div className="space-y-4">
            <SfMPipelineCanvas />

            {/* Pipeline steps */}
            <div className="flex gap-0 overflow-x-auto pb-2">
              {[
                { n: "01", label: "Feature Extract", sub: "SIFT 128-D" },
                { n: "02", label: "Match", sub: "vocab tree" },
                { n: "03", label: "RANSAC", sub: "F matrix" },
                { n: "04", label: "Init Two-View", sub: "E → R,t" },
                { n: "05", label: "PnP Register", sub: "new images" },
                { n: "06", label: "Triangulate", sub: "new 3D pts" },
                { n: "07", label: "Bundle Adjust", sub: "LM optimize" },
              ].map((step, i) => (
                <div
                  key={i}
                  className={`flex-1 min-w-[90px] border border-border p-2.5 relative ${
                    i === 0 ? "rounded-l-md" : i === 6 ? "rounded-r-md" : ""
                  }`}
                >
                  <p className="text-[9px] font-mono text-primary tracking-wider">{step.n}</p>
                  <p className="text-[10px] font-semibold text-foreground">{step.label}</p>
                  <p className="text-[8px] font-mono text-muted-foreground">{step.sub}</p>
                  {i < 6 && (
                    <span className="absolute right-[-8px] top-1/2 -translate-y-1/2 z-10 text-[8px] text-primary bg-background px-0.5">▶</span>
                  )}
                </div>
              ))}
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="DLT Triangulation" accent="hsl(var(--primary))">
                Given P₁, P₂ and 2D points x₁, x₂: set up AX=0 (each correspondence gives 2 equations). Solve via SVD: X = last column of V.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  A = [x₁×P₁ ; x₂×P₂]<br />X* = argmin ‖AX‖² s.t. ‖X‖=1
                </div>
              </ContentCard>
              <ContentCard title="Bundle Adjustment" accent="hsla(40, 95%, 50%, 1)">
                Joint optimization over all camera poses and 3D points. Minimize sum of squared reprojection errors using Levenberg-Marquardt with Schur complement trick.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  min Σᵢⱼ ‖xᵢⱼ − π(Rⱼ,tⱼ,Xᵢ)‖²<br />(JᵀJ+λI)δ = −Jᵀr
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══════ Part 4: MVS ═══════ */}
        <section id="mvs">
          <SectionHeader
            icon={Mountain}
            title="Multi-View Stereo (MVS)"
            number={4}
            subtitle="Dense reconstruction from calibrated cameras. Plane-sweep cost volumes, PatchMatch depth estimation, multi-view consistency, and depth map fusion."
          />
          <div className="space-y-4">
            <MVSCanvas />
            <TheoryInline title="Multi-View Stereo (MVS)" />

            <div className="grid md:grid-cols-3 gap-4">
              <ContentCard title="Plane Sweep" accent="hsl(var(--primary))">
                Sweep depth planes through scene. At each (pixel, depth) compute photo-consistency via homography warping. Build 3D cost volume C[x,y,d].
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  C[x,y,d]: H×W×D tensor<br />D = 64-256 depth hypotheses
                </div>
              </ContentCard>
              <ContentCard title="PatchMatch Stereo" accent="hsla(40, 95%, 50%, 1)">
                Random init → propagate good solutions to neighbors → random refinement. Converges in ~3 iterations to near-optimal depth at O(1/64) the cost of exhaustive search.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  Init: d ~ U(d_min, d_max)<br />Refine: d ± δ, δ halved each iter
                </div>
              </ContentCard>
              <ContentCard title="Depth Map Fusion" accent="hsla(270, 60%, 55%, 1)">
                Check cross-view consistency: reproject depth i into view j. If depth/pixel differs too much → reject. Fuse consistent estimates via weighted median.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  |dᵢ − dⱼ(π(Xᵢ))| / dᵢ &lt; 0.01<br />fuse: weighted median → dense cloud
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══════ Part 5: NeRF ═══════ */}
        <section id="nerf">
          <SectionHeader
            icon={Sparkles}
            title="Neural Radiance Fields (NeRF)"
            number={5}
            subtitle="Represent a scene as a continuous 5D function: position (x,y,z) + direction (θ,φ) → color + density. Render by integrating along rays. Train purely from posed images."
          />
          <div className="space-y-4">
            <NeRFCanvas />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Volume Rendering" accent="hsl(var(--primary))">
                Each ray sample contributes color weighted by its alpha (1−e^{`{−σδ}`}) and accumulated transmittance Tᵢ. Samples with high density and unoccluded path contribute most.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  Ĉ(r) = Σᵢ Tᵢ·(1−e^(−σᵢδᵢ))·cᵢ<br />Tᵢ = exp(−Σⱼ&lt;ᵢ σⱼδⱼ)
                </div>
              </ContentCard>
              <ContentCard title="Hierarchical Sampling" accent="hsla(40, 95%, 50%, 1)">
                Coarse network: 64 uniform samples. Fine network: 128 importance samples from coarse weight PDF. Concentrates evaluations near visible surfaces.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  Coarse: Nₒ=64 stratified<br />Fine: Nᶠ=128 importance
                </div>
              </ContentCard>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Positional Encoding" accent="hsla(270, 60%, 55%, 1)">
                MLPs have spectral bias toward low frequencies. PE maps p → 2L sinusoidals. L=10 for position (60-D), L=4 for direction (24-D). Enables sharp edges and fine textures.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  γ(p) = (sin(2⁰πp), cos(2⁰πp), …)<br />L=10 → 60-D input for (x,y,z)
                </div>
              </ContentCard>
              <ContentCard title="MLP Architecture" accent="hsl(160, 80%, 55%)">
                8 FC layers (256 units, ReLU) → density σ + feature. Feature + direction → 1 layer → RGB. Direction only affects color (view-dependent), not density (geometry).
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  F_Θ: (γ(x), γ(d)) → (σ, c)<br />σ ≥ 0 (geometry), c ∈ [0,1]³
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══════ Part 6: 3D Gaussian Splatting ═══════ */}
        <section id="gs">
          <SectionHeader
            icon={Blocks}
            title="3D Gaussian Splatting"
            number={6}
            subtitle="Represent scenes as millions of 3D Gaussians. Render via differentiable rasterization: project, sort by depth, alpha-composite. Real-time at 100+ FPS."
          />
          <div className="space-y-4">
            <GaussianSplattingCanvas />
            <TheoryInline title="3D Gaussian Splatting" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Per-Gaussian Parameters" accent="hsl(var(--primary))">
                Position μ ∈ ℝ³, rotation q ∈ ℝ⁴, scale s ∈ ℝ³, opacity α ∈ ℝ, SH coefficients (degree 3 = 48 floats). Total ~59 floats per Gaussian. 3M Gaussians → ~700MB GPU.
              </ContentCard>
              <ContentCard title="Tile-Based Rasterization" accent="hsla(40, 95%, 50%, 1)">
                Sort by depth. Divide image into 16×16 tiles. Per-tile: process only overlapping Gaussians. Parallel alpha compositing runs at 100+ FPS on modern GPUs.
              </ContentCard>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Adaptive Density Control" accent="hsl(160, 80%, 55%)">
                Start from SfM sparse points. Clone small Gaussians with large ∇μ (fills gaps). Split large ones (over-reconstructed). Prune low-opacity every 3k iterations.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  clone if: ‖∇μ‖ &gt; τ AND small<br />split if: ‖∇μ‖ &gt; τ AND large<br />prune if: α &lt; ε_α
                </div>
              </ContentCard>
              <ContentCard title="vs NeRF: Speed & Quality" accent="hsla(270, 60%, 55%, 1)">
                3DGS: explicit, editable, real-time (100+ FPS), trains ~30min. NeRF: implicit, smooth, slow (~30s/frame), hours to train. Both achieve ~26-27 dB PSNR on standard benchmarks.
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══════ Part 7: Consolidated Review ═══════ */}
        <section id="review">
          <SectionHeader
            icon={GraduationCap}
            title="Algorithms, Papers & Practice"
            number={7}
            subtitle="Consolidated algorithms, key research papers, and interactive quizzes."
          />
          <ModulePage content={reconstructionModule} hideHeader hideTheory />
        </section>
      </div>
    </div>
  );
}
