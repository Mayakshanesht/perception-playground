import ModulePage from "@/components/ModulePage";
import { ModuleContent } from "@/data/moduleContent";
import { moduleContents } from "@/data/moduleContent";
import { MathEquation } from "@/components/MathBlock";
import { ClassificationScene, DetectionScene, SemanticSegScene, InstanceSegScene } from "@/components/SemanticAnimations";
import { CNNArchitectureCanvas, ConvFilterCanvas, DetectionPipelineCanvas, SegmentationArchCanvas, ViTCanvas } from "@/components/SemanticCanvasAnimations";
import { ArrowLeft, GraduationCap, Lightbulb, Target, Grid3X3, Layers, Puzzle, Brain, Zap } from "lucide-react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

const semanticModule: ModuleContent = {
  id: "semantic",
  title: "Semantic Information",
  subtitle: "Extract meaning from images — classify scenes, detect objects, and segment regions at the pixel level.",
  color: "187 85% 53%",
  theory: [
    ...moduleContents.classification.theory,
    ...moduleContents.detection.theory,
    ...moduleContents.segmentation.theory,
  ],
  algorithms: [
    ...moduleContents.classification.algorithms,
    ...moduleContents.detection.algorithms,
    ...moduleContents.segmentation.algorithms,
  ],
  papers: [
    ...moduleContents.classification.papers,
    ...moduleContents.detection.papers,
    ...moduleContents.segmentation.papers,
  ].sort((a, b) => a.year - b.year),
  playgrounds: [
    ...(moduleContents.detection.playground ? [moduleContents.detection.playground] : []),
    ...(moduleContents.segmentation.playgrounds ?? []),
  ],
};

const color = semanticModule.color;

const theoryByTitle: Record<string, typeof semanticModule.theory[0]> = {};
semanticModule.theory.forEach(s => { theoryByTitle[s.title] = s; });

function TheoryInline({ title }: { title: string }) {
  const section = theoryByTitle[title];
  if (!section) return null;
  return (
    <div className="concept-card">
      <h3 className="font-semibold text-foreground mb-3 text-sm">{section.title}</h3>
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
      <p className="text-[10px] font-semibold uppercase tracking-wider mb-2" style={{ color: accent || "hsl(var(--primary))" }}>{title}</p>
      <div className="text-sm text-muted-foreground leading-relaxed">{children}</div>
    </div>
  );
}

function SectionHeader({ icon: Icon, title, number, subtitle }: { icon: any; title: string; number: number; subtitle?: string }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <div className="h-9 w-9 rounded-lg flex items-center justify-center shrink-0" style={{ backgroundColor: `hsl(${color} / 0.12)` }}>
        <Icon className="h-4 w-4" style={{ color: `hsl(${color})` }} />
      </div>
      <div>
        <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">Part {number}</p>
        <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">{title}</h2>
        {subtitle && <p className="text-[10px] text-muted-foreground mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}

export default function SemanticModule() {
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
          <h1 className="text-2xl font-bold text-foreground tracking-tight">{semanticModule.title}</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{semanticModule.subtitle}</p>
        </div>
      </div>

      <div className="space-y-10">

        {/* ═══ Part 1: The Three Core Tasks ═══ */}
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
          <SectionHeader icon={Lightbulb} title="The Three Core Tasks" number={1} subtitle="Classification → Detection → Segmentation: increasing spatial precision" />

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
            <ContentCard title="Classification" accent="#00d4ff">
              <p>Assign a single label to the entire image from K classes. Output: one softmax probability vector.</p>
              <pre className="mt-2 rounded bg-muted/40 border border-border border-l-2 border-l-[#00d4ff] p-2 text-[10px] font-mono text-[#00d4ff] whitespace-pre">{"P(y=k|x) = e^(wₖᵀx) / Σe^(wⱼᵀx)"}</pre>
            </ContentCard>
            <ContentCard title="Object Detection" accent="#ff6b35">
              <p>Predict class + bounding box for each object. Handles varying object counts with NMS post-processing.</p>
              <pre className="mt-2 rounded bg-muted/40 border border-border border-l-2 border-l-[#ff6b35] p-2 text-[10px] font-mono text-[#ff6b35] whitespace-pre">{"IoU(A,B) = |A∩B| / |A∪B|"}</pre>
            </ContentCard>
            <ContentCard title="Segmentation" accent="#7c3aed">
              <p>Assign a label to every pixel. Semantic (class-per-pixel), Instance (per-object), or Panoptic (both).</p>
              <pre className="mt-2 rounded bg-muted/40 border border-border border-l-2 border-l-[#7c3aed] p-2 text-[10px] font-mono text-[#7c3aed] whitespace-pre">{"L = -1/HW Σᵢⱼ Σc yᵢⱼc·log(ŷᵢⱼc)"}</pre>
            </ContentCard>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
            <ContentCard title="Softmax Classifier Intuition">
              <p>The softmax function converts raw logits into a probability distribution. Each weight vector wₖ defines a linear decision boundary in pixel space. The training signal (cross-entropy loss) pushes the correct class logit up and all others down.</p>
            </ContentCard>
            <ContentCard title="Why Pixel-Level Prediction Is Hard">
              <p>Classification: 1 label per image. Detection: ~100 boxes. Segmentation: H×W labels (e.g. 640×480 = 307,200 predictions). The model must balance global context (what) with local precision (where) — this tension drives all architecture choices.</p>
            </ContentCard>
          </div>
        </motion.section>

        {/* ═══ Part 2: Image Classification ═══ */}
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05, duration: 0.4 }}>
          <SectionHeader icon={Lightbulb} title="Image Classification" number={2} subtitle="Assign a single label to an entire image — f(x) → class label" />

          <div className="space-y-4 mb-6">
            <TheoryInline title="What is Image Classification?" />
          </div>

          <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-2">🔍 Interactive · Classification Visualization</p>
          <div className="mb-6">
            <ClassificationScene />
          </div>

          <div className="space-y-4 mb-6">
            <TheoryInline title="Convolutional Neural Networks (CNNs)" />
          </div>

          {/* CNN 3D Architecture Animation */}
          <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-2">🏗️ Interactive · CNN Architecture — 3D Walkthrough</p>
          <div className="mb-6">
            <CNNArchitectureCanvas />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
            <ContentCard title="Convolution" accent="#00d4ff">
              <p>Kernel W ∈ ℝ^(k×k×Cin×Cout) slides over input. Weight sharing: same filter everywhere → fewer params.</p>
              <pre className="mt-2 rounded bg-muted/40 border border-border border-l-2 border-l-[#00d4ff] p-2 text-[10px] font-mono text-[#00d4ff] whitespace-pre">{"(f*g)(i,j) = Σₘ Σₙ f(m,n)·g(i-m,j-n)\nOutput: O = (W-K+2P)/S + 1"}</pre>
            </ContentCard>
            <ContentCard title="ReLU & Batch Norm" accent="#ff6b35">
              <p>ReLU kills negative activations, preventing vanishing gradients. BatchNorm normalizes per mini-batch with learned γ, β rescaling.</p>
              <pre className="mt-2 rounded bg-muted/40 border border-border border-l-2 border-l-[#ff6b35] p-2 text-[10px] font-mono text-[#ff6b35] whitespace-pre">{"ReLU(x) = max(0, x)\nx̂ᵢ = (xᵢ-μB)/√(σ²B+ε), yᵢ = γx̂ᵢ + β"}</pre>
            </ContentCard>
          </div>

          {/* Convolution Filter Sweep */}
          <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-2">🔬 Interactive · Convolution Filter Sweep</p>
          <div className="mb-6">
            <ConvFilterCanvas />
          </div>

          <div className="space-y-4">
            <TheoryInline title="Batch Normalization" />
            <TheoryInline title="Residual Learning (ResNet)" />
            <TheoryInline title="Vision Transformers (ViT)" />
          </div>
        </motion.section>

        {/* ═══ Part 3: Object Detection ═══ */}
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1, duration: 0.4 }}>
          <SectionHeader icon={Target} title="Object Detection" number={3} subtitle="Localize and classify multiple objects — f(x) → {class, bbox, conf}" />

          <div className="space-y-4 mb-6">
            <TheoryInline title="What is Object Detection?" />
          </div>

          <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-2">🎯 Interactive · Detection Visualization</p>
          <div className="mb-6">
            <DetectionScene />
          </div>

          {/* Detection Pipeline Canvas */}
          <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-2">⚙️ Interactive · Detection Pipeline Comparison</p>
          <div className="mb-6">
            <DetectionPipelineCanvas />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
            <ContentCard title="Focal Loss (RetinaNet)" accent="#7c3aed">
              <p>One-stage detectors fail due to extreme class imbalance (thousands of background anchors vs. few objects). Focal loss down-weights easy negatives by factor (1-pₜ)^γ.</p>
              <pre className="mt-2 rounded bg-muted/40 border border-border border-l-2 border-l-[#7c3aed] p-2 text-[10px] font-mono text-[#7c3aed] whitespace-pre">{"FL(pₜ) = -αₜ(1-pₜ)^γ · log(pₜ)"}</pre>
              <p className="mt-1 text-[11px]">γ=2 → easy negatives contribute 100× less to loss than hard positives.</p>
            </ContentCard>
            <ContentCard title="Feature Pyramid Network" accent="#10b981">
              <p>Top-down pathway with lateral connections. Deep features have semantics; shallow features have spatial precision. Each level detects at one scale range.</p>
              <pre className="mt-2 rounded bg-muted/40 border border-border border-l-2 border-l-[#10b981] p-2 text-[10px] font-mono text-[#10b981] whitespace-pre">{"Pₗ = Conv₁ₓ₁(Cₗ) + Upsample(Pₗ₊₁)"}</pre>
            </ContentCard>
          </div>

          <div className="space-y-4">
            <TheoryInline title="Two-Stage Detectors: R-CNN Family" />
            <TheoryInline title="One-Stage Detectors: YOLO & SSD" />
            <TheoryInline title="Non-Maximum Suppression (NMS)" />
            <TheoryInline title="Feature Pyramid Networks (FPN)" />
          </div>
        </motion.section>

        {/* ═══ Part 4: Semantic Segmentation ═══ */}
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15, duration: 0.4 }}>
          <SectionHeader icon={Grid3X3} title="Semantic Segmentation" number={4} subtitle="Pixel-level class labeling — f(x) → pixel-class map (H×W)" />

          <div className="space-y-4 mb-6">
            <TheoryInline title="Types of Segmentation" />
          </div>

          <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-2">🎨 Interactive · Semantic Segmentation Visualization</p>
          <div className="mb-6">
            <SemanticSegScene />
          </div>

          {/* Segmentation Architecture Canvas */}
          <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-2">🏗️ Interactive · Segmentation Architecture Comparison</p>
          <div className="mb-6">
            <SegmentationArchCanvas />
          </div>

          <div className="space-y-4">
            <TheoryInline title="Fully Convolutional Networks (FCN)" />
            <TheoryInline title="U-Net Architecture" />
            <TheoryInline title="Atrous/Dilated Convolutions (DeepLab)" />
          </div>
        </motion.section>

        {/* ═══ Part 5: Instance Segmentation ═══ */}
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2, duration: 0.4 }}>
          <SectionHeader icon={Puzzle} title="Instance Segmentation" number={5} subtitle="Separate individual objects at pixel level — f(x) → {class, mask, id} × N" />

          <div className="space-y-4 mb-6">
            <TheoryInline title="Instance Segmentation: Mask R-CNN" />
          </div>

          <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-2">🧩 Interactive · Instance Segmentation Visualization</p>
          <div className="mb-6">
            <InstanceSegScene />
          </div>
        </motion.section>

        {/* ═══ Part 6: Vision Transformers ═══ */}
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25, duration: 0.4 }}>
          <SectionHeader icon={Brain} title="Vision Transformers & Self-Attention" number={6} subtitle="Pure attention replaces convolutions — patches as tokens" />

          {/* ViT Canvas */}
          <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-2">🧠 Interactive · Vision Transformer Visualization</p>
          <div className="mb-6">
            <ViTCanvas />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
            <ContentCard title="Patch Embedding" accent="#00d4ff">
              <p>Image H×W×C split into N patches of size P×P. Each flattened patch projected to D dimensions via linear layer E. [CLS] token prepended; 1D positional embedding added.</p>
              <pre className="mt-2 rounded bg-muted/40 border border-border border-l-2 border-l-[#00d4ff] p-2 text-[10px] font-mono text-[#00d4ff] whitespace-pre">{"z₀ = [x_cls; x_p¹E; …; x_pᴺE] + E_pos\nN = HW/P² (224² with P=16: N=196)"}</pre>
            </ContentCard>
            <ContentCard title="Scaled Dot-Product Attention" accent="#ff6b35">
              <p>Q, K, V matrices via learned projections. Scores scale by √dₖ to prevent vanishing softmax gradients.</p>
              <pre className="mt-2 rounded bg-muted/40 border border-border border-l-2 border-l-[#ff6b35] p-2 text-[10px] font-mono text-[#ff6b35] whitespace-pre">{"Attn(Q,K,V) = softmax(QKᵀ/√dₖ)·V\nComplexity: O(N²·d) per head"}</pre>
            </ContentCard>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <ContentCard title="Multi-Head Attention" accent="#7c3aed">
              <p>Run h attention heads in parallel with different learned projections. Concatenate and project. Different heads attend to different spatial relationships.</p>
              <pre className="mt-2 rounded bg-muted/40 border border-border border-l-2 border-l-[#7c3aed] p-2 text-[10px] font-mono text-[#7c3aed] whitespace-pre">{"MHA(Q,K,V) = Concat(head₁,…,headₕ)·Wᴼ"}</pre>
            </ContentCard>
            <ContentCard title="CNN vs ViT Inductive Bias" accent="#10b981">
              <p><strong className="text-[#00d4ff]">CNN:</strong> Built-in translation equivariance, local receptive fields. Strong inductive bias → works with less data.</p>
              <p className="mt-1"><strong className="text-[#10b981]">ViT:</strong> No spatial inductive bias. Learns all relationships from data. Needs large datasets (JFT-300M) but scales better at large compute.</p>
            </ContentCard>
          </div>
        </motion.section>

        {/* ═══ Part 7: Algorithms, Papers & Practice ═══ */}
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3, duration: 0.4 }}>
          <SectionHeader icon={Layers} title="Algorithms, Papers & Practice" number={7} subtitle="Pipelines, key papers, playgrounds, and quizzes" />
          <ModulePage content={semanticModule} hideHeader hideTheory />
        </motion.section>
      </div>
    </div>
  );
}
