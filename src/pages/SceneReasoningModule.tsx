import ModulePage from "@/components/ModulePage";
import { ModuleContent } from "@/data/moduleContent";
import { sceneReasoningModule as baseSceneModule } from "@/data/consolidatedModules";
import { MathEquation } from "@/components/MathBlock";
import AITutor from "@/components/AITutor";
import { CLIPEmbeddingCanvas, VLMArchitectureCanvas, Scene3DCanvas } from "@/components/SceneReasoningCanvasAnimations";
import { ViTPatchDemo, CLIPSimilarityDemo, VLMArchitectureTabs, NeRFRayCastingDemo } from "@/components/VLMCanvasAnimations";
import { ArrowLeft, GraduationCap, Sparkles, Brain, Eye, Box, Layers } from "lucide-react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

// Extend with 3D scene understanding + VLM content from Part 2
const sceneReasoningModule: ModuleContent = {
  ...baseSceneModule,
  title: "Scene Reasoning & Vision-Language Models",
  subtitle: "From ViT patch embeddings and CLIP contrastive learning to multimodal LLMs (LLaVA, Flamingo), visual grounding, NeRF, and 3D scene understanding.",
  theory: [
    ...baseSceneModule.theory,
    {
      title: "Vision Transformer (ViT)",
      content: "ViT treats images as sequences of patches, just like text tokens. An image is divided into fixed-size patches (e.g., 16×16), each flattened and linearly projected to a d-dimensional embedding. A [CLS] token aggregates global information. Positional embeddings add spatial awareness. The same Transformer encoder architecture used in NLP processes these visual tokens identically. ViT-B/16: 12 layers, 768d, 12 heads, 86M params. Key insight: no spatial inductive bias — ViT needs large-scale pretraining (JFT-300M+) to match CNNs, but DeiT enables data-efficient training via knowledge distillation.",
      equations: [
        {
          label: "ViT Patch Embedding",
          tex: "z_0 = [x_{\\text{class}}; x_p^1 E; \\ldots; x_p^N E] + E_{\\text{pos}}, \\quad N = HW/P^2",
        },
      ],
    },
    {
      title: "NeRF — Neural Radiance Fields",
      content: "NeRF learns a continuous 5D function F_θ mapping 3D position (x,y,z) and viewing direction (θ,φ) to color (RGB) and volume density σ. Novel views are synthesized via differentiable volume rendering: accumulate color along each camera ray by integrating density-weighted radiance. Positional encoding with sinusoidal functions at multiple frequencies enables the MLP to represent high-frequency detail. 3D Gaussian Splatting achieves 100× faster rendering by representing the scene as learnable 3D Gaussians with alpha-compositing.",
      equations: [
        {
          label: "Volume Rendering",
          tex: "\\hat{C}(r) = \\sum_i T_i (1-e^{-\\sigma_i \\delta_i}) c_i, \\quad T_i = \\prod_{j<i} e^{-\\sigma_j \\delta_j}",
          variables: [
            { symbol: "σᵢ", meaning: "volume density at sample i" },
            { symbol: "δᵢ", meaning: "distance between adjacent samples" },
            { symbol: "cᵢ", meaning: "color at sample i" },
            { symbol: "Tᵢ", meaning: "accumulated transmittance" },
          ],
        },
      ],
    },
    {
      title: "3D Scene Graphs",
      content: "3D scene graphs represent environments as structured graphs where nodes are objects with 3D locations, attributes (color, size, material), and affordances, while edges encode spatial relationships (on-top-of, next-to, inside). Methods like 3DSSG and SceneGraphFusion build scene graphs from RGB-D scans or point clouds, combining 3D object detection with relationship prediction.",
      equations: [{ label: "Scene Graph", tex: "\\mathcal{G} = (\\mathcal{V}, \\mathcal{E}), \\quad v_i = (c_i, \\mathbf{b}_i^{3D}, \\mathbf{a}_i), \\quad e_{ij} = r_{ij}" }],
    },
    {
      title: "3D Visual Question Answering",
      content: "3D VQA answers questions about 3D scenes by combining point cloud understanding with language reasoning. Given a 3D point cloud and a natural language question, the model must localize relevant objects, understand spatial relationships, and generate an answer.",
      equations: [{ label: "3D VQA Objective", tex: "\\hat{a} = \\arg\\max_a P(a | \\mathcal{P}_{3D}, q)" }],
    },
    {
      title: "Embodied AI & Vision-Language Navigation",
      content: "Embodied AI agents perceive 3D environments through egocentric vision and act based on language instructions. The agent must ground language in visual observations, maintain spatial memory, and plan paths. RT-2 and PaLM-E demonstrate that large VLMs can directly output robot actions.",
      equations: [
        {
          label: "Navigation Policy",
          tex: "a_t = \\pi(o_t, h_{t-1}, \\text{instruction})",
          variables: [
            { symbol: "aₜ", meaning: "action at time t (move, turn, stop)" },
            { symbol: "oₜ", meaning: "visual observation (egocentric RGB-D)" },
            { symbol: "h_{t-1}", meaning: "hidden state encoding exploration history" },
          ],
        },
      ],
    },
  ],
  papers: [
    ...baseSceneModule.papers,
    { year: 2020, title: "ViT", authors: "Dosovitskiy et al.", venue: "ICLR", summary: "Vision Transformer: images as patch sequences, outperforms CNNs at scale." },
    { year: 2020, title: "NeRF", authors: "Mildenhall et al.", venue: "ECCV", summary: "Neural Radiance Fields for novel view synthesis via differentiable volume rendering." },
    { year: 2021, title: "3DSSG", authors: "Wald et al.", venue: "CVPR", summary: "3D semantic scene graphs from point clouds." },
    { year: 2022, title: "ScanQA", authors: "Azuma et al.", venue: "CVPR", summary: "3D question answering grounded in 3D scans." },
    { year: 2023, title: "3D-LLM", authors: "Hong et al.", venue: "NeurIPS", summary: "LLM that processes 3D point clouds for captioning, QA, and planning." },
    { year: 2023, title: "3DGS", authors: "Kerbl et al.", venue: "SIGGRAPH", summary: "3D Gaussian Splatting: 100× faster than NeRF." },
  ].sort((a, b) => a.year - b.year),
  algorithms: [
    ...baseSceneModule.algorithms,
    {
      name: "3D Scene Understanding Pipeline",
      steps: [
        { step: "3D Perception", detail: "Build 3D point cloud from RGB-D or LiDAR; detect 3D objects" },
        { step: "Scene Graph Construction", detail: "Predict pairwise spatial relationships between detected objects" },
        { step: "Language Grounding", detail: "Map natural language query to relevant nodes/edges" },
        { step: "Spatial Reasoning", detail: "Traverse graph to answer spatial questions or plan navigation" },
        { step: "Response Generation", detail: "Generate natural language response or action sequence" },
      ],
    },
  ],
};

const color = sceneReasoningModule.color;

const theoryByTitle: Record<string, typeof sceneReasoningModule.theory[0]> = {};
sceneReasoningModule.theory.forEach(s => { theoryByTitle[s.title] = s; });

function TheoryInline({ title }: { title: string }) {
  const section = theoryByTitle[title];
  if (!section) return null;
  return (
    <div className="concept-card">
      <div className="flex items-center flex-wrap gap-y-1 mb-3">
        <h3 className="font-semibold text-foreground text-sm">{section.title}</h3>
        <AITutor conceptTitle={section.title} conceptContent={section.content} moduleName="Scene Reasoning" />
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

export default function SceneReasoningModule() {
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
          <h1 className="text-2xl font-bold text-foreground tracking-tight">{sceneReasoningModule.title}</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{sceneReasoningModule.subtitle}</p>
        </div>
      </div>

      {/* Learning flow nav */}
      <div className="rounded-xl border border-border bg-muted/30 p-4 mb-8">
        <h2 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3">Structured Learning Flow</h2>
        <div className="grid sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {[
            { id: "vit", icon: "🔲", label: "Vision Transformer" },
            { id: "clip", icon: "🔗", label: "CLIP & VL Pre-training" },
            { id: "vlm", icon: "🧠", label: "Large VLMs" },
            { id: "grounding", icon: "🎯", label: "Visual Grounding" },
            { id: "3d-scene", icon: "📦", label: "3D Scene & NeRF" },
            { id: "review", icon: "📚", label: "Papers & Practice" },
          ].map((item) => (
            <a key={item.id} href={`#${item.id}`} className="rounded-lg border border-border bg-card p-2.5 hover:border-primary/40 transition-colors text-center">
              <p className="text-sm mb-0.5">{item.icon}</p>
              <p className="text-[10px] text-foreground font-medium">{item.label}</p>
            </a>
          ))}
        </div>
      </div>

      <div className="space-y-12">

        {/* ═══ Part 1: Vision Transformer ═══ */}
        <section id="vit">
          <SectionHeader icon={Eye} title="Vision Transformer (ViT)" number={1} subtitle="Treat image patches as tokens — the bridge between NLP transformers and computer vision." />
          <div className="space-y-4">
            <ViTPatchDemo />
            <TheoryInline title="Vision Transformer (ViT)" />
            <div className="grid md:grid-cols-3 gap-4">
              <ContentCard title="ViT Configurations" accent="#14b8a6">
                <div className="font-mono text-xs leading-relaxed">ViT-B/16: 12L · 768d · 12H · 86M<br />ViT-L/16: 24L · 1024d · 16H · 307M<br />ViT-H/14: 32L · 1280d · 16H · 632M</div>
                <p className="mt-1 text-xs">/16, /14 = patch size in pixels.</p>
              </ContentCard>
              <ContentCard title="ViT vs CNN" accent="#3b82f6">
                <div className="font-mono text-xs leading-relaxed">CNN: translation equivariance built-in<br />ViT: no spatial inductive bias<br />ViT needs JFT-300M+ to outperform CNN</div>
                <p className="mt-1 text-xs">DeiT enables data-efficient ViT via knowledge distillation.</p>
              </ContentCard>
              <ContentCard title="Classification" accent="#f59e0b">
                <div className="font-mono text-xs leading-relaxed">y = LN(z_L^0) ← [CLS] token only<br />p(y|x) = softmax(y · W_head)</div>
                <p className="mt-1 text-xs">Only the [CLS] output is used for classification.</p>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 2: CLIP ═══ */}
        <section id="clip">
          <SectionHeader icon={Sparkles} title="CLIP & Vision-Language Pre-training" number={2} subtitle="Learn aligned image-text representations via contrastive learning on 400M pairs." />
          <div className="space-y-4">
            <CLIPSimilarityDemo />
            <CLIPEmbeddingCanvas />
            <TheoryInline title="Intuition" />
            <TheoryInline title="Vision-Language Pre-training" />
            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Contrastive Objective (InfoNCE)" accent="#a855f7">
                For a batch of N image-text pairs, maximize cosine similarity for N matching pairs while minimizing for N²-N non-matching. Temperature τ controls sharpness.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  L = -½·(Σᵢ log[exp(sᵢᵢ/τ)/Σⱼexp(sᵢⱼ/τ)] + Σᵢ log[exp(sᵢᵢ/τ)/Σⱼexp(sⱼᵢ/τ)])
                </div>
              </ContentCard>
              <ContentCard title="Zero-Shot Classification" accent="#06b6d4">
                Compare image embedding against text embeddings of class descriptions — no task-specific training needed.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  ŷ = argmax_k sim(f_img(x), f_text("a photo of k"))<br />
                  Prompt engineering: ensemble 80 prompts → +3.5% ImageNet
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 3: VLMs ═══ */}
        <section id="vlm">
          <SectionHeader icon={Brain} title="Large Vision-Language Models" number={3} subtitle="Combine frozen/fine-tuned vision encoders with LLMs — GPT-4V, Gemini, LLaVA, Flamingo enable multi-turn dialogue about images." />
          <div className="space-y-4">
            <VLMArchitectureTabs />
            <VLMArchitectureCanvas />
            <TheoryInline title="Large Vision-Language Models (LVLMs)" />
            <TheoryInline title="Florence-2: Unified Vision Foundation Model" />
            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="LLaVA: Simple & Effective" accent="#a855f7">
                CLIP ViT-L → 576 visual tokens → MLP projector → LLaMA processes visual + text jointly.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  zᵥ = Wₚ · ViT(I) + bₚ, h = LLM([zᵥ ; zₜ])
                </div>
              </ContentCard>
              <ContentCard title="Florence-2: One Model, Many Tasks" accent="#06b6d4">
                Single seq2seq for detection, segmentation, captioning, grounding, OCR — task via text prompt.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  {"<OD>"} → "car [120,80,340,220]"<br />
                  {"<CAPTION>"} → "A red car on a highway"
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 4: Visual Grounding ═══ */}
        <section id="grounding">
          <SectionHeader icon={Eye} title="Visual Grounding & Referring Expressions" number={4} subtitle="Localize objects described by natural language — requires spatial reasoning, attributes, and visual context." />
          <div className="space-y-4">
            <TheoryInline title="Visual Grounding & Referring Expressions" />
            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Grounding Approaches" accent="#a855f7">
                <strong className="text-foreground">Two-stage:</strong> Generate proposals → rank by text similarity (MDETR).<br /><br />
                <strong className="text-foreground">One-stage:</strong> Predict box directly conditioned on text (GLIP, Florence-2).
              </ContentCard>
              <ContentCard title="Open-Vocabulary Detection" accent="#06b6d4">
                Combine CLIP features with detection heads to detect any text-described object. OWL-ViT and Grounding DINO lead this direction.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  b̂ = argmax P(b | I, "red car behind tree")
                </div>
              </ContentCard>
            </div>
            <TheoryInline title="Real-World Applications" />
          </div>
        </section>

        {/* ═══ Part 5: 3D Scene & NeRF ═══ */}
        <section id="3d-scene">
          <SectionHeader icon={Box} title="3D Scene Understanding & NeRF" number={5} subtitle="Neural Radiance Fields, 3D scene graphs, and embodied agents that navigate and interact with the physical world." />
          <div className="space-y-4">
            <NeRFRayCastingDemo />
            <Scene3DCanvas />
            <TheoryInline title="NeRF — Neural Radiance Fields" />
            <TheoryInline title="3D Scene Graphs" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="3D Scene Graphs" accent="#a855f7">
                Nodes = objects with 3D bbox + attributes. Edges = spatial relationships. Built from RGB-D scans via 3D detection + GNN.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  G = (V, E): vᵢ = (class, bbox₃D, attrs), eᵢⱼ = spatial_relation
                </div>
              </ContentCard>
              <ContentCard title="3D Visual QA (ScanQA)" accent="#06b6d4">
                Answer spatial questions about 3D point cloud scenes — requires counting, relationship understanding beyond 2D VQA.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  â = argmax P(a | P₃D, question)
                </div>
              </ContentCard>
            </div>

            <TheoryInline title="3D Visual Question Answering" />
            <TheoryInline title="Embodied AI & Vision-Language Navigation" />

            <ContentCard title="Embodied AI: Observe → Plan → Act" accent="#f59e0b">
              VLMs as planning backbones: the agent observes 3D environments through egocentric vision, grounds instructions in visual observations, builds spatial memory, and plans paths. RT-2 and PaLM-E directly output robot actions.
            </ContentCard>
          </div>
        </section>

        {/* ═══ Part 6: Review ═══ */}
        <section id="review">
          <SectionHeader icon={Layers} title="Algorithms, Papers & Practice" number={6} subtitle="Consolidated algorithms, key papers, and quizzes." />
          <ModulePage content={sceneReasoningModule} hideHeader hideTheory />
        </section>
      </div>
    </div>
  );
}
