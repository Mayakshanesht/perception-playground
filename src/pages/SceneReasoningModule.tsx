import ModulePage from "@/components/ModulePage";
import { ModuleContent } from "@/data/moduleContent";
import { sceneReasoningModule as baseSceneModule } from "@/data/consolidatedModules";
import { MathEquation } from "@/components/MathBlock";
import AITutor from "@/components/AITutor";
import { CLIPEmbeddingCanvas, VLMArchitectureCanvas, Scene3DCanvas } from "@/components/SceneReasoningCanvasAnimations";
import { ArrowLeft, GraduationCap, Sparkles, Brain, Eye, Box, Layers } from "lucide-react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

// Extend with 3D scene understanding
const sceneReasoningModule: ModuleContent = {
  ...baseSceneModule,
  theory: [
    ...baseSceneModule.theory,
    {
      title: "3D Scene Graphs",
      content: "3D scene graphs represent environments as structured graphs where nodes are objects with 3D locations, attributes (color, size, material), and affordances, while edges encode spatial relationships (on-top-of, next-to, inside). Unlike 2D scene graphs from images, 3D scene graphs operate in metric space, enabling precise spatial reasoning. Methods like 3DSSG and SceneGraphFusion build scene graphs from RGB-D scans or point clouds, combining 3D object detection with relationship prediction.",
      equations: [
        {
          label: "Scene Graph",
          tex: "\\mathcal{G} = (\\mathcal{V}, \\mathcal{E}), \\quad v_i = (c_i, \\mathbf{b}_i^{3D}, \\mathbf{a}_i), \\quad e_{ij} = r_{ij}",
        },
      ],
    },
    {
      title: "3D Visual Question Answering",
      content: "3D VQA answers questions about 3D scenes by combining point cloud understanding with language reasoning. Given a 3D point cloud and a natural language question, the model must localize relevant objects, understand spatial relationships, and generate an answer. ScanQA and 3D-LLM extend VLMs to process 3D representations directly, enabling questions like 'How many chairs are near the table?' or 'What is to the left of the TV?'",
      equations: [
        {
          label: "3D VQA Objective",
          tex: "\\hat{a} = \\arg\\max_a P(a | \\mathcal{P}_{3D}, q)",
        },
      ],
    },
    {
      title: "Embodied AI & Vision-Language Navigation",
      content: "Embodied AI agents perceive 3D environments through egocentric vision and act based on language instructions. Vision-Language Navigation (VLN) requires an agent to follow natural language instructions in unseen 3D environments. The agent must ground language ('go past the kitchen') in visual observations, maintain a spatial memory of explored areas, and plan paths through unknown environments. Recent work uses VLMs as the planning backbone, combining visual perception with language-guided reasoning.",
      equations: [
        {
          label: "Navigation Policy",
          tex: "a_t = \\pi(o_t, h_{t-1}, \\text{instruction})",
          variables: [
            { symbol: "aₜ", meaning: "action at time t (move forward, turn left/right, stop)" },
            { symbol: "oₜ", meaning: "visual observation (egocentric RGB-D image)" },
            { symbol: "h_{t-1}", meaning: "hidden state encoding exploration history" },
          ],
        },
      ],
    },
  ],
  papers: [
    ...baseSceneModule.papers,
    { year: 2021, title: "3DSSG", authors: "Wald et al.", venue: "CVPR", summary: "3D semantic scene graphs from point clouds with spatial relationship prediction." },
    { year: 2022, title: "ScanQA", authors: "Azuma et al.", venue: "CVPR", summary: "3D question answering grounded in 3D scans — first large-scale 3D VQA benchmark." },
    { year: 2023, title: "3D-LLM", authors: "Hong et al.", venue: "NeurIPS", summary: "LLM that takes 3D point clouds as input for 3D captioning, QA, and planning." },
    { year: 2024, title: "LEO", authors: "Huang et al.", venue: "ICML", summary: "Embodied multi-modal agent combining 3D perception with language reasoning for navigation." },
  ].sort((a, b) => a.year - b.year),
  algorithms: [
    ...baseSceneModule.algorithms,
    {
      name: "3D Scene Understanding Pipeline",
      steps: [
        { step: "3D Perception", detail: "Build 3D point cloud from RGB-D or LiDAR; detect 3D objects and segment surfaces" },
        { step: "Scene Graph Construction", detail: "Predict pairwise spatial relationships between detected objects" },
        { step: "Language Grounding", detail: "Map natural language query to relevant nodes/edges in the scene graph" },
        { step: "Spatial Reasoning", detail: "Traverse graph to answer spatial questions or plan navigation paths" },
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
        <div className="grid sm:grid-cols-3 lg:grid-cols-5 gap-2">
          {[
            { id: "clip", icon: "🔗", label: "CLIP & VL Pre-training" },
            { id: "vlm", icon: "🧠", label: "Large VLMs" },
            { id: "grounding", icon: "🎯", label: "Visual Grounding" },
            { id: "3d-scene", icon: "📦", label: "3D Scene Reasoning" },
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

        {/* ═══ Part 1: CLIP & Vision-Language Pre-training ═══ */}
        <section id="clip">
          <SectionHeader
            icon={Sparkles}
            title="Vision-Language Pre-training (CLIP)"
            number={1}
            subtitle="Learn aligned image-text representations via contrastive learning on 400M image-text pairs. The foundation for zero-shot visual recognition and open-vocabulary understanding."
          />
          <div className="space-y-4">
            <CLIPEmbeddingCanvas />

            <TheoryInline title="Intuition" />
            <TheoryInline title="Vision-Language Pre-training" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Contrastive Objective" accent="#a855f7">
                For a batch of N image-text pairs, CLIP maximizes cosine similarity for N matching pairs while minimizing it for N²-N non-matching pairs. Temperature τ controls the sharpness.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  L = -log(exp(sim(vᵢ,tᵢ)/τ) / Σⱼ exp(sim(vᵢ,tⱼ)/τ))
                </div>
              </ContentCard>
              <ContentCard title="Zero-Shot Classification" accent="#06b6d4">
                At inference, compare the image embedding against text embeddings of class descriptions ("a photo of a [class]"). No task-specific training — just embedding comparison.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  ŷ = argmax_k sim(f_img(x), f_text("a photo of k"))
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 2: Large Vision-Language Models ═══ */}
        <section id="vlm">
          <SectionHeader
            icon={Brain}
            title="Large Vision-Language Models"
            number={2}
            subtitle="Combine frozen/fine-tuned vision encoders with LLMs for open-ended visual reasoning — GPT-4V, Gemini, LLaVA enable multi-turn dialogue about images."
          />
          <div className="space-y-4">
            <VLMArchitectureCanvas />

            <TheoryInline title="Large Vision-Language Models (LVLMs)" />
            <TheoryInline title="Florence-2: Unified Vision Foundation Model" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="LLaVA Architecture" accent="#a855f7">
                CLIP ViT-L produces 576 visual tokens → MLP projector maps to LLM embedding space → LLaMA processes visual + text tokens jointly. Simple but effective.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  zᵥ = Wₚ · ViT(I) + bₚ<br />
                  h = LLM([zᵥ ; zₜ])
                </div>
              </ContentCard>
              <ContentCard title="Florence-2: One Model, Many Tasks" accent="#06b6d4">
                Single seq2seq architecture handles detection, segmentation, captioning, grounding, and OCR. Task specified via text prompt; outputs are text sequences with coordinates.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  {"<OD>"} → "car [120,80,340,220]"<br />
                  {"<CAPTION>"} → "A red car on a highway"
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 3: Visual Grounding ═══ */}
        <section id="grounding">
          <SectionHeader
            icon={Eye}
            title="Visual Grounding & Referring Expressions"
            number={3}
            subtitle="Localize objects described by natural language — 'the red car behind the tree'. Requires understanding spatial relationships, attributes, and visual context."
          />
          <div className="space-y-4">
            <TheoryInline title="Visual Grounding & Referring Expressions" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Grounding Approaches" accent="#a855f7">
                <strong className="text-foreground">Two-stage:</strong> Generate proposals → rank by text similarity (MDETR).<br /><br />
                <strong className="text-foreground">One-stage:</strong> Predict box directly conditioned on text (GLIP, Florence-2).
              </ContentCard>
              <ContentCard title="Open-Vocabulary Detection" accent="#06b6d4">
                Combine CLIP features with detection heads to detect objects from any text description — not limited to fixed class vocabularies. OWL-ViT and Grounding DINO lead this direction.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  b̂ = argmax P(b | I, "red car behind tree")
                </div>
              </ContentCard>
            </div>

            <TheoryInline title="Real-World Applications" />
          </div>
        </section>

        {/* ═══ Part 4: 3D Scene Understanding ═══ */}
        <section id="3d-scene">
          <SectionHeader
            icon={Box}
            title="3D Scene Understanding & Reasoning"
            number={4}
            subtitle="Go beyond 2D — build 3D scene graphs, answer questions about 3D environments, and enable embodied agents to navigate and interact with the physical world."
          />
          <div className="space-y-4">
            <Scene3DCanvas />

            <TheoryInline title="3D Scene Graphs" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="3D Scene Graphs" accent="#a855f7">
                Nodes = objects with 3D bbox + attributes. Edges = spatial relationships (on, next-to, inside). Built from RGB-D scans via 3D detection + GNN-based relationship prediction.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  G = (V, E): vᵢ = (class, bbox₃D, attrs)<br />
                  eᵢⱼ = spatial_relation
                </div>
              </ContentCard>
              <ContentCard title="3D Visual QA (ScanQA)" accent="#06b6d4">
                Answer natural language questions about 3D point cloud scenes. Requires spatial reasoning, counting, and relationship understanding beyond what 2D VQA can provide.
                <div className="mt-2 font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  â = argmax P(a | P₃D, question)
                </div>
              </ContentCard>
            </div>

            <TheoryInline title="3D Visual Question Answering" />
            <TheoryInline title="Embodied AI & Vision-Language Navigation" />

            <ContentCard title="Embodied AI: Observe → Plan → Act" accent="#f59e0b">
              VLMs as planning backbones: the agent observes the 3D environment through egocentric vision, grounds language instructions in visual observations, builds a spatial memory of explored areas, and plans paths. RT-2 and PaLM-E demonstrate that large VLMs can directly output robot actions.
            </ContentCard>
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
          <ModulePage content={sceneReasoningModule} hideHeader hideTheory />
        </section>
      </div>
    </div>
  );
}
