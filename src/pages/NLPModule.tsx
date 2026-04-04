import ModulePage from "@/components/ModulePage";
import { nlpLLMModule } from "@/data/nlpModuleData";
import { MathEquation } from "@/components/MathBlock";
import AITutor from "@/components/AITutor";
import { BPETokenizerDemo, AttentionHeatmap, TransformerPipelineViz, AgentLoopViz } from "@/components/NLPCanvasAnimations";
import { ArrowLeft, GraduationCap, Type, Brain, Cpu, Layers, Zap, Bot, BookOpen, Wrench, Users, Network } from "lucide-react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useSectionObserver } from "@/hooks/useSectionObserver";
import { Progress } from "@/components/ui/progress";

const color = nlpLLMModule.color;

const theoryByTitle: Record<string, typeof nlpLLMModule.theory[0]> = {};
nlpLLMModule.theory.forEach(s => { theoryByTitle[s.title] = s; });

function TheoryInline({ title }: { title: string }) {
  const section = theoryByTitle[title];
  if (!section) return null;
  return (
    <div className="concept-card">
      <div className="flex items-center flex-wrap gap-y-1 mb-3">
        <h3 className="font-semibold text-foreground text-sm">{section.title}</h3>
        <AITutor conceptTitle={section.title} conceptContent={section.content} moduleName="Agentic AI" />
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

export default function NLPModule() {
  const progressPct = useSectionObserver("nlp-llm", ['foundations', 'attention', 'transformer', 'models', 'training', 'efficiency', 'agents', 'multiagent', 'review']);

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
          <h1 className="text-2xl font-bold text-foreground tracking-tight">{nlpLLMModule.title}</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{nlpLLMModule.subtitle}</p>
        </div>
      </div>

      {/* Learning flow nav */}
      <div className="rounded-xl border border-border bg-muted/30 p-4 mb-8">
        <h2 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3">Structured Learning Flow</h2>
        <div className="grid sm:grid-cols-3 lg:grid-cols-9 gap-2">
          {[
            { id: "foundations", icon: "📝", label: "Tokenization & Embeddings" },
            { id: "attention", icon: "🧠", label: "Self-Attention" },
            { id: "transformer", icon: "⚡", label: "Transformer" },
            { id: "models", icon: "🏗️", label: "BERT & GPT" },
            { id: "training", icon: "🎯", label: "Training & Alignment" },
            { id: "efficiency", icon: "🔧", label: "RAG & LoRA" },
            { id: "agents", icon: "🤖", label: "Tools & Agents" },
            { id: "multiagent", icon: "🌐", label: "Multi-Agent & MCP" },
            { id: "review", icon: "📚", label: "Review" },
          ].map((item) => (
            <a key={item.id} href={`#${item.id}`} className="rounded-lg border border-border bg-card p-2.5 hover:border-primary/40 transition-colors text-center">
              <p className="text-sm mb-0.5">{item.icon}</p>
              <p className="text-[10px] text-foreground font-medium">{item.label}</p>
            </a>
          ))}
        </div>
      </div>

      <div className="space-y-12">

        {/* ═══ Part 1: Tokenization & Embeddings ═══ */}
        <section id="foundations">
          <SectionHeader icon={Type} title="Tokenization & Embeddings" number={1} subtitle="Convert raw text into numerical representations — from LLM foundations through BPE tokenization and learned embeddings." />
          <div className="space-y-4">
            <TheoryInline title="Intuition" />
            <TheoryInline title="Foundation: What is an LLM?" />
            <BPETokenizerDemo />
            <TheoryInline title="Tokenization & BPE" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="BPE Worked Example" accent="#38bdf8">
                <p className="mb-2">Corpus: {"{"}low: 5, lower: 2, newest: 6{"}"}</p>
                <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                  <p>Step 1: merge(e,s) → es (freq 6)</p>
                  <p>Step 2: merge(es,t) → est (freq 6)</p>
                  <p>Step 3: merge(l,o) → lo (freq 7)</p>
                  <p>Step 4: merge(lo,w) → low (freq 7)</p>
                </div>
              </ContentCard>
              <ContentCard title="Context Window Evolution" accent="#a855f7">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">GPT-2:</span> 1,024 tokens</p>
                  <p><span className="text-foreground font-medium">GPT-3:</span> 2,048 tokens</p>
                  <p><span className="text-foreground font-medium">GPT-4:</span> 128K tokens</p>
                  <p><span className="text-foreground font-medium">Gemini:</span> 1M tokens</p>
                </div>
                <p className="mt-2 text-xs">Flash Attention reduces O(n²) memory → O(n) via tiling. RoPE interpolation extends beyond training context.</p>
              </ContentCard>
            </div>

            <TheoryInline title="Word Embeddings & Representations" />

            <ContentCard title="Embedding Analogy" accent="#06b6d4">
              <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                v(king) - v(man) + v(woman) ≈ v(queen)<br />
                Parallelogram structure: relational vectors are parallel
              </div>
            </ContentCard>
          </div>
        </section>

        {/* ═══ Part 2: Self-Attention ═══ */}
        <section id="attention">
          <SectionHeader icon={Brain} title="Self-Attention Mechanism" number={2} subtitle="The core operation that lets every token attend to every other token — understanding the QKV computation, scaling, and masking." />
          <div className="space-y-4">
            <AttentionHeatmap />
            <TheoryInline title="Self-Attention Mechanism" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Why √dₖ Scaling?" accent="#a855f7">
                For dₖ-dimensional vectors with unit-variance components:<br /><br />
                Var(q·k) = dₖ → std dev = √dₖ<br /><br />
                Without scaling: softmax becomes near-one-hot → <strong className="text-foreground">vanishing gradients</strong>. Dividing by √dₖ restores unit variance.
              </ContentCard>
              <ContentCard title="Complexity Analysis" accent="#06b6d4">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">Time:</span> O(n²·d) — n² attention scores × d dimension</p>
                  <p><span className="text-foreground font-medium">Space:</span> O(n²) — storing n×n attention matrix</p>
                  <p><span className="text-foreground font-medium">Flash Attention:</span> O(n) memory via block tiling</p>
                  <p><span className="text-foreground font-medium">Linear Attention:</span> O(n·d) via kernel approximation</p>
                </div>
              </ContentCard>
            </div>

            <TheoryInline title="Multi-Head Attention & Positional Encoding" />

            <ContentCard title="Worked Example — 3-Token Attention" accent="#f59e0b">
              <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                <p>Input X = [[1,0],[0,1],[1,1]], dₖ=2</p>
                <p>QKᵀ = [[1,0,1],[0,1,1],[1,1,2]] / √2</p>
                <p>softmax(row 3): [0.236, 0.236, 0.528]</p>
                <p>Output₃ = 0.236·[1,0] + 0.236·[0,1] + 0.528·[1,1] = [0.764, 0.764]</p>
              </div>
            </ContentCard>
          </div>
        </section>

        {/* ═══ Part 3: Transformer Architecture ═══ */}
        <section id="transformer">
          <SectionHeader icon={Cpu} title="Full Transformer Architecture" number={3} subtitle="Combine attention, FFN, normalization, and residuals into the universal backbone for text, images, and generation." />
          <div className="space-y-4">
            <TransformerPipelineViz />
            <TheoryInline title="Full Transformer Block" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="SwiGLU FFN (LLaMA)" accent="#a855f7">
                <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  FFN(x) = (xW₁ ⊙ swish(xW₃)) · W₂
                </div>
                <p className="mt-2 text-xs">Gated linear unit with Swish activation — used in LLaMA, PaLM, Gemma. 3 weight matrices instead of 2.</p>
              </ContentCard>
              <ContentCard title="Parameter Count" accent="#06b6d4">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">MHA:</span> 4·d²_model parameters</p>
                  <p><span className="text-foreground font-medium">FFN:</span> 8·d²_model parameters</p>
                  <p><span className="text-foreground font-medium">BERT-Base:</span> 12L, 768d, 12H = 110M</p>
                  <p><span className="text-foreground font-medium">GPT-3:</span> 96L, 12288d, 96H = 175B</p>
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 4: BERT vs GPT ═══ */}
        <section id="models">
          <SectionHeader icon={Layers} title="BERT vs GPT — Model Families" number={4} subtitle="Bidirectional understanding (BERT) vs autoregressive generation (GPT) — two paradigms that shaped modern NLP." />
          <div className="space-y-4">
            <TheoryInline title="BERT vs GPT — Encoder vs Decoder" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="BERT — Bidirectional Encoder" accent="#a855f7">
                <strong className="text-foreground">Mask 15%:</strong> 80% [MASK], 10% random, 10% unchanged.<br /><br />
                <strong className="text-foreground">[CLS] token:</strong> aggregates global sequence info for classification.<br /><br />
                Best for: NER, classification, semantic similarity.
              </ContentCard>
              <ContentCard title="GPT — Autoregressive Decoder" accent="#38bdf8">
                <strong className="text-foreground">Causal mask:</strong> position i cannot attend to j {">"} i.<br /><br />
                <strong className="text-foreground">Decoding strategies:</strong><br />
                • Greedy: argmax P(x|x{"<"}t)<br />
                • Top-p (nucleus): sample from top-p cumulative<br />
                • Temperature T: P_T(x) ∝ exp(logit/T)
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 5: Training & Alignment ═══ */}
        <section id="training">
          <SectionHeader icon={Zap} title="LLM Training & Alignment" number={5} subtitle="The four-stage pipeline: pre-training → SFT → RLHF/DPO — and the scaling laws that predict model capabilities." />
          <div className="space-y-4">
            <TheoryInline title="LLM Training Pipeline — SFT, RLHF & DPO" />
            <TheoryInline title="Neural Scaling Laws" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Chinchilla Predictions vs Reality" accent="#a855f7">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">GPT-3:</span> 175B / 300B tokens — ❌ under-trained 10×</p>
                  <p><span className="text-foreground font-medium">Chinchilla:</span> 70B / 1.4T tokens — ✅ optimal</p>
                  <p><span className="text-foreground font-medium">LLaMA 3:</span> 405B / 15T+ tokens — ✓ over-trained</p>
                  <p><span className="text-foreground font-medium">Gemma 2:</span> 9B / 8T tokens — ✓ highly over-trained</p>
                </div>
              </ContentCard>
              <ContentCard title="Emergent Abilities" accent="#06b6d4">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">~8B:</span> consistent instruction following</p>
                  <p><span className="text-foreground font-medium">~13B:</span> 3-digit arithmetic</p>
                  <p><span className="text-foreground font-medium">~100B:</span> chain-of-thought reasoning</p>
                  <p><span className="text-foreground font-medium">~540B:</span> multi-step code generation</p>
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 6: RAG & LoRA ═══ */}
        <section id="efficiency">
          <SectionHeader icon={BookOpen} title="RAG & Efficient Fine-Tuning" number={6} subtitle="Ground LLMs in external knowledge with retrieval-augmented generation, and adapt them efficiently with LoRA." />
          <div className="space-y-4">
            <TheoryInline title="RAG & LoRA — Retrieval and Efficient Fine-Tuning" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="LoRA Savings" accent="#a855f7">
                <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                  <p>d = k = 4096, r = 16</p>
                  <p>Full: 4096 × 4096 = 16.7M params</p>
                  <p>LoRA: 16 × (4096 + 4096) = 131K params</p>
                  <p>Savings: 99.2%</p>
                </div>
                <p className="mt-2 text-xs">QLoRA: 4-bit base + BF16 LoRA → 65B model on single 48GB GPU.</p>
              </ContentCard>
              <ContentCard title="Chain-of-Thought Prompting" accent="#06b6d4">
                <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  P(y|x) = P(y|r,x) · P(r|x)<br />
                  where r = reasoning chain
                </div>
                <p className="mt-2 text-xs">Intermediate steps reduce search space. Self-consistency: sample K chains, majority vote → higher accuracy.</p>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 7: Tools & Agents ═══ */}
        <section id="agents">
          <SectionHeader icon={Wrench} title="Tools & Agents" number={7} subtitle="From passive LLMs to autonomous decision-makers — tools extend capabilities, agents add reasoning loops." />
          <div className="space-y-4">
            <TheoryInline title="Tools: Giving LLMs Capabilities" />
            <AgentLoopViz />
            <TheoryInline title="Agents: Decision-Making Systems" />
            <TheoryInline title="Agent Architecture — Planner, Executor, Controller" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="ReAct Trajectory" accent="#f59e0b">
                <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                  <p>τ = (query, Thought₁, Action₁, Obs₁, Thought₂, ...)</p>
                  <p>Tool selection: P(aₜ|hₜ) ∝ exp(LLM_score(aₜ, hₜ))</p>
                </div>
              </ContentCard>
              <ContentCard title="Agent vs Chatbot" accent="#06b6d4">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">Chatbot:</span> input → output (single pass)</p>
                  <p><span className="text-foreground font-medium">Agent:</span> observe → think → act → repeat</p>
                  <p><span className="text-foreground font-medium">Key:</span> agents decide their own actions autonomously</p>
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 8: Multi-Agent & MCP ═══ */}
        <section id="multiagent">
          <SectionHeader icon={Network} title="Multi-Agent Systems & MCP" number={8} subtitle="Scale intelligence with specialized agent teams, memory systems, and the Model Context Protocol for standardized tool access." />
          <div className="space-y-4">
            <TheoryInline title="Memory: Adding State to Agents" />
            <TheoryInline title="Multi-Agent Systems" />
            <TheoryInline title="MCP — Model Context Protocol" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="MCP Architecture" accent="#a855f7">
                <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                  <p>LLM Agent → MCP Client → MCP Server(s)</p>
                  <p>MCP Server hosts tools via JSON-RPC 2.0</p>
                  <p>FastMCP: @tool → network-accessible tool</p>
                  <p>LangChain Adapters: MCP → LangGraph Agent</p>
                </div>
              </ContentCard>
              <ContentCard title="Memory Types" accent="#06b6d4">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">Short-term:</span> conversation history (context window)</p>
                  <p><span className="text-foreground font-medium">Long-term:</span> vector DB for cross-session retrieval</p>
                  <p><span className="text-foreground font-medium">Working:</span> current task state & scratchpad</p>
                </div>
              </ContentCard>
            </div>

            <TheoryInline title="OpenClaw & Paperclip — Agent OS and Organizations" />
            <TheoryInline title="Evolution of Intelligence" />

            <ContentCard title="Intelligence Evolution Stack" accent="#f59e0b">
              <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                <p>LLM (Think) → Tools (Act) → Agent (Decide)</p>
                <p>→ Memory (Remember) → Multi-Agent (Collaborate)</p>
                <p>→ MCP (Integrate) → OpenClaw (Systemize)</p>
                <p>→ Paperclip (Organize)</p>
              </div>
            </ContentCard>
          </div>
        </section>

        {/* ═══ Part 9: Review ═══ */}
        <section id="review">
          <SectionHeader icon={BookOpen} title="Papers, Algorithms & Practice" number={9} subtitle="Key research papers, consolidated algorithms, and quizzes to test your understanding." />
          <div className="space-y-4">
            <ModulePage content={nlpLLMModule} hideHeader hideTheory />
          </div>
        </section>
      </div>
    </div>
  );
}
