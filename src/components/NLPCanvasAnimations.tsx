import { useRef, useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";

// ════════════════════════════════════════════════
// 1. BPE TOKENIZER DEMO
// ════════════════════════════════════════════════

const tokenColors = [
  ["hsl(var(--primary) / 0.2)", "hsl(var(--primary) / 0.6)"],
  ["hsl(187 85% 53% / 0.2)", "hsl(187 85% 53% / 0.6)"],
  ["hsl(32 95% 55% / 0.2)", "hsl(32 95% 55% / 0.6)"],
  ["hsl(142 71% 45% / 0.2)", "hsl(142 71% 45% / 0.6)"],
  ["hsl(340 75% 55% / 0.2)", "hsl(340 75% 55% / 0.6)"],
];

const commonBPE: Record<string, string[]> = {
  transformer: ["transform", "er"],
  attention: ["at", "ten", "tion"],
  mechanism: ["mechan", "ism"],
  tokenization: ["token", "ization"],
  language: ["language"],
  learning: ["learn", "ing"],
  architecture: ["arch", "ite", "cture"],
  embedding: ["embed", "ding"],
  neural: ["neural"],
  network: ["net", "work"],
  model: ["model"],
  training: ["train", "ing"],
  generation: ["gen", "er", "ation"],
  queries: ["quer", "ies"],
  values: ["val", "ues"],
  keys: ["keys"],
  uses: ["uses"],
  the: ["the"],
  self: ["self"],
};

export function BPETokenizerDemo() {
  const [input, setInput] = useState("The transformer model uses attention mechanisms");
  const [tokens, setTokens] = useState<{ t: string; type: string; ci: number }[]>([]);
  const [stats, setStats] = useState("");

  const tokenize = useCallback(() => {
    const words = input.split(/(\s+|[.,!?;:])/);
    const result: { t: string; type: string; ci: number }[] = [];
    let idx = 0;
    words.forEach((w) => {
      if (!w || /^\s+$/.test(w)) { if (w) result.push({ t: "·", type: "space", ci: 0 }); return; }
      if (/[.,!?;:]/.test(w)) { result.push({ t: w, type: "punct", ci: idx % 5 }); idx++; return; }
      const subs = commonBPE[w.toLowerCase()] || [w];
      subs.forEach((s, i) => {
        result.push({ t: i === 0 ? s : "##" + s, type: "word", ci: idx % 5 });
        idx++;
      });
    });
    setTokens(result);
    const wordTokens = result.filter((t) => t.type !== "space");
    setStats(`→ ${wordTokens.length} tokens | ${input.length} chars | compression: ${(input.length / wordTokens.length).toFixed(1)}x`);
  }, [input]);

  useEffect(() => { tokenize(); }, []);

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-3">// Interactive BPE Tokenizer — type any text</p>
      <input
        className="w-full rounded-lg border border-border bg-muted/30 px-4 py-2.5 text-sm font-mono text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none transition-colors mb-3"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder='Try: "The self-attention mechanism computes queries, keys and values..."'
      />
      <button
        onClick={tokenize}
        className="px-5 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:bg-primary/90 transition-colors mb-3"
      >
        ⚡ Tokenize
      </button>
      {stats && <p className="text-[10px] font-mono text-muted-foreground mb-3">{stats}</p>}
      <div className="flex flex-wrap gap-2 min-h-[48px]">
        {tokens.map((tok, i) => (
          <motion.span
            key={`${tok.t}-${i}`}
            initial={{ scale: 0.7, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: i * 0.04 }}
            className="font-mono text-xs px-3 py-1.5 rounded-md border text-foreground"
            style={{
              background: tokenColors[tok.ci][0],
              borderColor: tokenColors[tok.ci][1],
              opacity: tok.type === "space" ? 0.4 : 1,
              fontSize: tok.type === "space" ? "10px" : undefined,
            }}
          >
            {tok.t}
          </motion.span>
        ))}
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════
// 2. ATTENTION HEATMAP
// ════════════════════════════════════════════════

const sentence = ["The", "transformer", "attends", "to", "each", "token", "in", "context"];
const attnWeights = [
  [0.4, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1],
  [0.1, 0.5, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05],
  [0.05, 0.25, 0.4, 0.1, 0.05, 0.05, 0.05, 0.05],
  [0.1, 0.1, 0.1, 0.4, 0.1, 0.05, 0.05, 0.1],
  [0.05, 0.05, 0.05, 0.1, 0.5, 0.1, 0.05, 0.1],
  [0.05, 0.1, 0.1, 0.05, 0.15, 0.4, 0.05, 0.1],
  [0.1, 0.05, 0.05, 0.15, 0.05, 0.05, 0.4, 0.15],
  [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.35],
];

export function AttentionHeatmap() {
  const [selected, setSelected] = useState<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const drawHeatmap = useCallback((sel: number | null) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    const n = sentence.length;
    const cellW = (W - 60) / n, cellH = (H - 30) / n;
    const ox = 56, oy = 14;

    ctx.clearRect(0, 0, W, H);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const val = attnWeights[r][c];
        const alpha = sel !== null ? (r === sel ? val : val * 0.15) : val * 0.5;
        ctx.fillStyle = `hsla(265, 70%, 60%, ${alpha})`;
        ctx.fillRect(ox + c * cellW, oy + r * cellH, cellW - 2, cellH - 2);
        if (sel === r) {
          ctx.fillStyle = `rgba(255,255,255,${Math.min(val * 2, 1)})`;
          ctx.font = `${Math.max(8, cellH * 0.4)}px monospace`;
          ctx.textAlign = "center";
          ctx.fillText((val * 100).toFixed(0), ox + c * cellW + cellW / 2, oy + r * cellH + cellH / 2 + 3);
        }
      }
    }
    ctx.fillStyle = "hsl(var(--muted-foreground))";
    ctx.font = "9px monospace";
    sentence.forEach((w, i) => {
      ctx.textAlign = "right";
      ctx.fillText(w.slice(0, 5), ox - 4, oy + i * cellH + cellH / 2 + 3);
      ctx.save();
      ctx.translate(ox + i * cellW + cellW / 2, oy - 2);
      ctx.rotate(-Math.PI / 4);
      ctx.textAlign = "left";
      ctx.fillText(w.slice(0, 5), 0, 0);
      ctx.restore();
    });
  }, []);

  useEffect(() => { drawHeatmap(selected); }, [selected, drawHeatmap]);

  const explain = selected !== null
    ? `"${sentence[selected]}" attends most to: ${attnWeights[selected].map((w, j) => ({ w, j })).sort((a, b) => b.w - a.w).slice(0, 3).map((x) => `"${sentence[x.j]}"(${(x.w * 100).toFixed(0)}%)`).join(", ")}`
    : "← Click any word to visualize its attention weights";

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-3">// Click a word — see attention pattern</p>
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <div className="flex flex-wrap gap-1.5 mb-3">
            {sentence.map((w, i) => (
              <button
                key={i}
                onClick={() => setSelected(i)}
                className={`font-mono text-xs px-2.5 py-1.5 rounded-md border transition-all ${
                  selected === i
                    ? "border-accent bg-accent/15 text-accent"
                    : "border-transparent bg-muted/30 text-foreground hover:border-muted-foreground/30"
                }`}
                style={
                  selected !== null
                    ? { background: `hsl(32 95% 55% / ${attnWeights[selected][i] * 0.8})`, borderColor: `hsl(32 95% 55% / ${attnWeights[selected][i]})` }
                    : undefined
                }
              >
                {w}
              </button>
            ))}
          </div>
          <p className="text-[11px] font-mono text-muted-foreground leading-relaxed">{explain}</p>
        </div>
        <canvas ref={canvasRef} width={400} height={220} className="w-full rounded-lg" />
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════
// 3. TRANSFORMER PIPELINE (Interactive hover)
// ════════════════════════════════════════════════

const tfBlocks = [
  { label: "Tokenizer", sub: "raw text", color: "hsl(var(--primary))", tip: "Raw text → integer IDs via BPE/WordPiece. Each ID indexes into the embedding matrix." },
  { label: "Embed + PosEnc", sub: "ℝ^{n×d}", color: "hsl(187 85% 53%)", tip: "E ∈ ℝ^{V×d}: maps token ID to d-dimensional vector. Positional encoding: PE(pos,2i)=sin(pos/10000^{2i/d})" },
  { label: "Multi-Head Attention", sub: "×L layers", color: "hsl(32 95% 55%)", tip: "h heads each compute Attn(Q,K,V)=softmax(QKᵀ/√d_k)V. Concatenated and projected: O(n²·d)." },
  { label: "FFN + LayerNorm", sub: "residual", color: "hsl(142 71% 45%)", tip: "FFN(x) = max(0,xW₁+b₁)W₂+b₂. d_ff=4×d_model. SwiGLU variant used in LLaMA." },
  { label: "LM Head Softmax", sub: "P(vocab)", color: "hsl(340 75% 55%)", tip: "Linear projection ℝ^d → ℝ^V then softmax → probability over vocabulary. Tied with embedding." },
];

export function TransformerPipelineViz() {
  const [hovered, setHovered] = useState<number | null>(null);

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-4">// Hover each block — full forward pass</p>
      <div className="flex items-center gap-0 overflow-x-auto pb-2">
        {tfBlocks.map((block, i) => (
          <div key={i} className="flex items-center">
            <motion.div
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
              whileHover={{ y: -4, scale: 1.03 }}
              className="flex flex-col items-center gap-1.5 min-w-[100px] cursor-pointer"
            >
              <div
                className="w-full px-2 py-3 rounded-lg text-center font-mono text-[10px] font-bold border transition-all"
                style={{
                  background: `${block.color.replace(")", " / 0.15)")}`,
                  borderColor: `${block.color.replace(")", " / 0.5)")}`,
                  color: block.color,
                }}
              >
                {block.label}
              </div>
              <span className="text-[9px] text-muted-foreground">{block.sub}</span>
            </motion.div>
            {i < tfBlocks.length - 1 && <span className="text-muted-foreground text-lg px-1 mt-[-20px]">→</span>}
          </div>
        ))}
      </div>
      {hovered !== null && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-3 rounded-lg border border-border bg-muted/40 p-3 text-xs font-mono text-muted-foreground"
        >
          <strong className="text-foreground">{tfBlocks[hovered].label}:</strong> {tfBlocks[hovered].tip}
        </motion.div>
      )}
      <p className="mt-3 text-[10px] font-mono text-center text-muted-foreground">
        Pre-LN Transformer Block: x' = x + MHA(LN(x)) → output = x' + FFN(LN(x'))
      </p>
    </div>
  );
}

// ════════════════════════════════════════════════
// 4. REACT AGENT LOOP
// ════════════════════════════════════════════════

const agentSteps = [
  { node: "think", log: "💭 THOUGHT: The user wants German customers with orders > $1000. I need to query the database." },
  { node: "act", log: '⚡ ACTION: sql_query("SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON c.id=o.customer_id WHERE c.country=\'Germany\' GROUP BY c.name HAVING SUM(o.amount) > 1000")' },
  { node: "obs", log: '👁 OBSERVATION: [("Mueller GmbH", 2340.00), ("Berlin Tech", 1856.50), ("Hamburg Co", 1102.20)]' },
  { node: "think2", log: "💭 THOUGHT: I have 3 results. Let me format this clearly." },
  { node: "answer", log: "✅ FINAL ANSWER: Found 3 German customers with total orders > $1,000:\n1. Mueller GmbH — $2,340\n2. Berlin Tech — $1,857\n3. Hamburg Co — $1,102" },
];

const nodeConfig: Record<string, { icon: string; label: string; color: string }> = {
  think: { icon: "🧠", label: "THINK", color: "hsl(var(--primary))" },
  think2: { icon: "🧠", label: "THINK", color: "hsl(var(--primary))" },
  act: { icon: "⚡", label: "ACT", color: "hsl(32 95% 55%)" },
  obs: { icon: "👁", label: "OBSERVE", color: "hsl(142 71% 45%)" },
  answer: { icon: "✅", label: "ANSWER", color: "hsl(187 85% 53%)" },
};

export function AgentLoopViz() {
  const [activeStep, setActiveStep] = useState(-1);
  const [log, setLog] = useState("Press Run to simulate a ReAct agent solving a SQL query task.");
  const [running, setRunning] = useState(false);

  const run = useCallback(() => {
    if (running) return;
    setRunning(true);
    setLog("");
    setActiveStep(-1);
    agentSteps.forEach((step, i) => {
      setTimeout(() => {
        setActiveStep(i);
        setLog((prev) => (prev ? prev + "\n" : "") + step.log);
        if (i === agentSteps.length - 1) {
          setTimeout(() => setRunning(false), 1000);
        }
      }, i * 1500);
    });
  }, [running]);

  const loopNodes = ["think", "act", "obs", "answer"];

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4 text-center">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-4">// ReAct Agent Loop — Thought → Action → Observe</p>
      <div className="flex items-center justify-center gap-0 flex-wrap mb-4">
        {loopNodes.map((nodeId, i) => {
          const cfg = nodeConfig[nodeId];
          const isActive = activeStep >= 0 && (agentSteps[activeStep]?.node === nodeId || agentSteps[activeStep]?.node === nodeId + "2");
          return (
            <div key={nodeId} className="flex items-center">
              <motion.div
                animate={isActive ? { scale: 1.1 } : { scale: 1 }}
                className={`w-[100px] h-[70px] rounded-xl flex flex-col items-center justify-center border-2 transition-all ${isActive ? "shadow-lg" : ""}`}
                style={{
                  background: `${cfg.color.replace(")", " / 0.15)")}`,
                  borderColor: isActive ? cfg.color : `${cfg.color.replace(")", " / 0.3)")}`,
                }}
              >
                <span className="text-xl">{cfg.icon}</span>
                <span className="text-[10px] font-mono font-bold" style={{ color: cfg.color }}>{cfg.label}</span>
              </motion.div>
              {i < loopNodes.length - 1 && <span className="text-muted-foreground text-lg px-2">→</span>}
            </div>
          );
        })}
      </div>
      <button
        onClick={run}
        disabled={running}
        className="px-6 py-2 rounded-lg bg-primary/80 text-primary-foreground text-xs font-semibold hover:bg-primary transition-colors disabled:opacity-50 mb-3"
      >
        {running ? "⏳ Running..." : "▶ Run Agent Example"}
      </button>
      <div className="rounded-lg bg-muted/30 border border-border p-3 text-left font-mono text-[10px] text-muted-foreground min-h-[60px] whitespace-pre-wrap leading-relaxed">
        {log}
      </div>
    </div>
  );
}
