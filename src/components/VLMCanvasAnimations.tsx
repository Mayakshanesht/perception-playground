import { useRef, useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";

// ════════════════════════════════════════════════
// 1. ViT PATCH DEMO
// ════════════════════════════════════════════════

export function ViTPatchDemo() {
  const [selectedPatch, setSelectedPatch] = useState<{ idx: number; r: number; c: number } | null>(null);
  const cols = 7, rows = 7;
  const hues = [180, 200, 160, 210, 170, 190, 220];

  const patches = Array.from({ length: rows * cols }, (_, i) => {
    const r = Math.floor(i / cols), c = i % cols;
    const h = hues[(r + c) % hues.length] + ((r * c) % 20);
    const l = 20 + (((r * 7 + c * 13) % 25));
    return { r, c, h, l, idx: i };
  });

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-3">// Click a patch — see how images become token sequences</p>
      <div className="flex gap-6 flex-wrap items-start">
        <div>
          <div className="grid gap-[3px]" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, width: 252, height: 252 }}>
            {patches.map((p) => (
              <motion.div
                key={p.idx}
                onClick={() => setSelectedPatch(p)}
                whileHover={{ scale: 1.08, zIndex: 10 }}
                className={`rounded cursor-pointer border transition-all ${
                  selectedPatch?.idx === p.idx ? "border-accent shadow-lg shadow-accent/30" : "border-transparent"
                }`}
                style={{ background: `hsl(${p.h}, 60%, ${p.l}%)` }}
              />
            ))}
          </div>
          <p className="text-[9px] font-mono text-muted-foreground mt-2 text-center">7×7 patches (P=40px) · N=49 tokens</p>
        </div>
        <div className="flex-1 min-w-[220px]">
          <div className="rounded-lg bg-muted/30 border-l-[3px] border-primary p-3 font-mono text-xs text-primary leading-relaxed mb-3">
            z₀ = [x_class; x_p¹E; …; x_pᴺE] + E_pos<br />
            N = HW/P² patches, E ∈ ℝ^{"{(P²C)×d}"}<br />
            For L layers: zₗ = MSA(LN(zₗ₋₁)) + zₗ₋₁
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed mb-2">Each patch is flattened and projected to dimension d. A [CLS] token aggregates global info. Positional embeddings add spatial awareness.</p>
          <p className="text-xs text-muted-foreground"><strong className="text-primary">Key insight:</strong> ViT treats image patches exactly like text tokens.</p>
          <div className="mt-3 rounded-lg border border-accent/20 bg-accent/5 p-2.5 font-mono text-[10px] text-accent min-h-[40px]">
            {selectedPatch
              ? <>
                  Patch #{selectedPatch.idx + 1} of 49 · Position (row {selectedPatch.r}, col {selectedPatch.c})<br />
                  Embedding token x_p{selectedPatch.idx + 1}E ∈ ℝ^{"{3072}"} (P²×C = 40²×3)<br />
                  After linear proj → ℝ^{"{768}"} (d_model)<br />
                  + positional embedding E_pos[{selectedPatch.idx + 1}]
                </>
              : "← Click any patch to inspect its position and embedding info."}
          </div>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════
// 2. CLIP SIMILARITY DEMO
// ════════════════════════════════════════════════

const imgLabels = ["🐶", "🏙️", "🌅", "🎸"];
const imgNames = ["dog", "city", "sunset", "guitar"];
const txtLabels = ['"a photo of a dog"', '"urban city skyline at night"', '"beautiful sunset over mountains"', '"electric guitar close-up"'];
const simMatrix = [
  [0.92, 0.08, 0.12, 0.06],
  [0.07, 0.91, 0.14, 0.09],
  [0.11, 0.13, 0.93, 0.07],
  [0.05, 0.08, 0.06, 0.94],
];

export function CLIPSimilarityDemo() {
  const [selImg, setSelImg] = useState(-1);
  const [selTxt, setSelTxt] = useState(-1);

  const sim = selImg >= 0 && selTxt >= 0 ? simMatrix[selImg][selTxt] : null;
  const isMatch = selImg === selTxt;

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-3">// Select an image AND text — watch CLIP similarity</p>
      <div className="grid grid-cols-[1fr_auto_1fr] gap-4 items-center">
        {/* Image side */}
        <div className="rounded-lg border border-border/50 bg-muted/20 p-4">
          <p className="text-[10px] font-mono text-primary mb-3">// IMAGE ENCODER (ViT)</p>
          <div className="grid grid-cols-2 gap-2.5 mb-2">
            {imgLabels.map((emoji, i) => (
              <button
                key={i}
                onClick={() => setSelImg(i)}
                className={`aspect-square rounded-lg flex items-center justify-center text-2xl border-2 transition-all bg-muted/20 ${
                  selImg === i ? "border-accent bg-accent/10 shadow-lg" : "border-transparent hover:border-primary/30"
                }`}
              >
                {emoji}
              </button>
            ))}
          </div>
          <p className="text-[9px] font-mono text-muted-foreground">v_x = CLIP_image(x) → ℝ^{"{512}"}</p>
        </div>

        {/* Middle */}
        <div className="flex flex-col items-center gap-2">
          <span className="font-mono text-2xl font-bold" style={{ color: sim !== null ? (sim > 0.7 ? "hsl(var(--primary))" : sim > 0.4 ? "hsl(32 95% 55%)" : "hsl(340 75% 55%)") : "hsl(var(--muted-foreground))" }}>
            {sim !== null ? sim.toFixed(2) : "—"}
          </span>
          <span className="text-[9px] font-mono text-muted-foreground text-center">cosine similarity<br />s(v,t)</span>
          <div className="w-20 h-2 rounded-full bg-muted/40 overflow-hidden">
            <div className="h-full rounded-full bg-primary transition-all duration-300" style={{ width: `${(sim || 0) * 100}%` }} />
          </div>
          {sim !== null && (
            <span className={`text-[10px] font-mono font-bold ${isMatch ? "text-primary" : "text-destructive"}`}>
              {isMatch ? "✓ MATCH" : "✗ MISMATCH"}
            </span>
          )}
        </div>

        {/* Text side */}
        <div className="rounded-lg border border-border/50 bg-muted/20 p-4">
          <p className="text-[10px] font-mono text-accent mb-3">// TEXT ENCODER (Transformer)</p>
          <div className="flex flex-col gap-2">
            {txtLabels.map((txt, i) => (
              <button
                key={i}
                onClick={() => setSelTxt(i)}
                className={`text-left font-mono text-[10px] px-3 py-2 rounded-md border transition-all ${
                  selTxt === i ? "border-accent bg-accent/10" : "border-transparent bg-muted/20 hover:border-primary/30"
                }`}
              >
                {txt}
              </button>
            ))}
          </div>
          <p className="text-[9px] font-mono text-muted-foreground mt-2">t_c = CLIP_text(caption) → ℝ^{"{512}"}</p>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════
// 3. VLM ARCHITECTURE TABS
// ════════════════════════════════════════════════

const vlmModels = [
  {
    name: "LLaVA",
    pipeline: [
      { label: "Image I", color: "hsl(var(--primary))" },
      { label: "CLIP ViT\nf_clip(I)", color: "hsl(265 70% 60%)" },
      { label: "Linear W\nℝ^{d_clip→d_LM}", color: "hsl(32 95% 55%)" },
      { label: "LLM\nVicuna/Mistral", color: "hsl(187 85% 53%)" },
      { label: "Text Output", color: "hsl(340 75% 55%)" },
    ],
    math: "H_v = f_clip(I_v)   (N×1024 CLIP features)\nX_v = W · H_v   where W ∈ ℝ^{d_LM × d_clip}\nP(y|I,x) = Π_t P_θ(y_t | visual_tokens(I), text(x), y_{<t})",
    notes: "Stage 1: freeze ViT + LLM, train only W. Stage 2: unfreeze LLM, full instruction fine-tuning. 665K visual instruction pairs.",
  },
  {
    name: "Flamingo",
    pipeline: [
      { label: "Image I", color: "hsl(var(--primary))" },
      { label: "Vision\nEncoder", color: "hsl(265 70% 60%)" },
      { label: "Gated\nCross-Attn", color: "hsl(32 95% 55%)" },
      { label: "Frozen\nChinchilla", color: "hsl(187 85% 53%)" },
      { label: "Generated\nAnswer", color: "hsl(340 75% 55%)" },
    ],
    math: "Y_attn = Attn(X_text, X_vis, X_vis)\nY = X_text + tanh(α) · Y_attn\nGate α = 0 at init → pure LM behavior at start",
    notes: "Gated residual: tanh(α) ensures model starts as pure language model. α learned per layer. Visual tokens injected via cross-attention.",
  },
  {
    name: "Perceiver Resampler",
    pipeline: [
      { label: "N Visual\nTokens", color: "hsl(var(--primary))" },
      { label: "Cross-Attn\nResampler", color: "hsl(265 70% 60%)" },
      { label: "64 Learned\nQueries", color: "hsl(32 95% 55%)" },
      { label: "Fixed 64\nTokens", color: "hsl(187 85% 53%)" },
      { label: "Into LLM", color: "hsl(340 75% 55%)" },
    ],
    math: "Z = CrossAttn(Q_learned ∈ ℝ^{64×d}, X_vis, X_vis)\n64 queries → compress variable N visual tokens\nFixed output regardless of image resolution",
    notes: "Critical for efficiency with many images. N can be 256 or 1024 — always compressed to exactly 64 tokens.",
  },
];

export function VLMArchitectureTabs() {
  const [active, setActive] = useState(0);
  const model = vlmModels[active];

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-3">// Select a VLM architecture to explore</p>
      <div className="flex gap-2 mb-4 flex-wrap">
        {vlmModels.map((m, i) => (
          <button
            key={m.name}
            onClick={() => setActive(i)}
            className={`font-mono text-[10px] px-4 py-1.5 rounded-full border transition-all ${
              active === i ? "bg-primary border-primary text-primary-foreground" : "border-border text-muted-foreground hover:border-primary/40"
            }`}
          >
            {m.name}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-0 overflow-x-auto pb-2 mb-3">
        {model.pipeline.map((block, i) => (
          <div key={i} className="flex items-center">
            <div
              className="min-w-[90px] px-3 py-3 rounded-lg text-center font-mono text-[10px] font-bold border whitespace-pre-line"
              style={{
                background: `${block.color.replace(")", " / 0.15)")}`,
                borderColor: `${block.color.replace(")", " / 0.5)")}`,
                color: block.color,
              }}
            >
              {block.label}
            </div>
            {i < model.pipeline.length - 1 && <span className="text-muted-foreground text-lg px-1.5">→</span>}
          </div>
        ))}
      </div>
      <div className="rounded-lg bg-muted/30 border-l-[3px] border-primary p-3 font-mono text-[11px] text-primary leading-relaxed mb-2 whitespace-pre-wrap">
        {model.math}
      </div>
      <p className="text-xs text-muted-foreground leading-relaxed">{model.notes}</p>
    </div>
  );
}

// ════════════════════════════════════════════════
// 4. NERF RAY CASTING DEMO
// ════════════════════════════════════════════════

export function NeRFRayCastingDemo() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stats, setStats] = useState("Move cursor over canvas to cast ray...");

  const drawScene = useCallback((mx: number, my: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background
    const grd = ctx.createLinearGradient(0, 0, 0, H);
    grd.addColorStop(0, "#020617");
    grd.addColorStop(1, "#0f172a");
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, W, H);

    // Grid floor
    ctx.strokeStyle = "rgba(13,148,136,0.15)";
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 20) {
      ctx.beginPath(); ctx.moveTo(x, H * 0.55); ctx.lineTo(W / 2 + (x - W / 2) * 0.3, H * 0.85); ctx.stroke();
    }

    // Spheres
    const objs = [
      { x: 120, y: 150, r: 45, col: "rgba(13,148,136,0.7)" },
      { x: 230, y: 140, r: 35, col: "rgba(139,92,246,0.7)" },
      { x: 180, y: 120, r: 25, col: "rgba(245,158,11,0.7)" },
    ];
    objs.forEach((o) => {
      const grd2 = ctx.createRadialGradient(o.x - o.r * 0.3, o.y - o.r * 0.3, 0, o.x, o.y, o.r);
      grd2.addColorStop(0, "rgba(255,255,255,0.4)");
      grd2.addColorStop(1, o.col);
      ctx.fillStyle = grd2;
      ctx.beginPath(); ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2); ctx.fill();
    });

    // Camera
    const cx = W / 2, cy = H - 24;
    ctx.fillStyle = "rgba(6,182,212,0.9)";
    ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.fill();

    // Ray
    if (mx > 0 && my > 0) {
      const dx = mx - cx, dy = my - cy;
      const samples = 12;
      ctx.lineWidth = 1.5;
      for (let i = 0; i < samples; i++) {
        const t = i / samples;
        const rx = cx + dx * t, ry = cy + dy * t;
        let density = 0;
        objs.forEach((o) => { const d = Math.sqrt((rx - o.x) ** 2 + (ry - o.y) ** 2); density += Math.max(0, 1 - d / o.r) * 0.5; });
        const alpha = Math.min(0.9, density + 0.2 * (1 - t));
        ctx.strokeStyle = `rgba(244,63,94,${alpha})`;
        ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(cx + dx * (i + 1) / samples, cy + dy * (i + 1) / samples); ctx.stroke();
      }
      for (let i = 1; i < samples; i += 2) {
        const t = i / samples;
        const rx = cx + dx * t, ry = cy + dy * t;
        let density = 0;
        objs.forEach((o) => { const d = Math.sqrt((rx - o.x) ** 2 + (ry - o.y) ** 2); density += Math.max(0, 1 - d / o.r) * 0.8; });
        ctx.fillStyle = `rgba(245,158,11,${Math.min(1, density)})`;
        ctx.beginPath(); ctx.arc(rx, ry, 2.5, 0, Math.PI * 2); ctx.fill();
      }
      setStats(`Ray (${cx},${cy}) → (${Math.round(mx)},${Math.round(my)}) · 12 samples · F_θ(x,y,z,θ,φ) → (RGB,σ)`);
    }
  }, []);

  useEffect(() => { drawScene(-1, -1); }, [drawScene]);

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-destructive mb-3">// Move mouse over scene — cast rays through the volume</p>
      <div className="flex gap-5 flex-wrap items-start">
        <canvas
          ref={canvasRef}
          width={320}
          height={280}
          className="rounded-lg cursor-crosshair shrink-0"
          onMouseMove={(e) => {
            const r = canvasRef.current?.getBoundingClientRect();
            if (r) drawScene(e.clientX - r.left, e.clientY - r.top);
          }}
          onMouseLeave={() => drawScene(-1, -1)}
        />
        <div className="flex-1 min-w-[220px]">
          <div className="rounded-lg bg-muted/30 border-l-[3px] border-destructive p-3 font-mono text-[11px] text-destructive leading-relaxed mb-3">
            F_θ: (x,y,z,θ,φ) → (RGB, density σ)<br />
            C(r) = ∫T(t)σ(r(t))c(r(t),d)dt<br />
            T(t) = exp(-∫σ(r(s))ds)
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed mb-2">NeRF learns a 5D function mapping (position + viewing direction) to (color + density). Novel views synthesized by volume rendering.</p>
          <p className="text-xs text-muted-foreground"><strong className="text-destructive">3D Gaussian Splatting:</strong> 100× faster. G(x) = exp(-½(x-μ)ᵀΣ⁻¹(x-μ))·α</p>
          <p className="mt-2 font-mono text-[10px] text-destructive">{stats}</p>
        </div>
      </div>
    </div>
  );
}
