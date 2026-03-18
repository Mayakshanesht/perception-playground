import { useRef, useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";

// ════════════════════════════════════════════════
// 1. VAE ELBO VISUALIZER
// ════════════════════════════════════════════════

export function VAEElboViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [recon, setRecon] = useState(0.7);
  const [kl, setKl] = useState(0.4);
  const [mu, setMu] = useState(0.3);

  const reconLoss = -(1 - recon) * 100;
  const klLoss = -(0.5 * (mu * mu + kl - Math.log(Math.max(kl, 0.01)) - 1));
  const elbo = reconLoss - klLoss;

  const drawVAE = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0d0916"; ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "rgba(168,85,247,0.2)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(50, H - 40); ctx.lineTo(W - 20, H - 40); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(50, H - 40); ctx.lineTo(50, 20); ctx.stroke();
    ctx.fillStyle = "rgba(168,85,247,0.5)"; ctx.font = "10px monospace";
    ctx.fillText("z", W / 2, H - 8); ctx.fillText("p(z)", 8, H / 2);

    ctx.strokeStyle = "rgba(236,72,153,0.7)"; ctx.lineWidth = 2; ctx.beginPath();
    for (let x = 0; x < W - 70; x++) {
      const z = (x / (W - 70)) * 6 - 3;
      const y = Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI);
      const px = 50 + x, py = H - 40 - y * (H - 80);
      if (x === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    const sigma = Math.sqrt(Math.max(kl, 0.01));
    const mean = mu * 2 - 1;
    ctx.strokeStyle = "rgba(168,85,247,0.9)"; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let x = 0; x < W - 70; x++) {
      const z = (x / (W - 70)) * 6 - 3;
      const y = Math.exp(-((z - mean) ** 2) / (2 * sigma ** 2)) / (sigma * Math.sqrt(2 * Math.PI));
      const px = 50 + x, py = H - 40 - y * (H - 80) * sigma * 0.6;
      if (x === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.font = "10px monospace";
    ctx.fillStyle = "rgba(236,72,153,0.8)"; ctx.fillRect(60, 24, 12, 3); ctx.fillText("Prior N(0,1)", 76, 28);
    ctx.fillStyle = "rgba(168,85,247,0.8)"; ctx.fillRect(60, 36, 12, 3); ctx.fillText("Posterior q_φ(z|x)", 76, 40);

    ctx.fillStyle = "rgba(34,211,238,0.3)"; ctx.fillRect(50, H - 30, (W - 70) * recon, 8);
    ctx.strokeStyle = "rgba(34,211,238,0.5)"; ctx.lineWidth = 1; ctx.strokeRect(50, H - 30, W - 70, 8);
    ctx.fillStyle = "rgba(34,211,238,0.8)"; ctx.font = "9px monospace"; ctx.fillText("reconstruction quality", 52, H - 14);
  }, [recon, kl, mu]);

  useEffect(() => { drawVAE(); }, [drawVAE]);

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-3">// Adjust sliders — watch ELBO components change</p>
      <div className="grid md:grid-cols-2 gap-5">
        <canvas ref={canvasRef} width={380} height={300} className="w-full rounded-lg" />
        <div className="space-y-4">
          <div>
            <label className="flex justify-between text-[11px] font-mono text-muted-foreground mb-1">
              Reconstruction quality β_recon <span className="text-primary">{recon.toFixed(2)}</span>
            </label>
            <input type="range" min="0" max="1" step="0.01" value={recon} onChange={(e) => setRecon(+e.target.value)} className="w-full accent-primary" />
          </div>
          <div>
            <label className="flex justify-between text-[11px] font-mono text-muted-foreground mb-1">
              KL divergence σ² <span className="text-primary">{kl.toFixed(2)}</span>
            </label>
            <input type="range" min="0.01" max="1" step="0.01" value={kl} onChange={(e) => setKl(+e.target.value)} className="w-full accent-primary" />
          </div>
          <div>
            <label className="flex justify-between text-[11px] font-mono text-muted-foreground mb-1">
              Latent μ offset <span className="text-primary">{mu.toFixed(2)}</span>
            </label>
            <input type="range" min="0" max="1" step="0.01" value={mu} onChange={(e) => setMu(+e.target.value)} className="w-full accent-primary" />
          </div>
          <div className="rounded-lg bg-muted/30 border border-border p-3 font-mono text-[11px] leading-[2]">
            <div>ELBO =</div>
            <div className="text-primary">  E[log p(x|z)]</div>
            <div>  = Reconstruction: <span className="text-accent">{reconLoss.toFixed(1)}</span></div>
            <div className="text-primary">− KL[q_φ(z|x) ‖ p(z)]</div>
            <div>  = Regulariser: <span className="text-accent">{(-klLoss).toFixed(1)}</span></div>
            <div>──────────────────</div>
            <div>  ELBO = <span className="text-accent font-bold">{elbo.toFixed(1)}</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════
// 2. GAN MINIMAX SIMULATION
// ════════════════════════════════════════════════

export function GANMinimaxViz() {
  const [gScore, setGScore] = useState(0.3);
  const [dScore, setDScore] = useState(0.8);
  const [iter, setIter] = useState(0);
  const [log, setLog] = useState('Press "Train Step" to simulate one G+D update...');
  const [autoRunning, setAutoRunning] = useState(false);
  const autoRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const step = useCallback(() => {
    setIter((prev) => prev + 1);
    const noise = (Math.random() - 0.5) * 0.08;
    setGScore((g) => {
      const newG = Math.min(0.95, g + 0.04 + noise);
      setDScore((d) => {
        const newD = Math.max(0.5, d - 0.015 + noise * 0.5);
        setLog(`Iter ${iter + 1}: G loss = ${(-Math.log(newG)).toFixed(3)} | D loss = ${(-Math.log(newD) - Math.log(1 - newG)).toFixed(3)}\nD(x) = ${newD.toFixed(3)} on real | D(G(z)) = ${newG.toFixed(3)} on fake\n${iter + 1 > 20 ? "Nash equilibrium approaching: G(z) ≈ p_data, D → 0.5" : "Training..."}`);
        return newD;
      });
      return newG;
    });
  }, [iter]);

  const toggleAuto = useCallback(() => {
    if (autoRunning) {
      if (autoRef.current) clearInterval(autoRef.current);
      setAutoRunning(false);
    } else {
      setAutoRunning(true);
      autoRef.current = setInterval(() => {
        step();
        setIter((i) => { if (i >= 50) { clearInterval(autoRef.current!); setAutoRunning(false); } return i; });
      }, 120);
    }
  }, [autoRunning, step]);

  useEffect(() => () => { if (autoRef.current) clearInterval(autoRef.current); }, []);

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-accent mb-3">// Simulate GAN minimax game: Generator vs Discriminator</p>
      <div className="grid grid-cols-[1fr_auto_1fr] gap-4 items-center">
        <div className="rounded-lg border border-border/50 bg-muted/20 p-4 text-center">
          <p className="text-[10px] font-mono text-primary font-bold mb-2">GENERATOR G</p>
          <div className="h-2 rounded-full bg-muted/40 overflow-hidden mb-2">
            <div className="h-full rounded-full transition-all bg-primary" style={{ width: `${gScore * 100}%` }} />
          </div>
          <p className="text-2xl font-bold text-primary">{gScore.toFixed(2)}</p>
          <p className="text-[9px] font-mono text-muted-foreground">D(G(z)) — fools D?</p>
        </div>
        <div className="flex flex-col items-center gap-2">
          <button onClick={step} className="font-mono text-[10px] px-4 py-2 rounded-md border border-accent text-accent hover:bg-accent hover:text-accent-foreground transition-colors">
            ▶ Train Step
          </button>
          <button onClick={toggleAuto} className="font-mono text-[10px] px-4 py-2 rounded-md border border-accent text-accent hover:bg-accent hover:text-accent-foreground transition-colors">
            {autoRunning ? "⏹ Stop" : "⚡ Auto-train"}
          </button>
          <p className="text-[9px] font-mono text-muted-foreground">Iteration: {iter}</p>
          <div className="font-mono text-[9px] text-muted-foreground text-center leading-relaxed">
            min_G max_D V(G,D):<br />E[log D(x)]<br />+E[log(1-D(G(z)))]
          </div>
        </div>
        <div className="rounded-lg border border-border/50 bg-muted/20 p-4 text-center">
          <p className="text-[10px] font-mono text-accent font-bold mb-2">DISCRIMINATOR D</p>
          <div className="flex gap-1 mb-2 justify-center">
            <div className="w-10 h-10 rounded bg-accent/20 border border-accent/40 flex items-center justify-center text-lg">🖼️</div>
            <div className="w-10 h-10 rounded bg-primary/20 border border-primary/40 flex items-center justify-center text-lg">🎨</div>
          </div>
          <div className="h-2 rounded-full bg-muted/40 overflow-hidden mb-2">
            <div className="h-full rounded-full transition-all bg-accent" style={{ width: `${dScore * 100}%` }} />
          </div>
          <p className="text-2xl font-bold text-accent">{dScore.toFixed(2)}</p>
          <p className="text-[9px] font-mono text-muted-foreground">Accuracy on real vs fake</p>
        </div>
      </div>
      <div className="mt-3 rounded-lg bg-muted/30 border border-border p-3 font-mono text-[10px] text-muted-foreground min-h-[60px] whitespace-pre-wrap leading-relaxed">
        {log}
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════
// 3. DIFFUSION TIMELINE
// ════════════════════════════════════════════════

export function DiffusionTimeline() {
  const [currentStep, setCurrentStep] = useState(0);
  const T = 10;

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-3">// Walk the forward (noise) and reverse (denoise) process</p>
      <div className="flex items-center gap-0 overflow-x-auto pb-3">
        {Array.from({ length: T + 1 }, (_, t) => (
          <div key={t} className="flex items-center">
            <div className="flex flex-col items-center">
              <motion.div
                onClick={() => setCurrentStep(t)}
                whileHover={{ scale: 1.08 }}
                className={`w-[60px] h-[60px] rounded-lg flex items-center justify-center cursor-pointer border-2 transition-all relative overflow-hidden ${
                  currentStep === t ? "border-primary shadow-lg" : "border-transparent"
                }`}
                style={{ background: `hsl(var(--muted) / ${0.3 + (t / T) * 0.5})` }}
              >
                <div className="relative w-full h-full">
                  <div className="absolute inset-0 flex items-center justify-center" style={{ opacity: Math.max(0.05, 1 - (t / T) * 1.1) }}>
                    {[0, 1, 2, 3, 4, 5].map((i) => (
                      <div
                        key={i}
                        className="absolute w-2 h-2 rounded-full"
                        style={{
                          background: `hsl(${280 + i * 30}, 70%, 60%)`,
                          transform: `translate(${Math.cos((i / 6) * Math.PI * 2) * 10}px, ${Math.sin((i / 6) * Math.PI * 2) * 10}px)`,
                        }}
                      />
                    ))}
                    <div className="w-3 h-3 rounded-full bg-primary/80" />
                  </div>
                  <div className="absolute inset-0" style={{ opacity: (t / T) * 0.7, background: "repeating-conic-gradient(hsl(var(--muted-foreground) / 0.3) 0%, transparent 1%)" }} />
                </div>
              </motion.div>
              <span className="text-[8px] font-mono text-muted-foreground mt-1">
                {t === 0 ? "x₀ (clean)" : t === T ? "xT (noise)" : `t=${t}`}
              </span>
            </div>
            {t < T && <span className="text-muted-foreground text-sm px-0.5 mt-[-16px]">→</span>}
          </div>
        ))}
      </div>
      <div className="flex gap-3 items-center flex-wrap mt-2">
        <button onClick={() => setCurrentStep((s) => Math.min(T, s + 1))} className="font-mono text-[10px] px-4 py-1.5 rounded-md bg-destructive/20 border border-destructive text-destructive hover:bg-destructive/30 transition-colors">
          → Add Noise
        </button>
        <button onClick={() => setCurrentStep((s) => Math.max(0, s - 1))} className="font-mono text-[10px] px-4 py-1.5 rounded-md bg-primary/20 border border-primary text-primary hover:bg-primary/30 transition-colors">
          ← Denoise
        </button>
        <span className="text-[10px] font-mono text-muted-foreground">
          Step t={currentStep} · noise_level={(currentStep / T).toFixed(1)} · ᾱ_t={Math.exp(-currentStep * 0.02).toFixed(3)}
        </span>
      </div>
      <div className="mt-3 rounded-lg bg-muted/30 border-l-[3px] border-primary p-3 font-mono text-[11px] text-primary leading-relaxed">
        Forward: q(x_t|x_{"{t-1}"}) = N(x_t; √(1−β_t)·x_{"{t-1}"}, β_t·I)<br />
        Closed-form: x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε, ε ~ N(0,I)<br />
        Reverse: p_θ(x_{"{t-1}"}|x_t) = N(x_{"{t-1}"}; μ_θ(x_t,t), β_t·I)<br />
        Training: L = E[‖ε − ε_θ(√ᾱ_t·x₀+√(1−ᾱ_t)·ε, t)‖²]
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════
// 4. STABLE DIFFUSION PIPELINE + CFG
// ════════════════════════════════════════════════

const sdBlocks = [
  { label: "📝 Text Encoder", color: "hsl(265 70% 60%)", tip: "CLIP/T5 text encoder. 77-token context → conditioning vectors c ∈ ℝ^{77×768}." },
  { label: "🗜️ VAE Encoder", color: "hsl(var(--primary))", tip: "Compress 512×512×3 → 64×64×4 latent. 8× spatial compression. KL regularized." },
  { label: "🔁 U-Net ×T", color: "hsl(340 75% 55%)", tip: "Operates in latent space 64×64×4. Cross-attention injects text. T=1000, DDIM 50 steps." },
  { label: "🔭 VAE Decoder", color: "hsl(var(--primary))", tip: "Upsample 64×64×4 latent → 512×512×3 pixel image." },
  { label: "🖼️ Output", color: "hsl(187 85% 53%)", tip: "512×512 (SD 1.x) or 1024 (SDXL). ControlNet adds structural conditioning." },
];

export function StableDiffusionPipelineViz() {
  const [hovered, setHovered] = useState<number | null>(null);
  const [cfg, setCfg] = useState(7.5);

  const quality = cfg < 3 ? "Weak guidance — prompt ignored" :
    cfg < 6 ? "Mild guidance — diverse outputs" :
    cfg <= 10 ? "Recommended — text-image balance" :
    cfg <= 15 ? "High guidance — less diversity" :
    "Too high — over-saturation, artifacts";

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-4">// Hover each stage — Stable Diffusion pipeline</p>
      <div className="flex items-center gap-0 overflow-x-auto pb-2 mb-3">
        {sdBlocks.map((block, i) => (
          <div key={i} className="flex items-center">
            <motion.div
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
              whileHover={{ y: -4 }}
              className="min-w-[95px] px-2.5 py-3 rounded-lg text-center font-mono text-[10px] font-bold border cursor-pointer transition-all"
              style={{
                background: `${block.color.replace(")", " / 0.15)")}`,
                borderColor: `${block.color.replace(")", " / 0.5)")}`,
                color: block.color,
              }}
            >
              {block.label}
            </motion.div>
            {i < sdBlocks.length - 1 && <span className="text-muted-foreground text-lg px-1">→</span>}
          </div>
        ))}
      </div>
      {hovered !== null && (
        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-xs font-mono text-muted-foreground mb-3">
          {sdBlocks[hovered].tip}
        </motion.p>
      )}

      <p className="text-[10px] font-mono uppercase tracking-wider text-accent mb-3 mt-4">// Classifier-Free Guidance — adjust scale</p>
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <label className="flex justify-between text-[11px] font-mono text-muted-foreground mb-1">
            Guidance scale ω <span className="text-primary">{cfg.toFixed(1)}</span>
          </label>
          <input type="range" min="1" max="20" step="0.5" value={cfg} onChange={(e) => setCfg(+e.target.value)} className="w-full accent-primary" />
          <div className="mt-3 rounded-lg bg-muted/30 border border-border p-3 font-mono text-[11px] leading-[2]">
            <div>ε_guided = ε_uncond</div>
            <div className="text-primary">  + ω·(ε_cond − ε_uncond)</div>
            <div>──────────────────</div>
            <div>ω=1: <span className="text-muted-foreground">no guidance</span></div>
            <div>ω=7.5: <span className="text-primary">good text-image alignment</span></div>
            <div>ω=20: <span className="text-destructive">over-saturated, artifacts</span></div>
            <div>Current ω={cfg.toFixed(1)} → <span className="text-primary font-bold">{quality}</span></div>
          </div>
        </div>
        <div className="rounded-lg bg-muted/20 border border-border p-4 flex items-center justify-center">
          <div className="w-full h-[160px] relative">
            <svg viewBox="0 0 200 100" className="w-full h-full">
              <path
                d={Array.from({ length: 100 }, (_, i) => {
                  const w = (i / 100) * 20;
                  const q = Math.exp(-((w - 7.5) ** 2) / 8) * 0.9 + 0.05 * (1 - Math.exp(-w));
                  return `${i === 0 ? "M" : "L"} ${i * 2} ${100 - q * 80}`;
                }).join(" ")}
                fill="none"
                stroke="hsl(340 75% 55%)"
                strokeWidth="2"
              />
              {(() => {
                const cx = (cfg / 20) * 200;
                const q = Math.exp(-((cfg - 7.5) ** 2) / 8) * 0.9 + 0.05 * (1 - Math.exp(-cfg));
                const cy = 100 - q * 80;
                return <circle cx={cx} cy={cy} r="4" fill="hsl(187 85% 53%)" />;
              })()}
              <text x="5" y="12" fill="hsl(var(--muted-foreground))" fontSize="8" fontFamily="monospace">quality</text>
              <text x="150" y="95" fill="hsl(var(--muted-foreground))" fontSize="8" fontFamily="monospace">ω →</text>
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════
// 5. NOISE SCHEDULE COMPARISON
// ════════════════════════════════════════════════

export function NoiseScheduleViz() {
  const [schedule, setSchedule] = useState<"linear" | "cosine" | "scaled-linear">("linear");
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const getAlphaBar = useCallback((t: number, type: string): number => {
    if (type === "linear") {
      const beta = 1e-4 + t * (0.02 - 1e-4);
      return Math.exp(-t * (1e-4 + 0.5 * t * (0.02 - 1e-4)) * 1000);
    } else if (type === "cosine") {
      const s = 0.008;
      const f = Math.cos(((t + s) / (1 + s)) * Math.PI / 2);
      const f0 = Math.cos((s / (1 + s)) * Math.PI / 2);
      return Math.max(0.001, (f * f) / (f0 * f0));
    } else {
      const beta = Math.sqrt(1e-4 + t * (0.02 - 1e-4));
      return Math.exp(-t * beta * 500);
    }
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const ox = 50, oy = 20, gw = W - 70, gh = H - 60;

    // Axes
    ctx.strokeStyle = "hsl(var(--muted-foreground) / 0.3)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox, oy + gh); ctx.lineTo(ox + gw, oy + gh); ctx.stroke();

    ctx.fillStyle = "hsl(var(--muted-foreground))";
    ctx.font = "9px monospace"; ctx.textAlign = "center";
    ctx.fillText("timestep t →", ox + gw / 2, H - 5);
    ctx.save(); ctx.translate(10, oy + gh / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("ᾱₜ (signal)", 0, 0); ctx.restore();

    // Draw all three for comparison
    const schedules: { id: string; color: string; label: string }[] = [
      { id: "linear", color: "hsl(340 75% 55%)", label: "Linear" },
      { id: "cosine", color: "hsl(187 85% 53%)", label: "Cosine" },
      { id: "scaled-linear", color: "hsl(32 95% 55%)", label: "Scaled Linear" },
    ];

    schedules.forEach(({ id, color }) => {
      const isActive = id === schedule;
      ctx.strokeStyle = color;
      ctx.lineWidth = isActive ? 3 : 1;
      ctx.globalAlpha = isActive ? 1 : 0.3;
      ctx.beginPath();
      for (let i = 0; i <= gw; i++) {
        const t = i / gw;
        const alpha = getAlphaBar(t, id);
        const px = ox + i;
        const py = oy + gh - alpha * gh;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    });

    // SNR annotation at midpoint
    const midT = 0.5;
    const alpha = getAlphaBar(midT, schedule);
    const snr = alpha / (1 - alpha + 1e-8);
    ctx.fillStyle = "hsl(var(--primary))";
    ctx.font = "10px monospace"; ctx.textAlign = "left";
    ctx.fillText(`SNR(t=0.5) = ᾱ/(1-ᾱ) = ${snr.toFixed(2)}`, ox + 5, oy + 15);
  }, [schedule, getAlphaBar]);

  useEffect(() => { draw(); }, [draw]);

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-3">// Compare noise schedules — how signal decays over time</p>
      <div className="flex gap-2 mb-3">
        {(["linear", "cosine", "scaled-linear"] as const).map((s) => (
          <button key={s} onClick={() => setSchedule(s)}
            className={`font-mono text-[10px] px-3 py-1.5 rounded-md border transition-all ${
              schedule === s ? "bg-primary/20 border-primary text-primary" : "border-border text-muted-foreground hover:border-primary/40"
            }`}>{s}</button>
        ))}
      </div>
      <canvas ref={canvasRef} width={480} height={240} className="w-full rounded-lg" />
      <div className="mt-3 rounded-lg bg-muted/30 border border-border p-3 font-mono text-[10px] text-muted-foreground leading-relaxed">
        <strong className="text-foreground">Linear:</strong> β_t ∈ [1e-4, 0.02] — fast signal decay, wastes steps near t=T.{" "}
        <strong className="text-foreground">Cosine:</strong> ᾱ_t = cos²((t/T+s)/(1+s) · π/2) — smoother, more uniform SNR.{" "}
        <strong className="text-foreground">Scaled Linear:</strong> β_t = (√β_min + t·(√β_max−√β_min))² — SDXL default.
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════
// 6. CONTROLNET ARCHITECTURE VISUALIZER
// ════════════════════════════════════════════════

const controlTypes = [
  { id: "canny", label: "🔲 Canny Edge", desc: "Edge detection → structural control. Preserves composition while allowing style changes.", condition: "Binary edges from Canny detector" },
  { id: "depth", label: "🏔️ Depth Map", desc: "MiDaS depth estimation → 3D-aware generation. Foreground/background preserved.", condition: "Monocular depth from MiDaS/ZoeDepth" },
  { id: "pose", label: "🤸 OpenPose", desc: "Skeleton keypoints → pose-guided generation. Body proportions and position controlled.", condition: "18-joint skeleton from OpenPose" },
  { id: "seg", label: "🎨 Segmentation", desc: "Semantic map → region-aware generation. Each color → material/object class.", condition: "Semantic label map (ADE20K classes)" },
];

export function ControlNetViz() {
  const [activeType, setActiveType] = useState(0);
  const [zeroConvScale, setZeroConvScale] = useState(0);
  const ct = controlTypes[activeType];

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-3">// ControlNet — zero-convolution architecture</p>
      <div className="flex gap-2 mb-4 flex-wrap">
        {controlTypes.map((c, i) => (
          <button key={c.id} onClick={() => setActiveType(i)}
            className={`font-mono text-[10px] px-3 py-1.5 rounded-md border transition-all ${
              activeType === i ? "bg-primary/20 border-primary text-primary" : "border-border text-muted-foreground hover:border-primary/40"
            }`}>{c.label}</button>
        ))}
      </div>

      {/* Architecture diagram */}
      <div className="grid md:grid-cols-[2fr_1fr] gap-4 mb-4">
        <div className="space-y-3">
          <div className="flex items-center gap-0 overflow-x-auto pb-1">
            {[
              { label: "Condition\n(edge/depth)", color: "hsl(32 95% 55%)" },
              { label: "Zero Conv\n(init=0)", color: "hsl(340 75% 55%)" },
              { label: "Trainable\nEncoder Copy", color: "hsl(265 70% 60%)" },
              { label: "Zero Conv\n(output)", color: "hsl(340 75% 55%)" },
              { label: "⊕ Locked\nU-Net", color: "hsl(187 85% 53%)" },
            ].map((block, i) => (
              <div key={i} className="flex items-center">
                <div
                  className="min-w-[85px] px-2 py-2.5 rounded-lg text-center font-mono text-[9px] font-bold border whitespace-pre-line"
                  style={{
                    background: `${block.color.replace(")", " / 0.12)")}`,
                    borderColor: `${block.color.replace(")", " / 0.4)")}`,
                    color: block.color,
                  }}
                >{block.label}</div>
                {i < 4 && <span className="text-muted-foreground text-sm px-0.5">→</span>}
              </div>
            ))}
          </div>
          <div className="rounded-lg bg-muted/30 border-l-[3px] border-primary p-3 font-mono text-[11px] text-primary leading-relaxed">
            y = F_locked(x) + zero_conv(F_trainable(x, c))<br />
            At init: zero_conv weights = 0 → output = F_locked(x)<br />
            Training gradually increases ControlNet contribution
          </div>
        </div>
        <div className="space-y-3">
          <div>
            <label className="flex justify-between text-[10px] font-mono text-muted-foreground mb-1">
              Zero-Conv Scale (training progress) <span className="text-primary">{zeroConvScale.toFixed(2)}</span>
            </label>
            <input type="range" min="0" max="1" step="0.01" value={zeroConvScale} onChange={(e) => setZeroConvScale(+e.target.value)} className="w-full accent-primary" />
          </div>
          <div className="rounded-lg border border-border bg-muted/20 p-3 text-center">
            <div className="flex gap-2 items-center justify-center mb-2">
              <div className="w-16 h-16 rounded bg-muted/40 border border-border flex items-center justify-center text-xl">🖼️</div>
              <span className="text-muted-foreground text-xs">+</span>
              <div className="w-16 h-16 rounded border border-border flex items-center justify-center text-xl"
                style={{ background: `hsl(32 95% 55% / ${0.1 + zeroConvScale * 0.3})`, borderColor: `hsl(32 95% 55% / ${0.3 + zeroConvScale * 0.5})` }}>
                {ct.id === "canny" ? "🔲" : ct.id === "depth" ? "🏔️" : ct.id === "pose" ? "🤸" : "🎨"}
              </div>
            </div>
            <p className="text-[9px] font-mono text-muted-foreground">
              ControlNet weight: {zeroConvScale < 0.1 ? "0 (pure SD)" : zeroConvScale < 0.5 ? "low (subtle)" : zeroConvScale < 0.9 ? "medium (balanced)" : "high (strong control)"}
            </p>
          </div>
        </div>
      </div>
      <p className="text-xs text-muted-foreground leading-relaxed">
        <strong className="text-foreground">{ct.label}:</strong> {ct.desc} <span className="text-primary">Input: {ct.condition}</span>
      </p>
    </div>
  );
}

// ════════════════════════════════════════════════
// 7. VIDEO GENERATION PIPELINE VISUALIZER
// ════════════════════════════════════════════════

const videoModels = [
  {
    name: "Stable Video Diffusion",
    pipeline: [
      { label: "Image I₀", color: "hsl(var(--primary))" },
      { label: "CLIP Embed\n+ Noise Aug", color: "hsl(265 70% 60%)" },
      { label: "3D U-Net\n(Spatial+Temporal)", color: "hsl(340 75% 55%)" },
      { label: "Temporal\nAttention", color: "hsl(32 95% 55%)" },
      { label: "VAE Decode\nPer Frame", color: "hsl(187 85% 53%)" },
    ],
    math: "z_t^{1:F} = Denoise(z_T^{1:F}, c_img, t)\nTemporal Attn: Q_f = z_f W^Q, K/V = [z_1...z_F] W^{K,V}\nSpatial layers shared, temporal layers inserted",
    notes: "Image-to-video: conditions on single frame. 3D U-Net factorizes spatial (2D conv) and temporal (1D conv + temporal attention) processing. 14-25 frames at 576×1024.",
  },
  {
    name: "Sora (DiT)",
    pipeline: [
      { label: "Text +\nOptional Image", color: "hsl(265 70% 60%)" },
      { label: "Spacetime\nPatchify", color: "hsl(var(--primary))" },
      { label: "DiT Blocks\n(Transformer)", color: "hsl(340 75% 55%)" },
      { label: "Spacetime\nDe-patchify", color: "hsl(32 95% 55%)" },
      { label: "Variable Res\nVideo Output", color: "hsl(187 85% 53%)" },
    ],
    math: "Patches: p_{t,h,w} = video[t:t+τ, h:h+P, w:w+P]\nDiT: z' = z + Attn(LN(z, γ, β)) where γ,β=f(t,c)\nNative resolution: no cropping/resizing needed",
    notes: "Treats video as spacetime patches, not separate frames. Diffusion Transformer (DiT) replaces U-Net. Trains on variable duration/resolution. Emergent 3D consistency from scale.",
  },
  {
    name: "AnimateDiff",
    pipeline: [
      { label: "Frozen\nSD U-Net", color: "hsl(var(--primary))" },
      { label: "Motion\nModule", color: "hsl(340 75% 55%)" },
      { label: "Temporal\nSelf-Attn", color: "hsl(32 95% 55%)" },
      { label: "LoRA\n(Optional)", color: "hsl(265 70% 60%)" },
      { label: "16-Frame\nVideo", color: "hsl(187 85% 53%)" },
    ],
    math: "Motion Module: Attn(z_f, [z_1...z_F], [z_1...z_F])\nInserted after each spatial attention block\nFrozen SD weights + trained temporal layers only",
    notes: "Plug-and-play: adds temporal attention to any SD checkpoint/LoRA. Train only the motion module on video data. Compatible with personalized models (DreamBooth, LoRA).",
  },
];

export function VideoGenPipelineViz() {
  const [active, setActive] = useState(0);
  const model = videoModels[active];

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-3">// Video Generation Architectures — temporal diffusion models</p>
      <div className="flex gap-2 mb-4 flex-wrap">
        {videoModels.map((m, i) => (
          <button key={m.name} onClick={() => setActive(i)}
            className={`font-mono text-[10px] px-4 py-1.5 rounded-full border transition-all ${
              active === i ? "bg-primary border-primary text-primary-foreground" : "border-border text-muted-foreground hover:border-primary/40"
            }`}>{m.name}</button>
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
            >{block.label}</div>
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
// 8. SCORE FUNCTION & TWEEDIE VISUALIZER
// ════════════════════════════════════════════════

export function ScoreFunctionViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [noiseLevel, setNoiseLevel] = useState(0.3);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0d0916"; ctx.fillRect(0, 0, W, H);

    const ox = 50, oy = 20, gw = W - 70, gh = H - 50;

    // Axes
    ctx.strokeStyle = "rgba(168,85,247,0.2)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ox, oy + gh / 2); ctx.lineTo(ox + gw, oy + gh / 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ox + gw / 2, oy); ctx.lineTo(ox + gw / 2, oy + gh); ctx.stroke();

    // Data distribution (mixture of 2 Gaussians)
    const sigma = 0.1 + noiseLevel * 0.8;
    const mu1 = -1.2, mu2 = 1.0;

    // Draw noisy distribution
    ctx.strokeStyle = "rgba(236,72,153,0.7)"; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i <= gw; i++) {
      const x = (i / gw) * 6 - 3;
      const p1 = Math.exp(-((x - mu1) ** 2) / (2 * sigma ** 2));
      const p2 = Math.exp(-((x - mu2) ** 2) / (2 * sigma ** 2));
      const y = (p1 + p2) / (sigma * Math.sqrt(2 * Math.PI) * 2);
      const py = oy + gh / 2 - y * gh * 0.4;
      if (i === 0) ctx.moveTo(ox + i, py); else ctx.lineTo(ox + i, py);
    }
    ctx.stroke();

    // Score arrows (∇_x log p(x))
    ctx.strokeStyle = "rgba(34,211,238,0.7)"; ctx.lineWidth = 1.5;
    for (let i = 0; i < 20; i++) {
      const x = (i / 20) * 6 - 3;
      const px = ox + ((x + 3) / 6) * gw;
      const py = oy + gh / 2;

      // Compute score
      const p1 = Math.exp(-((x - mu1) ** 2) / (2 * sigma ** 2));
      const p2 = Math.exp(-((x - mu2) ** 2) / (2 * sigma ** 2));
      const dp1 = -(x - mu1) / (sigma ** 2) * p1;
      const dp2 = -(x - mu2) / (sigma ** 2) * p2;
      const score = (dp1 + dp2) / (p1 + p2 + 1e-8);
      const arrowLen = Math.min(Math.abs(score) * 15, 30) * Math.sign(score);

      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + arrowLen, py);
      ctx.stroke();

      // Arrowhead
      if (Math.abs(arrowLen) > 3) {
        ctx.beginPath();
        ctx.moveTo(px + arrowLen, py);
        ctx.lineTo(px + arrowLen - Math.sign(arrowLen) * 4, py - 3);
        ctx.lineTo(px + arrowLen - Math.sign(arrowLen) * 4, py + 3);
        ctx.fill();
        ctx.fillStyle = "rgba(34,211,238,0.7)";
      }
    }

    // Labels
    ctx.fillStyle = "rgba(236,72,153,0.8)"; ctx.font = "10px monospace"; ctx.textAlign = "left";
    ctx.fillText(`p_σ(x) — noised data (σ=${sigma.toFixed(2)})`, ox + 5, oy + 12);
    ctx.fillStyle = "rgba(34,211,238,0.8)";
    ctx.fillText("→ score: ∇_x log p(x) — points toward data", ox + 5, oy + 25);
  }, [noiseLevel]);

  useEffect(() => { draw(); }, [draw]);

  return (
    <div className="rounded-xl border border-border bg-card/60 p-5 mb-4">
      <p className="text-[10px] font-mono uppercase tracking-wider text-primary mb-3">// Score function — gradient of log-density points toward data modes</p>
      <div className="grid md:grid-cols-2 gap-4">
        <canvas ref={canvasRef} width={420} height={220} className="w-full rounded-lg" />
        <div className="space-y-3">
          <div>
            <label className="flex justify-between text-[10px] font-mono text-muted-foreground mb-1">
              Noise level σ <span className="text-primary">{(0.1 + noiseLevel * 0.8).toFixed(2)}</span>
            </label>
            <input type="range" min="0" max="1" step="0.01" value={noiseLevel} onChange={(e) => setNoiseLevel(+e.target.value)} className="w-full accent-primary" />
          </div>
          <div className="rounded-lg bg-muted/30 border border-border p-3 font-mono text-[10px] text-muted-foreground leading-relaxed">
            <p><strong className="text-foreground">Score matching:</strong></p>
            <p className="text-primary">s_θ(x,t) ≈ ∇_x log p_t(x)</p>
            <p className="mt-1"><strong className="text-foreground">Tweedie's formula:</strong></p>
            <p className="text-primary">E[x₀|xₜ] = (xₜ + σₜ² · ∇ log p(xₜ)) / ᾱₜ</p>
            <p className="mt-1 text-accent">Connection: ε_θ = -σₜ · s_θ</p>
          </div>
        </div>
      </div>
    </div>
  );
}
