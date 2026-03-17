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

    // Axes
    ctx.strokeStyle = "rgba(168,85,247,0.2)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(50, H - 40); ctx.lineTo(W - 20, H - 40); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(50, H - 40); ctx.lineTo(50, 20); ctx.stroke();
    ctx.fillStyle = "rgba(168,85,247,0.5)"; ctx.font = "10px monospace";
    ctx.fillText("z", W / 2, H - 8); ctx.fillText("p(z)", 8, H / 2);

    // Prior N(0,1)
    ctx.strokeStyle = "rgba(236,72,153,0.7)"; ctx.lineWidth = 2; ctx.beginPath();
    for (let x = 0; x < W - 70; x++) {
      const z = (x / (W - 70)) * 6 - 3;
      const y = Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI);
      const px = 50 + x, py = H - 40 - y * (H - 80);
      if (x === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Posterior
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

    // Legend
    ctx.font = "10px monospace";
    ctx.fillStyle = "rgba(236,72,153,0.8)"; ctx.fillRect(60, 24, 12, 3); ctx.fillText("Prior N(0,1)", 76, 28);
    ctx.fillStyle = "rgba(168,85,247,0.8)"; ctx.fillRect(60, 36, 12, 3); ctx.fillText("Posterior q_φ(z|x)", 76, 40);

    // Recon quality bar
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
        {/* Generator */}
        <div className="rounded-lg border border-border/50 bg-muted/20 p-4 text-center">
          <p className="text-[10px] font-mono text-primary font-bold mb-2">GENERATOR G</p>
          <div className="h-2 rounded-full bg-muted/40 overflow-hidden mb-2">
            <div className="h-full rounded-full transition-all bg-primary" style={{ width: `${gScore * 100}%` }} />
          </div>
          <p className="text-2xl font-bold text-primary">{gScore.toFixed(2)}</p>
          <p className="text-[9px] font-mono text-muted-foreground">D(G(z)) — fools D?</p>
        </div>

        {/* Middle controls */}
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

        {/* Discriminator */}
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
                {/* Flower pattern with noise overlay */}
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
            {/* Simple quality curve visualization */}
            <svg viewBox="0 0 200 100" className="w-full h-full">
              {/* Quality curve */}
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
              {/* Current position */}
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
