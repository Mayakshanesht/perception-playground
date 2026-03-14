import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";

// ── Utility helpers ──
function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }
function clamp(v: number, a: number, b: number) { return Math.max(a, Math.min(b, v)); }
function rnd(a: number, b: number) { return a + Math.random() * (b - a); }

// ── Canvas-safe theme colors (from CSS vars) ──
const C = {
  primary: "hsla(168, 80%, 58%, 1)",
  primaryDim: "hsla(168, 80%, 58%, 0.5)",
  primaryFaint: "hsla(168, 80%, 58%, 0.15)",
  primaryGhost: "hsla(168, 80%, 58%, 0.06)",
  accent: "hsla(38, 92%, 60%, 1)",
  accentDim: "hsla(38, 92%, 60%, 0.5)",
  accentFaint: "hsla(38, 92%, 60%, 0.15)",
  green: "hsla(160, 80%, 55%, 1)",
  greenDim: "hsla(160, 80%, 55%, 0.5)",
  greenFaint: "hsla(160, 80%, 55%, 0.15)",
  red: "hsla(0, 70%, 60%, 1)",
  redDim: "hsla(0, 70%, 60%, 0.5)",
  redFaint: "hsla(0, 70%, 60%, 0.08)",
  purple: "hsla(265, 70%, 60%, 1)",
  purpleDim: "hsla(265, 70%, 60%, 0.5)",
  bg: "hsla(222, 47%, 8%, 0.9)",
  bgSolid: "hsla(222, 47%, 8%, 1)",
  fg: "hsla(210, 40%, 96%, 1)",
  muted: "hsla(222, 28%, 15%, 1)",
  mutedDim: "hsla(222, 28%, 15%, 0.3)",
  mutedFg: "hsla(215, 20%, 60%, 1)",
  border: "hsla(222, 24%, 20%, 1)",
  card: "hsla(222, 44%, 11%, 0.3)",
};

function useCanvas(draw: (ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => void, height = 420) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tRef = useRef(0);
  const animRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const w = parent.clientWidth;
      canvas.width = w * (window.devicePixelRatio || 1);
      canvas.height = height * (window.devicePixelRatio || 1);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${height}px`;
      ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
    };
    resize();
    window.addEventListener("resize", resize);

    const frame = () => {
      tRef.current += 0.012;
      const w = canvas.width / (window.devicePixelRatio || 1);
      const h = canvas.height / (window.devicePixelRatio || 1);
      ctx.save();
      ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
      draw(ctx, w, h, tRef.current);
      ctx.restore();
      animRef.current = requestAnimationFrame(frame);
    };
    animRef.current = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [draw, height]);

  return canvasRef;
}

function CanvasStage({ children, label, modes, activeMode, onModeChange, hint }: {
  children: React.ReactNode;
  label: string;
  modes?: { id: string; label: string }[];
  activeMode?: string;
  onModeChange?: (id: string) => void;
  hint?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="rounded-lg border border-border bg-card/30 overflow-hidden"
    >
      <div className="relative">
        {children}
        <div className="absolute top-2 left-3 px-2 py-0.5 rounded bg-background/90 border border-border">
          <span className="text-[9px] font-mono font-semibold uppercase tracking-widest text-primary">{label}</span>
        </div>
        {hint && (
          <div className="absolute top-2 right-3 text-[9px] font-mono text-muted-foreground">{hint}</div>
        )}
      </div>
      {modes && modes.length > 0 && (
        <div className="flex gap-1.5 flex-wrap items-center p-3 border-t border-border bg-background/80">
          {modes.map(m => (
            <button
              key={m.id}
              onClick={() => onModeChange?.(m.id)}
              className={`px-2.5 py-1 rounded text-[9px] font-mono uppercase tracking-wider border transition-all ${
                activeMode === m.id
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border bg-card text-muted-foreground hover:border-primary/40 hover:text-foreground"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      )}
    </motion.div>
  );
}

// ══════════════════════════════════════════════════════════════
// 1. FEATURE MATCHING
// ══════════════════════════════════════════════════════════════
const NUM_KP = 30;
const featureKeypoints = Array.from({ length: NUM_KP }, () => ({
  x: rnd(30, 400), y: rnd(30, 380),
  scale: rnd(4, 16), angle: rnd(0, Math.PI * 2),
  response: rnd(0.2, 1),
  color: `hsla(${rnd(160, 200)}, 80%, 55%, 0.8)`,
}));
const featureKeypointsR = featureKeypoints.map(kp => ({
  ...kp, x: kp.x + rnd(-20, 20), y: kp.y + rnd(-8, 8),
}));

export function FeatureMatchingCanvas() {
  const [mode, setMode] = useState("detect");
  const drawRef = useRef(mode);
  drawRef.current = mode;

  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.clearRect(0, 0, w, h);
    const m = drawRef.current;

    if (m === "detect") {
      // Two image panels with keypoints
      const imgW = (w - 60) / 2, imgH = h - 60;
      ctx.fillStyle = "hsl(var(--muted) / 0.3)";
      ctx.fillRect(20, 30, imgW, imgH);
      ctx.fillRect(w / 2 + 10, 30, imgW, imgH);
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1;
      ctx.strokeRect(20, 30, imgW, imgH);
      ctx.strokeRect(w / 2 + 10, 30, imgW, imgH);

      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "9px monospace"; ctx.textAlign = "center";
      ctx.fillText("Image 1", 20 + imgW / 2, 22);
      ctx.fillText("Image 2", w / 2 + 10 + imgW / 2, 22);

      const visible = Math.min(Math.floor(t * 2) % (NUM_KP + 5), NUM_KP);
      for (let i = 0; i < visible; i++) {
        const kp = featureKeypoints[i];
        const nx = 20 + (kp.x / 440) * imgW;
        const ny = 30 + (kp.y / 400) * imgH;
        const pulse = i === visible - 1 ? (Math.sin(t * 8) * 0.5 + 0.5) : 0;

        ctx.save();
        ctx.translate(nx, ny);
        ctx.rotate(kp.angle);
        ctx.strokeStyle = kp.color;
        ctx.lineWidth = i === visible - 1 ? 2 : 1;
        if (i === visible - 1) { ctx.shadowColor = kp.color; ctx.shadowBlur = 8 + pulse * 6; }
        ctx.beginPath(); ctx.arc(0, 0, kp.scale * 0.5, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(kp.scale * 0.6, 0); ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.restore();

        // Right image keypoint
        const kpR = featureKeypointsR[i];
        const rx = w / 2 + 10 + (kpR.x / 440) * imgW;
        const ry = 30 + (kpR.y / 400) * imgH;
        ctx.save(); ctx.translate(rx, ry); ctx.rotate(kpR.angle);
        ctx.strokeStyle = kp.color; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(0, 0, kp.scale * 0.5, 0, Math.PI * 2); ctx.stroke();
        ctx.restore();
      }

      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "9px monospace"; ctx.textAlign = "left";
      ctx.fillText(`SIFT keypoints: ${visible}/${NUM_KP}`, 20, h - 6);
    } else if (m === "scale") {
      // Scale space / DoG pyramid
      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "10px monospace"; ctx.textAlign = "center";
      ctx.fillText("Gaussian Pyramid → Difference of Gaussians", w / 2, 18);

      const octaves = 3, levels = 4;
      const baseW = Math.min(80, (w - 100) / (octaves * levels));
      const baseH = h * 0.5;

      for (let oct = 0; oct < octaves; oct++) {
        const octScale = Math.pow(2, oct);
        for (let lv = 0; lv < levels; lv++) {
          const cw = baseW / octScale;
          const ch = baseH / octScale;
          const x = 30 + (oct * levels + lv) * (baseW + 8);
          const y = h / 2 - ch / 2;
          const sigma = Math.pow(2, oct) * Math.pow(1.414, lv);
          const blur = Math.min(sigma * 1.5, 8);

          ctx.fillStyle = `hsla(${180 + lv * 20}, 60%, ${30 + lv * 10}%, 0.6)`;
          ctx.fillRect(x, y, cw, ch);
          ctx.strokeStyle = "hsl(var(--primary) / 0.4)"; ctx.lineWidth = 1;
          ctx.strokeRect(x, y, cw, ch);
          ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "7px monospace"; ctx.textAlign = "center";
          ctx.fillText(`σ=${sigma.toFixed(1)}`, x + cw / 2, y + ch + 10);

          if (lv < levels - 1) {
            const pulse = (Math.sin(t * 3 + lv + oct * 4) + 1) / 2;
            ctx.fillStyle = `hsla(40, 95%, 50%, ${0.3 + pulse * 0.5})`;
            ctx.fillRect(x + cw + 1, y, 3, ch);
          }
        }
      }
      ctx.fillStyle = "hsla(40, 95%, 50%, 0.8)"; ctx.font = "8px monospace"; ctx.textAlign = "right";
      ctx.fillText("█ DoG = L(kσ) − L(σ)", w - 20, h - 8);
    } else if (m === "match") {
      const imgW = (w - 60) / 2, imgH = h - 60;
      ctx.fillStyle = "hsl(var(--muted) / 0.3)";
      ctx.fillRect(20, 30, imgW, imgH);
      ctx.fillRect(w / 2 + 10, 30, imgW, imgH);
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1;
      ctx.strokeRect(20, 30, imgW, imgH);
      ctx.strokeRect(w / 2 + 10, 30, imgW, imgH);

      const visMatches = Math.floor((t * 0.5) % NUM_KP) + 3;
      for (let i = 0; i < Math.min(visMatches, NUM_KP); i++) {
        const kp = featureKeypoints[i];
        const kpR = featureKeypointsR[i];
        const lx = 20 + (kp.x / 440) * imgW;
        const ly = 30 + (kp.y / 400) * imgH;
        const rx = w / 2 + 10 + (kpR.x / 440) * imgW;
        const ry = 30 + (kpR.y / 400) * imgH;
        const isNew = i === visMatches - 1;
        const isGood = kp.response > 0.5;

        ctx.save();
        ctx.globalAlpha = isNew ? 1 : 0.35;
        ctx.strokeStyle = isGood ? "hsl(160, 80%, 55%)" : "hsla(0, 70%, 60%, 0.5)";
        ctx.lineWidth = isGood ? 1.5 : 0.8;
        if (!isGood) ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(rx, ry); ctx.stroke();
        ctx.setLineDash([]);

        ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2);
        ctx.fillStyle = isGood ? "hsl(160, 80%, 55%)" : "hsla(0, 70%, 60%, 0.5)"; ctx.fill();
        ctx.beginPath(); ctx.arc(rx, ry, 3, 0, Math.PI * 2); ctx.fill();

        if (isNew) {
          const ratio = (isGood ? rnd(0.45, 0.7) : rnd(0.78, 0.95)).toFixed(2);
          const midX = (lx + rx) / 2, midY = (ly + ry) / 2;
          ctx.fillStyle = "hsl(var(--background) / 0.9)";
          ctx.beginPath(); (ctx as any).roundRect(midX - 28, midY - 10, 56, 18, 2); ctx.fill();
          ctx.fillStyle = parseFloat(ratio) < 0.75 ? "hsl(160, 80%, 55%)" : "hsl(0, 70%, 60%)";
          ctx.font = "9px monospace"; ctx.textAlign = "center";
          ctx.fillText(`r=${ratio}`, midX, midY + 3);
        }
        ctx.restore();
      }

      ctx.fillStyle = "hsl(160, 80%, 55%)"; ctx.font = "8px monospace"; ctx.textAlign = "left";
      ctx.fillText("— accepted (r<0.75)", 20, h - 6);
      ctx.fillStyle = "hsl(0, 70%, 60%)";
      ctx.fillText("--- rejected (r≥0.75)", 160, h - 6);
    } else if (m === "ransac") {
      const imgW = (w - 60) / 2, imgH = h - 60;
      ctx.fillStyle = "hsl(var(--muted) / 0.3)";
      ctx.fillRect(20, 30, imgW, imgH);
      ctx.fillRect(w / 2 + 10, 30, imgW, imgH);

      const iter = Math.floor(t * 0.5) % 50;
      for (let i = 0; i < NUM_KP; i++) {
        const kp = featureKeypoints[i];
        const kpR = featureKeypointsR[i];
        const lx = 20 + (kp.x / 440) * imgW;
        const ly = 30 + (kp.y / 400) * imgH;
        const rx = w / 2 + 10 + (kpR.x / 440) * imgW;
        const ry = 30 + (kpR.y / 400) * imgH;
        const isInlier = i < 22;

        ctx.save(); ctx.globalAlpha = 0.6;
        ctx.strokeStyle = isInlier ? "hsl(160, 80%, 55%)" : "hsla(0, 70%, 60%, 0.3)";
        ctx.lineWidth = isInlier ? 1.5 : 0.7;
        if (!isInlier) ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(rx, ry); ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2);
        ctx.fillStyle = isInlier ? "hsl(160, 80%, 55%)" : "hsla(0, 70%, 60%, 0.5)"; ctx.fill();
        ctx.beginPath(); ctx.arc(rx, ry, 3, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
      }

      ctx.fillStyle = "hsl(var(--background) / 0.9)";
      ctx.beginPath(); (ctx as any).roundRect(w / 2 - 130, h - 70, 260, 58, 4); ctx.fill();
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "10px monospace"; ctx.textAlign = "center";
      ctx.fillText(`RANSAC iter: ${iter + 1}/50`, w / 2, h - 52);
      ctx.fillStyle = "hsl(160, 80%, 55%)";
      ctx.fillText(`Inliers: 22/${NUM_KP} (${((22 / NUM_KP) * 100).toFixed(0)}%)`, w / 2, h - 36);
      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace";
      ctx.fillText("Sampson dist threshold: 1.0px", w / 2, h - 20);
    }
  };

  const canvasRef = useCanvas(draw, 420);
  return (
    <CanvasStage
      label="Feature Detection & Matching"
      modes={[
        { id: "detect", label: "Keypoint Detection" },
        { id: "scale", label: "Scale Space (DoG)" },
        { id: "match", label: "Ratio Test Matching" },
        { id: "ransac", label: "RANSAC Filtering" },
      ]}
      activeMode={mode}
      onModeChange={setMode}
      hint="animated"
    >
      <canvas ref={canvasRef} className="w-full block" />
    </CanvasStage>
  );
}

// ══════════════════════════════════════════════════════════════
// 2. EPIPOLAR GEOMETRY
// ══════════════════════════════════════════════════════════════
export function EpipolarGeometryCanvas() {
  const [mode, setMode] = useState("basic");
  const drawRef = useRef(mode);
  drawRef.current = mode;

  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.clearRect(0, 0, w, h);
    const m = drawRef.current;

    if (m === "basic") {
      const lCamX = w * 0.2, rCamX = w * 0.8, camY = h * 0.78;

      // Cameras
      [{ cx: lCamX, col: "hsl(var(--primary))", lbl: "Left" },
       { cx: rCamX, col: "hsla(40, 95%, 50%, 1)", lbl: "Right" }].forEach(({ cx, col, lbl }) => {
        ctx.save();
        ctx.shadowColor = col; ctx.shadowBlur = 10;
        ctx.fillStyle = col.replace("1)", "0.15)"); ctx.strokeStyle = col; ctx.lineWidth = 1.5;
        ctx.beginPath(); (ctx as any).roundRect(cx - 28, camY - 14, 56, 28, 4); ctx.fill(); ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.fillStyle = col; ctx.font = "bold 9px monospace"; ctx.textAlign = "center";
        ctx.fillText(lbl, cx, camY + 3);
        ctx.restore();
      });

      // Baseline
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(lCamX, camY); ctx.lineTo(rCamX, camY); ctx.stroke();
      ctx.setLineDash([]);

      // 3D point
      const ptX = w / 2 + Math.cos(t * 0.6) * 120;
      const ptY = h * 0.25 + Math.sin(t * 0.4) * 50;

      // Rays
      ctx.strokeStyle = "hsl(var(--primary) / 0.5)"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(lCamX, camY - 14); ctx.lineTo(ptX, ptY); ctx.stroke();
      ctx.strokeStyle = "hsla(40, 95%, 50%, 0.5)";
      ctx.beginPath(); ctx.moveTo(rCamX, camY - 14); ctx.lineTo(ptX, ptY); ctx.stroke();

      // Epipolar line
      ctx.save();
      ctx.strokeStyle = "hsla(40, 95%, 50%, 0.8)"; ctx.lineWidth = 1.5;
      ctx.shadowColor = "hsla(40, 95%, 50%, 1)"; ctx.shadowBlur = 6;
      const projY = camY - 45;
      ctx.beginPath(); ctx.moveTo(rCamX - 38, projY - 6); ctx.lineTo(rCamX + 38, projY - 3); ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "hsla(40, 95%, 50%, 1)"; ctx.font = "8px monospace"; ctx.textAlign = "right";
      ctx.fillText("epipolar line l′=Fx", rCamX + 38, projY - 14);
      ctx.restore();

      // 3D point
      ctx.save();
      ctx.shadowColor = "#fff"; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(ptX, ptY, 7, 0, Math.PI * 2);
      ctx.fillStyle = "#fff"; ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "hsl(var(--foreground))"; ctx.font = "bold 9px monospace"; ctx.textAlign = "center";
      ctx.fillText("X ∈ ℝ³", ptX, ptY - 13);
      ctx.restore();

      // Formula
      ctx.fillStyle = "hsl(var(--background) / 0.9)";
      ctx.beginPath(); (ctx as any).roundRect(w / 2 - 110, h - 62, 220, 48, 4); ctx.fill();
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "12px monospace"; ctx.textAlign = "center";
      ctx.fillText("x′ᵀ F x = 0", w / 2, h - 42);
      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace";
      ctx.fillText("Fe = 0,  Fᵀe′ = 0  (epipoles)", w / 2, h - 22);
    } else if (m === "decompose") {
      const cx = w / 2, cy = h / 2;

      // SVD header
      ctx.fillStyle = "hsl(var(--background) / 0.9)";
      ctx.beginPath(); (ctx as any).roundRect(cx - 150, 16, 300, 50, 4); ctx.fill();
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "11px monospace"; ctx.textAlign = "center";
      ctx.fillText("E = U · diag(1,1,0) · Vᵀ", cx, 38);
      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "9px monospace";
      ctx.fillText("4 (R,t) candidate solutions", cx, 54);

      const solutions = [
        { label: "(R₁, +t)", valid: true },
        { label: "(R₁, −t)", valid: false },
        { label: "(R₂, +t)", valid: false },
        { label: "(R₂, −t)", valid: false },
      ];
      const positions = [
        [cx - 140, cy - 50], [cx + 60, cy - 50],
        [cx - 140, cy + 60], [cx + 60, cy + 60],
      ];

      solutions.forEach((sol, i) => {
        const [px, py] = positions[i];
        const pulse = sol.valid ? (Math.sin(t * 3) * 0.5 + 0.5) : 0;
        ctx.save();
        ctx.shadowColor = sol.valid ? "hsl(160, 80%, 55%)" : "hsl(0, 70%, 60%)";
        ctx.shadowBlur = sol.valid ? 12 + pulse * 8 : 3;
        ctx.fillStyle = sol.valid ? "hsla(160, 80%, 55%, 0.1)" : "hsla(0, 70%, 60%, 0.06)";
        ctx.strokeStyle = sol.valid ? "hsl(160, 80%, 55%)" : "hsla(0, 70%, 60%, 0.4)";
        ctx.lineWidth = sol.valid ? 2 : 1;
        ctx.beginPath(); (ctx as any).roundRect(px - 55, py - 30, 110, 60, 5); ctx.fill(); ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.fillStyle = sol.valid ? "hsl(160, 80%, 55%)" : "hsla(0, 70%, 60%, 0.6)";
        ctx.font = "bold 10px monospace"; ctx.textAlign = "center";
        ctx.fillText(sol.label, px, py - 8);
        ctx.fillStyle = sol.valid ? "hsl(var(--foreground))" : "hsl(var(--muted-foreground))";
        ctx.font = "9px monospace";
        ctx.fillText(sol.valid ? "✓ positive depth" : "✗ behind camera", px, py + 8);
        if (sol.valid) {
          ctx.fillStyle = "hsl(160, 80%, 55%)"; ctx.font = "bold 8px monospace";
          ctx.fillText("← SELECTED", px, py + 22);
        }
        ctx.restore();
      });

      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace"; ctx.textAlign = "center";
      ctx.fillText("Disambiguate: check Z > 0 in both cameras", cx, h - 10);
    } else if (m === "rectify") {
      const imgW = (w - 80) / 2, imgH = h * 0.55;
      const lx = 20, rx = w / 2 + 20, iy = (h - imgH) / 2 - 10;

      // Before
      ctx.fillStyle = "hsl(var(--muted) / 0.2)";
      ctx.beginPath(); (ctx as any).roundRect(lx, iy, imgW - 10, imgH, 4); ctx.fill();
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "9px monospace"; ctx.textAlign = "center";
      ctx.fillText("BEFORE Rectification", lx + (imgW - 10) / 2, iy - 6);

      for (let i = 0; i < 8; i++) {
        const tt = i / 7;
        const y0 = iy + imgH * 0.1 + tt * imgH * 0.8;
        const skew = (tt - 0.5) * 35;
        const active = Math.floor(t * 1.5) % 8 === i;
        ctx.save(); ctx.globalAlpha = active ? 0.9 : 0.3;
        ctx.strokeStyle = active ? "hsla(40, 95%, 50%, 1)" : "hsla(40, 95%, 50%, 0.4)";
        ctx.lineWidth = active ? 2 : 0.8;
        ctx.beginPath(); ctx.moveTo(lx + 10, y0 - skew); ctx.lineTo(lx + imgW - 20, y0 + skew); ctx.stroke();
        ctx.restore();
      }

      // After
      ctx.fillStyle = "hsl(var(--muted) / 0.2)";
      ctx.beginPath(); (ctx as any).roundRect(rx, iy, imgW - 10, imgH, 4); ctx.fill();
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "hsl(160, 80%, 55%)"; ctx.font = "9px monospace"; ctx.textAlign = "center";
      ctx.fillText("AFTER Rectification", rx + (imgW - 10) / 2, iy - 6);

      for (let i = 0; i < 8; i++) {
        const tt = i / 7;
        const y0 = iy + imgH * 0.1 + tt * imgH * 0.8;
        const active = Math.floor(t * 1.5) % 8 === i;
        ctx.save(); ctx.globalAlpha = active ? 0.9 : 0.35;
        ctx.strokeStyle = active ? "hsl(160, 80%, 55%)" : "hsla(160, 80%, 55%, 0.4)";
        ctx.lineWidth = active ? 2 : 0.8;
        ctx.beginPath(); ctx.moveTo(rx + 10, y0); ctx.lineTo(rx + imgW - 20, y0); ctx.stroke();
        ctx.restore();
      }

      // H, H' arrow
      ctx.fillStyle = "hsl(var(--background) / 0.9)";
      const arrowY = iy + imgH / 2;
      ctx.beginPath(); (ctx as any).roundRect(w / 2 - 40, arrowY - 16, 80, 32, 4); ctx.fill();
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "bold 10px monospace"; ctx.textAlign = "center";
      ctx.fillText("H, H′", w / 2, arrowY - 1);
      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "7px monospace";
      ctx.fillText("homographies", w / 2, arrowY + 10);

      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace"; ctx.textAlign = "center";
      ctx.fillText("After rectification: disparity is purely horizontal (1D)", w / 2, h - 8);
    }
  };

  const canvasRef = useCanvas(draw, 400);
  return (
    <CanvasStage
      label="Epipolar Geometry"
      modes={[
        { id: "basic", label: "Fundamental Matrix" },
        { id: "decompose", label: "E → R, t" },
        { id: "rectify", label: "Rectification" },
      ]}
      activeMode={mode}
      onModeChange={setMode}
      hint="animated"
    >
      <canvas ref={canvasRef} className="w-full block" />
    </CanvasStage>
  );
}

// ══════════════════════════════════════════════════════════════
// 3. SfM PIPELINE
// ══════════════════════════════════════════════════════════════
const sfmPoints3D = Array.from({ length: 200 }, (_, i) => ({
  x: rnd(-2, 2), y: rnd(-1.5, 1.5), z: rnd(1, 8),
  color: `hsl(${180 + i * 2}, 70%, ${40 + rnd(-10, 20)}%)`,
  size: rnd(1.5, 3.5),
}));
const NUM_CAMS = 8;
const sfmCameras = Array.from({ length: NUM_CAMS }, (_, i) => {
  const angle = (i / NUM_CAMS) * Math.PI * 2;
  return { x: Math.cos(angle) * 3.5, y: rnd(-0.3, 0.3), z: Math.sin(angle) * 3.5 + 4, angle };
});

function project3D(x: number, y: number, z: number, rotY: number, scale: number, cx: number, cy: number) {
  const cosR = Math.cos(rotY), sinR = Math.sin(rotY);
  const rx = x * cosR - z * sinR, rz = x * sinR + z * cosR;
  return { x: cx + rx * scale, y: cy - y * scale * 0.6, z: rz };
}

export function SfMPipelineCanvas() {
  const [mode, setMode] = useState("pipeline");
  const drawRef = useRef(mode);
  drawRef.current = mode;

  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.clearRect(0, 0, w, h);
    const m = drawRef.current;
    const rotY = t * 0.3;
    const cx = w / 2, cy = h * 0.52;
    const scale = 55;

    if (m === "pipeline" || m === "sparse") {
      // Floor grid
      ctx.save(); ctx.globalAlpha = 0.1;
      for (let x = -4; x <= 4; x++) {
        const p1 = project3D(x, -1.5, -1, rotY, scale, cx, cy);
        const p2 = project3D(x, -1.5, 9, rotY, scale, cx, cy);
        ctx.strokeStyle = "hsl(var(--primary))"; ctx.lineWidth = 0.4;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      }
      ctx.restore();

      // Points
      const visibleCams = m === "sparse" ? NUM_CAMS : Math.floor((t * 0.3) % NUM_CAMS) + 2;
      const sorted = [...sfmPoints3D].map(p => {
        const proj = project3D(p.x, p.y, p.z, rotY, scale, cx, cy);
        return { ...p, px: proj.x, py: proj.y, pz: proj.z };
      }).sort((a, b) => a.pz - b.pz);

      sorted.forEach(p => {
        const depthA = clamp(0.3 + (p.pz + 0.5) * 0.15, 0.2, 1);
        ctx.globalAlpha = depthA;
        ctx.beginPath(); ctx.arc(p.px, p.py, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color; ctx.fill();
      });
      ctx.globalAlpha = 1;

      // Cameras
      sfmCameras.slice(0, Math.min(visibleCams, NUM_CAMS)).forEach((cam, i) => {
        const proj = project3D(cam.x, cam.y, cam.z, rotY, scale, cx, cy);
        const isNew = m === "pipeline" && i === visibleCams - 1;
        const col = isNew ? "hsla(40, 95%, 50%, 1)" : "hsl(var(--primary))";
        ctx.save();
        ctx.shadowColor = col; ctx.shadowBlur = isNew ? 14 : 5;
        ctx.fillStyle = col.replace(/1\)$/, "0.15)").replace(")", " / 0.15)");
        ctx.strokeStyle = col; ctx.lineWidth = 1.5;
        ctx.beginPath(); (ctx as any).roundRect(proj.x - 12, proj.y - 8, 24, 16, 3); ctx.fill(); ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.fillStyle = col; ctx.font = "7px monospace"; ctx.textAlign = "center";
        ctx.fillText(`C${i + 1}`, proj.x, proj.y + 3);
        ctx.restore();
      });

      // Info
      ctx.fillStyle = "hsl(var(--background) / 0.85)";
      ctx.beginPath(); (ctx as any).roundRect(12, 12, 200, 48, 4); ctx.fill();
      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "9px monospace"; ctx.textAlign = "left";
      ctx.fillText(`Cameras: ${Math.min(visibleCams, NUM_CAMS)} / ${NUM_CAMS}`, 22, 30);
      ctx.fillStyle = "hsla(40, 95%, 50%, 1)";
      ctx.fillText(`3D points: ${Math.floor(sfmPoints3D.length * (visibleCams / NUM_CAMS))}`, 22, 44);
    } else if (m === "ba") {
      // Bundle adjustment — reprojection error animation
      const epoch = Math.floor(t * 0.6) % 20;
      const errorScale = Math.exp(-epoch * 0.15);

      const pts = [
        { x: -1.2, y: 0.4, z: 3 }, { x: 0.3, y: -0.6, z: 2.5 },
        { x: 1.1, y: 0.2, z: 3.5 }, { x: -0.4, y: 0.9, z: 2.2 },
      ];
      const cams = [
        { x: -2, y: 0, z: 0 }, { x: 2, y: 0, z: 0 }, { x: 0, y: 0, z: -1 },
      ];

      cams.forEach((cam, ci) => {
        const cp = project3D(cam.x, cam.y, cam.z, rotY, scale, cx, cy);
        ctx.save();
        ctx.shadowColor = "hsl(var(--primary))"; ctx.shadowBlur = 6;
        ctx.fillStyle = "hsl(var(--primary) / 0.15)"; ctx.strokeStyle = "hsl(var(--primary))"; ctx.lineWidth = 1.5;
        ctx.beginPath(); (ctx as any).roundRect(cp.x - 12, cp.y - 8, 24, 16, 3); ctx.fill(); ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "7px monospace"; ctx.textAlign = "center";
        ctx.fillText(`C${ci + 1}`, cp.x, cp.y + 3);
        ctx.restore();

        pts.forEach((pt, pi) => {
          const pp = project3D(pt.x, pt.y, pt.z, rotY, scale, cx, cy);
          const obsX = pp.x + Math.sin(t * 3 + pi + ci * 2) * 8 * errorScale;
          const obsY = pp.y + Math.cos(t * 2.5 + pi * 2 + ci) * 6 * errorScale;

          ctx.save(); ctx.globalAlpha = 0.6;
          ctx.strokeStyle = "hsl(0, 70%, 60%)"; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(obsX, obsY); ctx.lineTo(pp.x, pp.y); ctx.stroke();
          ctx.beginPath(); ctx.arc(obsX, obsY, 3, 0, Math.PI * 2);
          ctx.fillStyle = "hsla(40, 95%, 50%, 1)"; ctx.fill();
          ctx.beginPath(); ctx.arc(pp.x, pp.y, 3, 0, Math.PI * 2);
          ctx.fillStyle = "hsl(160, 80%, 55%)"; ctx.fill();
          ctx.restore();
        });
      });

      pts.forEach(pt => {
        const pp = project3D(pt.x, pt.y, pt.z, rotY, scale, cx, cy);
        ctx.beginPath(); ctx.arc(pp.x, pp.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = "#fff"; ctx.fill();
      });

      // Info
      ctx.fillStyle = "hsl(var(--background) / 0.9)";
      ctx.beginPath(); (ctx as any).roundRect(12, 12, w - 24, 65, 4); ctx.fill();
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "hsla(40, 95%, 50%, 1)"; ctx.font = "10px monospace"; ctx.textAlign = "left";
      ctx.fillText(`Bundle Adjustment — Iteration: ${epoch}/20`, 22, 30);

      const totalErr = (15 * errorScale).toFixed(2);
      const barMaxW = w - 180;
      ctx.fillStyle = "hsl(var(--muted) / 0.5)";
      ctx.beginPath(); (ctx as any).roundRect(22, 40, barMaxW, 12, 2); ctx.fill();
      ctx.fillStyle = "hsl(0, 70%, 60%)";
      ctx.beginPath(); (ctx as any).roundRect(22, 40, barMaxW * errorScale, 12, 2); ctx.fill();
      ctx.fillStyle = "hsl(var(--foreground))"; ctx.font = "9px monospace"; ctx.textAlign = "right";
      ctx.fillText(`RMS error: ${totalErr}px`, w - 22, 50);

      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace"; ctx.textAlign = "left";
      ctx.fillText("● observed  ● reprojected  │ error", 22, 68);
      ctx.fillStyle = "hsla(40, 95%, 50%, 1)"; ctx.textAlign = "right";
      ctx.fillText("(JᵀJ+λI)δ = −Jᵀr", w - 22, 68);
    } else if (m === "triangulate") {
      const cam1 = project3D(-2, 0, 0, rotY, scale, cx, cy);
      const cam2 = project3D(2, 0, 0, rotY, scale, cx, cy);
      const ptX = Math.sin(t * 0.5) * 1.5;
      const ptY = Math.sin(t * 0.3) * 0.8 - 0.5;
      const ptZ = 3 + Math.cos(t * 0.4);
      const pt3d = project3D(ptX, ptY, ptZ, rotY, scale, cx, cy);

      [{ c: cam1, col: "hsl(var(--primary))" }, { c: cam2, col: "hsla(40, 95%, 50%, 1)" }].forEach(({ c, col }) => {
        ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.7;
        ctx.shadowColor = col; ctx.shadowBlur = 5;
        const dx = pt3d.x - c.x, dy = pt3d.y - c.y;
        ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(pt3d.x + dx * 0.3, pt3d.y + dy * 0.3); ctx.stroke();
        ctx.shadowBlur = 0; ctx.restore();
      });

      // Mid-point
      ctx.save(); ctx.shadowColor = "#fff"; ctx.shadowBlur = 14;
      ctx.beginPath(); ctx.arc(pt3d.x, pt3d.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#fff"; ctx.fill(); ctx.shadowBlur = 0; ctx.restore();

      // Camera bodies
      [{ c: cam1, col: "hsl(var(--primary))", lbl: "C₁" }, { c: cam2, col: "hsla(40, 95%, 50%, 1)", lbl: "C₂" }].forEach(({ c, col, lbl }) => {
        ctx.save(); ctx.shadowColor = col; ctx.shadowBlur = 8;
        ctx.fillStyle = col.replace(/[^,]+\)$/, "0.15)"); ctx.strokeStyle = col; ctx.lineWidth = 2;
        ctx.beginPath(); (ctx as any).roundRect(c.x - 16, c.y - 10, 32, 20, 4); ctx.fill(); ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.fillStyle = col; ctx.font = "bold 9px monospace"; ctx.textAlign = "center";
        ctx.fillText(lbl, c.x, c.y + 3); ctx.restore();
      });

      ctx.fillStyle = "hsl(var(--background) / 0.9)";
      ctx.beginPath(); (ctx as any).roundRect(12, 12, 280, 72, 4); ctx.fill();
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "10px monospace"; ctx.textAlign = "left";
      ctx.fillText("DLT Triangulation:", 22, 30);
      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "9px monospace";
      ctx.fillText("x₁ × (P₁X) = 0 → 2 equations/view", 22, 46);
      ctx.fillStyle = "hsla(40, 95%, 50%, 1)";
      ctx.fillText("X* = Vₙ (last col of SVD(A))", 22, 60);
      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace";
      ctx.fillText("|reproj error| < 1px", 22, 74);
    }
  };

  const canvasRef = useCanvas(draw, 420);
  return (
    <CanvasStage
      label="SfM Pipeline — COLMAP"
      modes={[
        { id: "pipeline", label: "Full Pipeline" },
        { id: "triangulate", label: "Triangulation" },
        { id: "ba", label: "Bundle Adjustment" },
        { id: "sparse", label: "Sparse Cloud" },
      ]}
      activeMode={mode}
      onModeChange={setMode}
      hint="auto-rotate"
    >
      <canvas ref={canvasRef} className="w-full block" />
    </CanvasStage>
  );
}

// ══════════════════════════════════════════════════════════════
// 4. MVS
// ══════════════════════════════════════════════════════════════
const sceneW = 32, sceneH = 18;
const sceneDepth = Array.from({ length: sceneH * sceneW }, (_, i) => {
  const r = Math.floor(i / sceneW) / sceneH, c = (i % sceneW) / sceneW;
  return clamp(
    0.2 + 0.5 * Math.exp(-((c - 0.35) ** 2 + (r - 0.45) ** 2) / 0.04) +
    0.3 * Math.exp(-((c - 0.72) ** 2 + (r - 0.55) ** 2) / 0.025) +
    0.1 * (1 - r) * 0.5, 0, 1
  );
});

function depthColor(d: number, a = 1) {
  // Turbo-inspired colormap: blue → cyan → green → yellow → red
  const t = clamp(d, 0, 1);
  const h = lerp(240, 0, t);        // hue: blue(240) → red(0)
  const s = lerp(85, 90, Math.abs(t - 0.5) * 2); // saturation stays high
  const l = lerp(55, 50, t);        // lightness
  return `hsla(${h}, ${s}%, ${l}%, ${a})`;
}

export function MVSCanvas() {
  const [mode, setMode] = useState("sweep");
  const drawRef = useRef(mode);
  drawRef.current = mode;

  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.clearRect(0, 0, w, h);
    const m = drawRef.current;

    if (m === "sweep") {
      const DMAX = 16;
      const currentD = Math.floor(t * 1.5) % DMAX;
      const cellW = (w - 40) / sceneW, cellH = (h * 0.45) / sceneH;

      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "9px monospace"; ctx.textAlign = "center";
      ctx.fillText(`Reference Image + Cost Volume slice at d=${currentD}`, w / 2, 16);

      for (let r = 0; r < sceneH; r++) {
        for (let c = 0; c < sceneW; c++) {
          const trueD = sceneDepth[r * sceneW + c] * DMAX;
          const cost = Math.abs(Math.sin((currentD - trueD) * 0.4)) * 0.8 + 0.05;
          ctx.fillStyle = depthColor(cost, 0.9);
          ctx.fillRect(20 + c * cellW, 26 + r * cellH, cellW - 0.5, cellH - 0.5);
        }
      }

      ctx.strokeStyle = "hsl(var(--primary) / 0.5)"; ctx.lineWidth = 1.5;
      ctx.strokeRect(20, 26, sceneW * cellW, sceneH * cellH);

      // Plane indicator
      const planeX = 20 + currentD * (sceneW * cellW / DMAX);
      ctx.fillStyle = "hsl(var(--primary) / 0.12)";
      ctx.fillRect(planeX, 26, sceneW * cellW / DMAX, sceneH * cellH);
      ctx.strokeStyle = "hsl(var(--primary))"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(planeX, 26); ctx.lineTo(planeX, 26 + sceneH * cellH); ctx.stroke();

      // Cost bar chart
      const cvY = 26 + sceneH * cellH + 24, cvH = h - (cvY + 24);
      const midDepth = sceneDepth[9 * sceneW + 10] * DMAX;

      ctx.fillStyle = "hsl(var(--muted) / 0.3)";
      ctx.beginPath(); (ctx as any).roundRect(20, cvY, sceneW * cellW, cvH, 4); ctx.fill();

      for (let d = 0; d < DMAX; d++) {
        const cost = Math.abs(Math.sin((d - midDepth) * 0.4)) * 0.85 + 0.05;
        const bx = 20 + d * (sceneW * cellW / DMAX);
        const bh = cost * (cvH - 16);
        const isCurrent = d === currentD;
        ctx.fillStyle = isCurrent ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.25)";
        if (isCurrent) { ctx.shadowColor = "hsl(var(--primary))"; ctx.shadowBlur = 6; }
        ctx.fillRect(bx + 1, cvY + cvH - bh - 8, (sceneW * cellW / DMAX) - 2, bh);
        ctx.shadowBlur = 0;
      }

      ctx.fillStyle = "hsla(40, 95%, 50%, 1)"; ctx.font = "8px monospace"; ctx.textAlign = "center";
      const bestD = Math.round(midDepth);
      const bestX = 20 + bestD * (sceneW * cellW / DMAX);
      ctx.fillText(`▲ d*=${bestD}`, bestX + sceneW * cellW / DMAX / 2, cvY + cvH - 2);
    } else if (m === "patchmatch") {
      const cellW = (w - 40) / sceneW, cellH = (h * 0.6) / sceneH;
      const iteration = Math.floor(t * 0.5) % 4;
      const labels = ["Init: random depth", "Propagate from neighbors", "Random refinement", "Final depth map"];
      const noiseScale = Math.pow(0.5, iteration) * 0.4;

      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "9px monospace"; ctx.textAlign = "center";
      ctx.fillText(`PatchMatch — Step ${iteration + 1}: ${labels[iteration]}`, w / 2, 16);

      for (let r = 0; r < sceneH; r++) {
        for (let c = 0; c < sceneW; c++) {
          const trueD = sceneDepth[r * sceneW + c];
          const noise = iteration === 0 ? rnd(-0.4, 0.4) : noiseScale * (Math.sin(r * 7 + c * 3) * 0.5);
          ctx.fillStyle = depthColor(clamp(trueD + noise, 0, 1), 0.92);
          ctx.fillRect(20 + c * cellW, 26 + r * cellH, cellW - 0.5, cellH - 0.5);
        }
      }

      if (iteration === 1) {
        const ac = Math.floor(t * 4) % sceneW, ar = Math.floor(t * 2) % sceneH;
        const px = 20 + ac * cellW + cellW / 2, py = 26 + ar * cellH + cellH / 2;
        [[-1, 0], [1, 0], [0, -1], [0, 1]].forEach(([dc, dr]) => {
          ctx.strokeStyle = "hsla(40, 95%, 50%, 0.8)"; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + dc * cellW, py + dr * cellH); ctx.stroke();
        });
        ctx.save(); ctx.shadowColor = "hsla(40, 95%, 50%, 1)"; ctx.shadowBlur = 8;
        ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2);
        ctx.fillStyle = "hsla(40, 95%, 50%, 1)"; ctx.fill(); ctx.shadowBlur = 0; ctx.restore();
      }

      const mse = (noiseScale * 0.15).toFixed(4);
      ctx.fillStyle = "hsl(var(--background) / 0.9)";
      ctx.beginPath(); (ctx as any).roundRect(20, 26 + sceneH * cellH + 8, w - 40, 40, 4); ctx.fill();
      ctx.fillStyle = "hsl(160, 80%, 55%)"; ctx.font = "9px monospace"; ctx.textAlign = "left";
      ctx.fillText(`Depth MAE: ${mse} | Iter: ${iteration + 1}/3`, 30, 26 + sceneH * cellH + 26);
      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace";
      ctx.fillText("PatchMatch ≈ exhaustive sweep at O(1/64) cost", 30, 26 + sceneH * cellH + 40);
    } else if (m === "fusion") {
      const cellW = (w - 40) / sceneW, cellH = (h * 0.6) / sceneH;

      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "9px monospace"; ctx.textAlign = "center";
      ctx.fillText("Fused Dense Depth Map → Point Cloud", w / 2, 16);

      const anim = Math.sin(t * 0.8) * 0.5 + 0.5;
      for (let r = 0; r < sceneH; r++) {
        for (let c = 0; c < sceneW; c++) {
          const d = sceneDepth[r * sceneW + c];
          const pulse = Math.sin(r * 0.5 + c * 0.3 + t * 2) * 0.03 * anim;
          ctx.fillStyle = depthColor(clamp(d + pulse, 0, 1), 0.95);
          ctx.fillRect(20 + c * cellW, 26 + r * cellH, cellW - 0.5, cellH - 0.5);
        }
      }

      // Colorbar
      const cbX = w - 30, cbY = 26;
      for (let i = 0; i < sceneH * cellH; i++) {
        ctx.fillStyle = depthColor(i / (sceneH * cellH));
        ctx.fillRect(cbX, cbY + i, 10, 1);
      }
      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "7px monospace"; ctx.textAlign = "left";
      ctx.fillText("near", cbX + 12, cbY + 8); ctx.fillText("far", cbX + 12, cbY + sceneH * cellH - 4);

      ctx.fillStyle = "hsl(var(--background) / 0.9)";
      ctx.beginPath(); (ctx as any).roundRect(20, 26 + sceneH * cellH + 8, w - 50, 40, 4); ctx.fill();
      ctx.fillStyle = "hsla(40, 95%, 50%, 1)"; ctx.font = "9px monospace"; ctx.textAlign = "left";
      ctx.fillText("D_fused(p) = median{Dᵢ(πᵢ(Xₚ))}", 30, 26 + sceneH * cellH + 26);
      ctx.fillStyle = "hsl(160, 80%, 55%)"; ctx.font = "8px monospace";
      ctx.fillText("→ Poisson/Delaunay meshing for surface", 30, 26 + sceneH * cellH + 40);
    }
  };

  const canvasRef = useCanvas(draw, 380);
  return (
    <CanvasStage
      label="Multi-View Stereo (MVS)"
      modes={[
        { id: "sweep", label: "Plane Sweep" },
        { id: "patchmatch", label: "PatchMatch" },
        { id: "fusion", label: "Depth Fusion" },
      ]}
      activeMode={mode}
      onModeChange={setMode}
      hint="animated"
    >
      <canvas ref={canvasRef} className="w-full block" />
    </CanvasStage>
  );
}

// ══════════════════════════════════════════════════════════════
// 5. NeRF
// ══════════════════════════════════════════════════════════════
function sceneQuery(x: number, y: number, z: number) {
  const d1 = Math.exp(-((x - 0.2) ** 2 + (y + 0.1) ** 2 + (z - 3) ** 2) * 4);
  const d2 = Math.exp(-((x + 0.4) ** 2 + (y - 0.2) ** 2 + (z - 4.5) ** 2) * 3);
  const sigma = Math.max(d1, d2) * 8;
  const tt = d1 / (d1 + d2 + 1e-6);
  const r = lerp(0.1, 0.9, tt), g = lerp(0.6, 0.4, tt), b = lerp(0.9, 0.1, tt);
  return { sigma, r, g, b };
}

export function NeRFCanvas() {
  const [mode, setMode] = useState("ray");
  const drawRef = useRef(mode);
  drawRef.current = mode;

  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.clearRect(0, 0, w, h);
    const m = drawRef.current;

    if (m === "ray") {
      const NUM_RAYS = 9;
      const camX = w * 0.1, camY = h * 0.5;

      // Camera
      ctx.save();
      ctx.shadowColor = "hsl(var(--primary))"; ctx.shadowBlur = 10;
      ctx.fillStyle = "hsl(var(--primary) / 0.12)"; ctx.strokeStyle = "hsl(var(--primary))"; ctx.lineWidth = 2;
      ctx.beginPath(); (ctx as any).roundRect(camX - 30, camY - 18, 40, 36, 4); ctx.fill(); ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "8px monospace"; ctx.textAlign = "center";
      ctx.fillText("Camera", camX - 10, camY + 3); ctx.restore();

      // Image plane
      const planeX = camX + 24;
      ctx.fillStyle = "hsl(var(--primary) / 0.06)";
      ctx.fillRect(planeX, camY - 50, 3, 100);
      ctx.strokeStyle = "hsl(var(--primary) / 0.3)"; ctx.lineWidth = 1;
      ctx.strokeRect(planeX, camY - 50, 3, 100);

      const activeRay = Math.floor(t * 1.5) % NUM_RAYS;
      for (let ri = 0; ri < NUM_RAYS; ri++) {
        const py = camY - 50 + ri * (100 / NUM_RAYS) + 6;
        const angle = (py - camY) * 0.008;
        const isActive = ri === activeRay;

        ctx.save(); ctx.globalAlpha = isActive ? 1 : 0.15;
        // Sample points along ray
        for (let si = 0; si < 40; si++) {
          const tt = si / 40;
          const rx = camX + tt * (w * 0.7);
          const ry = py + tt * (w * 0.7) * Math.sin(angle);
          const sx = (rx - w * 0.5) / 100, sy = (ry - h * 0.5) / 100, sz = tt * 6 + 1;
          const q = sceneQuery(sx, sy, sz);
          if (isActive && q.sigma > 0.1) {
            const cr = Math.floor(q.r * 255), cg = Math.floor(q.g * 255), cb = Math.floor(q.b * 255);
            ctx.beginPath(); ctx.arc(rx, ry, 2 + q.sigma * 2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${cr},${cg},${cb},${clamp(q.sigma * 0.5, 0, 0.8)})`;
            ctx.shadowColor = `rgb(${cr},${cg},${cb})`; ctx.shadowBlur = 4;
            ctx.fill(); ctx.shadowBlur = 0;
          }
        }

        // Ray line
        ctx.strokeStyle = isActive ? "hsl(var(--primary) / 0.6)" : "hsl(var(--primary) / 0.15)";
        ctx.lineWidth = isActive ? 1.5 : 0.5;
        ctx.beginPath(); ctx.moveTo(camX + 12, py);
        ctx.lineTo(camX + 12 + w * 0.7, py + (w * 0.7) * Math.sin(angle)); ctx.stroke();

        if (isActive) {
          ctx.beginPath(); ctx.arc(planeX + 1, py, 4, 0, Math.PI * 2);
          ctx.fillStyle = "hsla(40, 95%, 50%, 1)"; ctx.shadowColor = "hsla(40, 95%, 50%, 1)";
          ctx.shadowBlur = 6; ctx.fill(); ctx.shadowBlur = 0;
        }
        ctx.restore();
      }

      // Scene objects
      [[w * 0.58, h * 0.42, 18, "hsla(15, 75%, 55%, 0.4)"], [w * 0.72, h * 0.55, 14, "hsla(210, 75%, 55%, 0.4)"]].forEach(([ox, oy, r, col]) => {
        ctx.save(); ctx.shadowColor = col as string; ctx.shadowBlur = 12;
        ctx.beginPath(); ctx.arc(ox as number, oy as number, r as number, 0, Math.PI * 2);
        ctx.fillStyle = col as string; ctx.fill();
        ctx.strokeStyle = (col as string).replace("0.4)", "1)"); ctx.lineWidth = 1; ctx.stroke();
        ctx.shadowBlur = 0; ctx.restore();
      });

      ctx.fillStyle = "hsl(var(--background) / 0.9)";
      ctx.beginPath(); (ctx as any).roundRect(w - 240, 10, 228, 44, 4); ctx.fill();
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "10px monospace"; ctx.textAlign = "left";
      ctx.fillText("r(t) = o + t·d", w - 230, 28);
      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace";
      ctx.fillText("o: origin, d: direction", w - 230, 42);
    } else if (m === "volume") {
      const SAMPLES = 48;
      const cx = w * 0.16, camCy = h / 2;
      const rayEnd = w * 0.82;

      const samples = Array.from({ length: SAMPLES }, (_, i) => {
        const tt = i / SAMPLES;
        const rx = cx + tt * (rayEnd - cx);
        const sx = (rx - w * 0.5) / 120, sz = tt * 7;
        const q = sceneQuery(sx, 0, sz);
        const delta = 1 / SAMPLES;
        const alpha = 1 - Math.exp(-q.sigma * delta * 3);
        return { rx, tt, q, alpha, sigma: q.sigma };
      });

      let T = 1;
      const weights: number[] = [];
      samples.forEach(s => {
        weights.push(T * s.alpha);
        T *= (1 - s.alpha);
      });

      samples.forEach((s, i) => {
        const wt = weights[i];
        const cr = Math.floor(s.q.r * 255), cg = Math.floor(s.q.g * 255), cb = Math.floor(s.q.b * 255);
        ctx.beginPath(); ctx.arc(s.rx, camCy, 2 + s.sigma * 6, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${cr},${cg},${cb},${clamp(s.sigma * 0.35, 0, 0.6)})`;
        ctx.fill();
        ctx.fillStyle = `rgba(${cr},${cg},${cb},0.7)`;
        ctx.fillRect(s.rx - 1, camCy + 40, 2, wt * 100);
      });

      ctx.strokeStyle = "hsl(var(--primary) / 0.4)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, camCy); ctx.lineTo(rayEnd, camCy); ctx.stroke();

      const activeS = Math.floor(t * 2) % SAMPLES;
      ctx.save(); ctx.shadowColor = "hsla(40, 95%, 50%, 1)"; ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.arc(samples[activeS].rx, camCy, 4, 0, Math.PI * 2);
      ctx.fillStyle = "hsla(40, 95%, 50%, 1)"; ctx.fill();
      ctx.shadowBlur = 0; ctx.restore();

      ctx.fillStyle = "hsl(var(--background) / 0.9)";
      ctx.beginPath(); (ctx as any).roundRect(12, 10, w - 24, 56, 4); ctx.fill();
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "9px monospace"; ctx.textAlign = "left";
      ctx.fillText("Ĉ(r) = Σᵢ Tᵢ·(1−e^{−σᵢδᵢ})·cᵢ", 22, 28);
      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace";
      ctx.fillText("Tᵢ = exp(−Σⱼ<ᵢ σⱼδⱼ)  |  αᵢ = 1−e^{−σᵢδᵢ}  |  wᵢ = Tᵢ·αᵢ", 22, 42);
      ctx.fillStyle = "hsla(40, 95%, 50%, 1)"; ctx.font = "8px monospace";
      ctx.fillText(`Sample ${activeS}: σ=${samples[activeS].sigma.toFixed(2)}, w=${weights[activeS].toFixed(3)}`, 22, 56);

      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace"; ctx.textAlign = "center";
      ctx.fillText("weight wᵢ per sample (bar height)", w / 2, camCy + 155);
    } else if (m === "mlp") {
      const layers = [
        { n: 6, label: "γ(x) 60-D", color: "hsl(var(--primary))", x: 0.08 },
        { n: 6, label: "FC 256", color: "hsl(var(--muted-foreground))", x: 0.2 },
        { n: 6, label: "FC 256", color: "hsl(var(--muted-foreground))", x: 0.32 },
        { n: 6, label: "FC 256", color: "hsl(var(--muted-foreground))", x: 0.44 },
        { n: 6, label: "FC+skip", color: "hsla(270, 60%, 55%, 1)", x: 0.56 },
        { n: 4, label: "σ+feat", color: "hsla(40, 95%, 50%, 1)", x: 0.68 },
        { n: 4, label: "γ(d) 24-D", color: "hsl(160, 80%, 55%)", x: 0.8 },
        { n: 3, label: "RGB", color: "hsl(0, 70%, 60%)", x: 0.92 },
      ];

      const activeLayer = Math.floor(t * 0.8) % layers.length;
      const neurons: { lx: number; ny: number }[][] = [];

      layers.forEach((layer, li) => {
        const lx = layer.x * w;
        const isActive = li === activeLayer;
        const gap = (h - 40) / (layer.n + 1);
        const layerNeurons: { lx: number; ny: number }[] = [];

        for (let ni = 0; ni < layer.n; ni++) {
          const ny = 20 + gap * (ni + 1);
          const pulse = isActive ? (Math.sin(t * 6 + ni * 0.7) * 0.5 + 0.5) : 0;
          ctx.save();
          ctx.shadowColor = layer.color; ctx.shadowBlur = isActive ? 8 + pulse * 6 : 2;
          ctx.beginPath(); ctx.arc(lx, ny, isActive ? 5 : 3, 0, Math.PI * 2);
          ctx.fillStyle = isActive ? layer.color : layer.color.replace(/[^,]+\)$/, "0.4)");
          ctx.fill(); ctx.shadowBlur = 0; ctx.restore();
          layerNeurons.push({ lx, ny });
        }
        neurons.push(layerNeurons);

        ctx.fillStyle = isActive ? layer.color : "hsl(var(--muted-foreground))";
        ctx.font = `${isActive ? "bold " : ""}7px monospace`; ctx.textAlign = "center";
        ctx.fillText(layer.label, lx, h - 6);

        // Connections
        if (li < layers.length - 1 && neurons[li]) {
          const nextLx = layers[li + 1].x * w;
          neurons[li].forEach(n1 => {
            for (let j = 0; j < Math.min(3, layers[li + 1].n); j++) {
              const nextGap = (h - 40) / (layers[li + 1].n + 1);
              const ny2 = 20 + nextGap * (j + 1);
              ctx.save(); ctx.globalAlpha = li === activeLayer ? 0.12 : 0.03;
              ctx.strokeStyle = layer.color; ctx.lineWidth = 0.5;
              ctx.beginPath(); ctx.moveTo(n1.lx, n1.ny); ctx.lineTo(nextLx, ny2); ctx.stroke();
              ctx.restore();
            }
          });
        }
      });

      // Skip connection
      ctx.save();
      ctx.strokeStyle = "hsla(270, 60%, 55%, 0.5)"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(layers[0].x * w, h * 0.15);
      ctx.quadraticCurveTo(w * 0.32, h * 0.05, layers[4].x * w, h * 0.15);
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "hsla(270, 60%, 55%, 1)"; ctx.font = "7px monospace"; ctx.textAlign = "center";
      ctx.fillText("skip concat", w * 0.32, h * 0.05);
      ctx.restore();

      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace"; ctx.textAlign = "center";
      ctx.fillText("F_Θ: (γ(x), γ(d)) → (σ, c)  |  8-layer MLP  |  ~1.3M params", w / 2, h - 16);
    } else if (m === "training") {
      const epoch = Math.floor(t * 0.5) % 100;
      const lossVal = 0.9 * Math.exp(-epoch * 0.06) + 0.02 + 0.01 * Math.sin(epoch * 1.2);
      const psnr = 20 * Math.log10(1 / Math.sqrt(lossVal));

      const stages = [
        { x: 0.1, label: "Posed\nImages", color: "hsl(var(--primary))" },
        { x: 0.28, label: "Sample\nRay", color: "hsl(var(--muted-foreground))" },
        { x: 0.46, label: "MLP\nQuery", color: "hsla(270, 60%, 55%, 1)" },
        { x: 0.64, label: "Volume\nRender", color: "hsla(40, 95%, 50%, 1)" },
        { x: 0.82, label: "MSE\nLoss", color: "hsl(0, 70%, 60%)" },
        { x: 0.95, label: "Back\nprop", color: "hsl(160, 80%, 55%)" },
      ];

      const activeS = Math.floor(t * 2) % stages.length;
      const cy = h * 0.25;

      stages.forEach((s, i) => {
        const sx = s.x * w;
        const isActive = i === activeS;
        ctx.save();
        ctx.shadowColor = s.color; ctx.shadowBlur = isActive ? 12 : 3;
        ctx.fillStyle = s.color.replace(/[^,]+\)$/, isActive ? "0.15)" : "0.08)");
        ctx.strokeStyle = s.color; ctx.lineWidth = isActive ? 2 : 1;
        ctx.beginPath(); (ctx as any).roundRect(sx - 35, cy - 22, 70, 44, 4); ctx.fill(); ctx.stroke();
        ctx.shadowBlur = 0;
        const lines = s.label.split("\n");
        ctx.fillStyle = s.color; ctx.font = "bold 8px monospace"; ctx.textAlign = "center";
        lines.forEach((ln, li) => ctx.fillText(ln, sx, cy - 6 + li * 12));

        if (i < stages.length - 1) {
          ctx.strokeStyle = s.color + "40"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(sx + 35, cy); ctx.lineTo(stages[i + 1].x * w - 35, cy); ctx.stroke();
        }
        ctx.restore();
      });

      // Loss curve
      const gx = 16, gy = cy + 40, gw = w - 32, gh = h - gy - 20;
      ctx.fillStyle = "hsl(var(--muted) / 0.2)";
      ctx.beginPath(); (ctx as any).roundRect(gx, gy, gw, gh, 4); ctx.fill();
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.stroke();

      ctx.save(); ctx.beginPath();
      for (let e = 0; e <= epoch; e++) {
        const lv = 0.9 * Math.exp(-e * 0.06) + 0.02 + 0.008 * Math.sin(e * 1.2);
        const x = gx + 4 + (e / 100) * gw;
        const y = gy + gh - 4 - lv * gh * 0.85;
        e === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = "hsl(0, 70%, 60%)"; ctx.lineWidth = 2; ctx.stroke(); ctx.restore();

      // PSNR
      ctx.save(); ctx.beginPath();
      for (let e = 0; e <= epoch; e++) {
        const lv = 0.9 * Math.exp(-e * 0.06) + 0.02;
        const p = 20 * Math.log10(1 / Math.sqrt(lv));
        const x = gx + 4 + (e / 100) * gw;
        const y = gy + gh - 4 - (p / 35) * gh * 0.85;
        e === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = "hsl(160, 80%, 55%)"; ctx.lineWidth = 1.5; ctx.stroke(); ctx.restore();

      ctx.fillStyle = "hsla(40, 95%, 50%, 1)"; ctx.font = "9px monospace"; ctx.textAlign = "left";
      ctx.fillText(`Epoch: ${epoch * 1000}  Loss: ${lossVal.toFixed(4)}  PSNR: ${psnr.toFixed(1)}dB`, gx + 6, gy + 14);
      ctx.fillStyle = "hsl(0, 70%, 60%)"; ctx.fillText("— MSE", gx + 6, gy + 26);
      ctx.fillStyle = "hsl(160, 80%, 55%)"; ctx.fillText("— PSNR", gx + 60, gy + 26);
    }
  };

  const canvasRef = useCanvas(draw, 420);
  return (
    <CanvasStage
      label="Neural Radiance Fields (NeRF)"
      modes={[
        { id: "ray", label: "Ray Marching" },
        { id: "volume", label: "Volume Rendering" },
        { id: "mlp", label: "MLP Architecture" },
        { id: "training", label: "Training Loop" },
      ]}
      activeMode={mode}
      onModeChange={setMode}
      hint="animated"
    >
      <canvas ref={canvasRef} className="w-full block" />
    </CanvasStage>
  );
}

// ══════════════════════════════════════════════════════════════
// 6. GAUSSIAN SPLATTING
// ══════════════════════════════════════════════════════════════
const NUM_G = 100;
const gsGaussians = Array.from({ length: NUM_G }, () => ({
  x: rnd(-3, 3), y: rnd(-2, 2), z: rnd(1, 8),
  sx: rnd(0.05, 0.4), sy: rnd(0.05, 0.3),
  rotAngle: rnd(0, Math.PI * 2),
  r: rnd(0.1, 1), g: rnd(0.1, 1), b: rnd(0.1, 1),
  alpha: rnd(0.4, 0.95),
}));

export function GaussianSplattingCanvas() {
  const [mode, setMode] = useState("gaussians");
  const drawRef = useRef(mode);
  drawRef.current = mode;

  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.clearRect(0, 0, w, h);
    const m = drawRef.current;

    if (m === "gaussians") {
      const rotY = t * 0.3;
      const cx = w / 2, cy = h * 0.52, sc = 50;

      const sorted = [...gsGaussians].map(g => {
        const proj = project3D(g.x, g.y, g.z, rotY, sc, cx, cy);
        return { ...g, px: proj.x, py: proj.y, pz: proj.z };
      }).sort((a, b) => b.pz - a.pz);

      sorted.forEach(g => {
        const scaleX = Math.max(3, g.sx * sc * 0.6);
        const scaleY = Math.max(2, g.sy * sc * 0.5);
        const depthA = clamp(0.2 + (g.pz + 1) * 0.1, 0.1, 0.85);
        ctx.save();
        ctx.translate(g.px, g.py);
        ctx.rotate(g.rotAngle + t * 0.05);
        const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, scaleX);
        grad.addColorStop(0, `rgba(${Math.floor(g.r * 255)},${Math.floor(g.g * 255)},${Math.floor(g.b * 255)},${g.alpha * depthA})`);
        grad.addColorStop(1, `rgba(${Math.floor(g.r * 255)},${Math.floor(g.g * 255)},${Math.floor(g.b * 255)},0)`);
        ctx.fillStyle = grad;
        ctx.scale(1, scaleY / scaleX);
        ctx.beginPath(); ctx.arc(0, 0, scaleX, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
      });

      ctx.fillStyle = "hsl(var(--background) / 0.85)";
      ctx.beginPath(); (ctx as any).roundRect(12, 12, 240, 60, 4); ctx.fill();
      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "9px monospace"; ctx.textAlign = "left";
      ctx.fillText(`${NUM_G} Gaussians (3D view)`, 22, 28);
      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace";
      ctx.fillText("G(x) = exp(−½(x−μ)ᵀΣ⁻¹(x−μ))", 22, 42);
      ctx.fillText("Σ = RSSᵀRᵀ  (rotation × scale)", 22, 54);
      ctx.fillStyle = "hsla(40, 95%, 50%, 1)"; ctx.font = "8px monospace";
      ctx.fillText(`~${NUM_G * 59} floats | 3M→180M real scenes`, 22, 66);
    } else if (m === "alpha") {
      const NUM_COMP = 6;
      const sortedG = Array.from({ length: NUM_COMP }, (_, i) => ({
        r: 0.3 + i * 0.1, g: 0.2 + (NUM_COMP - i) * 0.12, b: 0.1 + i * 0.08,
        alpha: 0.3 + i * 0.1, rx: 60 + i * 8,
      }));

      let accumR = 0, accumG = 0, accumB = 0, accumA = 1;
      const stepResults: { r: number; g: number; b: number; a: number }[] = [];
      sortedG.forEach(g => {
        const contribA = g.alpha * accumA;
        accumR += g.r * contribA;
        accumG += g.g * contribA;
        accumB += g.b * contribA;
        accumA *= (1 - g.alpha);
        stepResults.push({ r: accumR, g: accumG, b: accumB, a: 1 - accumA });
      });

      const activeStep = Math.floor(t * 1.5) % NUM_COMP;
      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "9px monospace"; ctx.textAlign = "center";
      ctx.fillText("Alpha Compositing — C = Σᵢ cᵢ αᵢ Πⱼ<ᵢ(1−αⱼ)", w / 2, 16);

      sortedG.forEach((g, i) => {
        const isActive = i <= activeStep;
        const py = h * 0.2 + i * ((h * 0.6) / NUM_COMP);
        const px = w * 0.18;

        ctx.save(); ctx.globalAlpha = isActive ? 0.9 : 0.25;
        const grd = ctx.createRadialGradient(px, py, 0, px, py, g.rx);
        grd.addColorStop(0, `rgba(${Math.floor(g.r * 255)},${Math.floor(g.g * 255)},${Math.floor(g.b * 255)},${g.alpha})`);
        grd.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = grd;
        ctx.beginPath(); ctx.arc(px, py, g.rx, 0, Math.PI * 2); ctx.fill();
        ctx.restore();

        ctx.fillStyle = isActive ? "hsl(var(--muted-foreground))" : "hsl(var(--border))";
        ctx.font = "7px monospace"; ctx.textAlign = "right";
        ctx.fillText(`G${i + 1} α=${g.alpha.toFixed(2)}`, px - g.rx - 6, py + 3);

        if (isActive) {
          const res = stepResults[i];
          ctx.fillStyle = `rgba(${Math.floor(res.r * 255)},${Math.floor(res.g * 255)},${Math.floor(res.b * 255)},${res.a})`;
          ctx.beginPath(); (ctx as any).roundRect(w * 0.6, py - 10, 60, 20, 3); ctx.fill();
          ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.stroke();
        }
      });

      const final = stepResults[activeStep];
      ctx.save(); ctx.shadowColor = "#fff"; ctx.shadowBlur = 12;
      ctx.fillStyle = `rgba(${Math.floor(final.r * 255)},${Math.floor(final.g * 255)},${Math.floor(final.b * 255)},1)`;
      ctx.beginPath(); ctx.arc(w * 0.85, h * 0.5, 16, 0, Math.PI * 2); ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "hsl(var(--foreground))"; ctx.font = "7px monospace"; ctx.textAlign = "center";
      ctx.fillText("pixel", w * 0.85, h * 0.5 + 26);
      ctx.restore();
    } else if (m === "density") {
      ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "9px monospace"; ctx.textAlign = "center";
      ctx.fillText("Adaptive Density Control", w / 2, 16);

      const ops = [
        { x: w * 0.22, y: h * 0.35, label: "CLONE", color: "hsl(160, 80%, 55%)", desc: "‖∇μ‖ > τ AND small\n→ duplicate + offset" },
        { x: w * 0.55, y: h * 0.35, label: "SPLIT", color: "hsla(40, 95%, 50%, 1)", desc: "‖∇μ‖ > τ AND large\n→ replace with 2 smaller" },
        { x: w * 0.85, y: h * 0.35, label: "PRUNE", color: "hsl(0, 70%, 60%)", desc: "α < ε_α\n→ remove" },
      ];

      const animPhase = Math.floor(t * 0.4) % 3;
      ops.forEach((op, i) => {
        const isActive = i === animPhase;
        const pulse = isActive ? (Math.sin(t * 4) * 0.5 + 0.5) : 0;

        ctx.save();
        ctx.shadowColor = op.color; ctx.shadowBlur = isActive ? 15 + pulse * 10 : 4;
        ctx.fillStyle = op.color.replace(/[^,]+\)$/, isActive ? "0.15)" : "0.08)");
        ctx.strokeStyle = op.color; ctx.lineWidth = isActive ? 2 : 1;
        ctx.beginPath(); (ctx as any).roundRect(op.x - 80, op.y - 45, 160, 90, 6); ctx.fill(); ctx.stroke();
        ctx.shadowBlur = 0;

        ctx.fillStyle = op.color; ctx.font = "bold 10px monospace"; ctx.textAlign = "center";
        ctx.fillText(op.label, op.x, op.y - 22);
        ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace";
        op.desc.split("\n").forEach((ln, li) => ctx.fillText(ln, op.x, op.y - 2 + li * 13));

        // Animation
        if (isActive) {
          const tt = (t * 2) % 1;
          const gy = op.y + 55;
          if (op.label === "CLONE") {
            const grd1 = ctx.createRadialGradient(op.x - 16 * tt, gy, 0, op.x, gy, 14);
            grd1.addColorStop(0, `rgba(6,214,160,0.6)`); grd1.addColorStop(1, "rgba(6,214,160,0)");
            ctx.fillStyle = grd1; ctx.beginPath(); ctx.arc(op.x - 16 * tt, gy, 14, 0, Math.PI * 2); ctx.fill();
            const grd2 = ctx.createRadialGradient(op.x + 16 * tt, gy, 0, op.x + 16 * tt, gy, 14 * tt);
            grd2.addColorStop(0, `rgba(6,214,160,${0.6 * tt})`); grd2.addColorStop(1, "rgba(6,214,160,0)");
            ctx.fillStyle = grd2; ctx.beginPath(); ctx.arc(op.x + 16 * tt, gy, 14 * tt, 0, Math.PI * 2); ctx.fill();
          } else if (op.label === "SPLIT") {
            const grd = ctx.createRadialGradient(op.x, gy, 0, op.x, gy, 22 * (1 - tt * 0.6));
            grd.addColorStop(0, `rgba(255,183,3,${0.6 * (1 - tt)})`); grd.addColorStop(1, "rgba(255,183,3,0)");
            ctx.fillStyle = grd; ctx.beginPath(); ctx.arc(op.x, gy, 22 * (1 - tt * 0.6), 0, Math.PI * 2); ctx.fill();
            [-1, 1].forEach(side => {
              const ex = op.x + side * 14 * tt;
              const grd2 = ctx.createRadialGradient(ex, gy, 0, ex, gy, 10 * tt);
              grd2.addColorStop(0, `rgba(255,183,3,${0.7 * tt})`); grd2.addColorStop(1, "rgba(255,183,3,0)");
              ctx.fillStyle = grd2; ctx.beginPath(); ctx.arc(ex, gy, 10 * tt, 0, Math.PI * 2); ctx.fill();
            });
          } else {
            ctx.save(); ctx.globalAlpha = 1 - tt;
            const grd = ctx.createRadialGradient(op.x, gy, 0, op.x, gy, 16);
            grd.addColorStop(0, "rgba(255,107,107,0.6)"); grd.addColorStop(1, "rgba(255,107,107,0)");
            ctx.fillStyle = grd; ctx.beginPath(); ctx.arc(op.x, gy, 16, 0, Math.PI * 2); ctx.fill();
            ctx.restore();
          }
        }
        ctx.restore();
      });

      ctx.fillStyle = "hsl(var(--background) / 0.9)";
      ctx.beginPath(); (ctx as any).roundRect(12, h - 60, w - 24, 48, 4); ctx.fill();
      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "hsl(var(--primary))"; ctx.font = "8px monospace"; ctx.textAlign = "left";
      ctx.fillText("Every 100 iters: check ‖∇μ‖ per Gaussian", 22, h - 42);
      ctx.fillStyle = "hsl(160, 80%, 55%)"; ctx.fillText("  clone if small + large gradient", 22, h - 28);
      ctx.fillStyle = "hsla(40, 95%, 50%, 1)"; ctx.fillText("  split if large + large gradient", 22, h - 16);
    } else if (m === "compare") {
      const splitX = w / 2;

      ctx.fillStyle = "hsla(270, 60%, 55%, 0.05)"; ctx.fillRect(0, 0, splitX, h);
      ctx.fillStyle = "hsl(var(--primary) / 0.03)"; ctx.fillRect(splitX, 0, splitX, h);

      ctx.fillStyle = "hsla(270, 60%, 55%, 1)"; ctx.font = "bold 10px monospace"; ctx.textAlign = "center";
      ctx.fillText("NeRF (Implicit)", splitX / 2, 22);
      ctx.fillStyle = "hsl(var(--primary))";
      ctx.fillText("3DGS (Explicit)", splitX + splitX / 2, 22);

      // NeRF MLP
      for (let l = 0; l < 5; l++) {
        for (let n = 0; n < 4; n++) {
          const lx = splitX * 0.4 + (l - 2) * 30;
          const ny = h * 0.35 + (n - 1.5) * 20;
          const pulse = (Math.sin(t * 2 + l * 0.5 + n * 0.3) + 1) / 2;
          ctx.beginPath(); ctx.arc(lx, ny, 4, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(270, 60%, 55%, ${0.3 + pulse * 0.5})`; ctx.fill();
        }
      }

      // GS blobs
      for (let i = 0; i < 12; i++) {
        const gx = splitX + splitX * 0.5 + rnd(-60, 60);
        const gy = h * 0.35 + rnd(-40, 40);
        const grad = ctx.createRadialGradient(gx, gy, 0, gx, gy, rnd(5, 14));
        grad.addColorStop(0, `hsla(${rnd(0, 360)}, 60%, 50%, 0.5)`);
        grad.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(gx, gy, rnd(5, 14), 0, Math.PI * 2); ctx.fill();
      }

      const metrics = [
        ["Train time", "Hours", "~30 min"],
        ["Render", "~30s/frame", "100+ FPS"],
        ["Memory", "~5MB", "~700MB"],
        ["Editability", "Hard", "Direct"],
        ["PSNR", "~26 dB", "~27 dB"],
      ];

      metrics.forEach((m, i) => {
        const y = h * 0.58 + i * 20;
        ctx.fillStyle = i % 2 === 0 ? "hsl(var(--muted) / 0.2)" : "hsl(var(--muted) / 0.1)";
        ctx.fillRect(8, y, splitX - 16, 18);
        ctx.fillRect(splitX + 8, y, splitX - 16, 18);

        ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.font = "8px monospace"; ctx.textAlign = "left";
        ctx.fillText(m[0], 14, y + 13);
        ctx.fillStyle = "hsla(270, 60%, 55%, 1)"; ctx.fillText(m[1], splitX * 0.55, y + 13);
        ctx.fillStyle = "hsl(var(--muted-foreground))"; ctx.fillText(m[0], splitX + 14, y + 13);
        ctx.fillStyle = "hsl(var(--primary))"; ctx.fillText(m[2], splitX + splitX * 0.55, y + 13);
      });

      ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(splitX, 0); ctx.lineTo(splitX, h); ctx.stroke();
    }
  };

  const canvasRef = useCanvas(draw, 420);
  return (
    <CanvasStage
      label="3D Gaussian Splatting"
      modes={[
        { id: "gaussians", label: "3D Gaussians" },
        { id: "alpha", label: "Alpha Compositing" },
        { id: "density", label: "Adaptive Density" },
        { id: "compare", label: "NeRF vs 3DGS" },
      ]}
      activeMode={mode}
      onModeChange={setMode}
      hint="animated"
    >
      <canvas ref={canvasRef} className="w-full block" />
    </CanvasStage>
  );
}
