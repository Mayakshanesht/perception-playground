import { useEffect, useRef, useState, useCallback } from "react";

// ─── Helpers ───
function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }
function clamp(v: number, a: number, b: number) { return Math.max(a, Math.min(b, v)); }

function depthColor(d: number, alpha = 1) {
  const t = clamp(d, 0, 1);
  let r: number, g: number, b: number;
  if (t < 0.33) {
    const tt = t / 0.33;
    r = lerp(255, 220, tt); g = lerp(50, 200, tt); b = lerp(50, 50, tt);
  } else if (t < 0.66) {
    const tt = (t - 0.33) / 0.33;
    r = lerp(220, 50, tt); g = lerp(200, 180, tt); b = lerp(50, 220, tt);
  } else {
    const tt = (t - 0.66) / 0.34;
    r = lerp(50, 20, tt); g = lerp(180, 80, tt); b = lerp(220, 180, tt);
  }
  return `rgba(${r|0},${g|0},${b|0},${alpha})`;
}

// ─── Canvas wrapper ───
function CanvasStage({ label, modes, activeMode, onModeChange, slider, children, height = 440 }: {
  label: string;
  modes: { id: string; label: string }[];
  activeMode: string;
  onModeChange: (m: string) => void;
  slider?: React.ReactNode;
  children: React.ReactNode;
  height?: number;
}) {
  return (
    <div className="rounded-xl border border-border bg-card/50 overflow-hidden">
      <div className="relative" style={{ height }}>
        {children}
        <div className="absolute top-3 left-3 bg-background/80 backdrop-blur-sm border border-border rounded-md px-2.5 py-1">
          <span className="text-[10px] font-mono font-bold uppercase tracking-wider" style={{ color: "hsl(32, 95%, 55%)" }}>{label}</span>
        </div>
      </div>
      <div className="flex items-center gap-2 flex-wrap p-3 border-t border-border bg-muted/20">
        {modes.map(m => (
          <button
            key={m.id}
            onClick={() => onModeChange(m.id)}
            className={`text-[10px] font-mono uppercase tracking-wider px-3 py-1.5 rounded-md border transition-colors ${
              activeMode === m.id
                ? "border-primary/50 bg-primary/10 text-primary"
                : "border-border bg-card text-muted-foreground hover:border-primary/30"
            }`}
          >
            {m.label}
          </button>
        ))}
        {slider && <div className="ml-auto">{slider}</div>}
      </div>
    </div>
  );
}

function useCanvas(draw: (ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => void, deps: any[] = []) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tRef = useRef(0);
  const rafRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
    };
    resize();
    const obs = new ResizeObserver(resize);
    obs.observe(canvas.parentElement!);

    const frame = () => {
      tRef.current += 0.012;
      if (canvas.width > 0 && canvas.height > 0) {
        draw(ctx, canvas.width, canvas.height, tRef.current);
      }
      rafRef.current = requestAnimationFrame(frame);
    };
    rafRef.current = requestAnimationFrame(frame);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      obs.disconnect();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return canvasRef;
}

// ════════════════════════════════
// 1. MONOCULAR DEPTH CANVAS
// ════════════════════════════════
export function MonocularDepthCanvas() {
  const [mode, setMode] = useState("cues");

  const objects = [
    { x: 0.12, y: 0.72, size: 0.08, depth: 0.05, label: "close tree", cue: "Occludes background → close", color: "#2d6a2d" },
    { x: 0.35, y: 0.65, size: 0.05, depth: 0.25, label: "mid tree", cue: "Smaller → farther", color: "#3a8a3a" },
    { x: 0.7, y: 0.62, size: 0.03, depth: 0.55, label: "far tree", cue: "Even smaller → even farther", color: "#4aaa4a" },
    { x: 0.5, y: 0.74, size: 0.07, depth: 0.15, label: "person", cue: "Known size prior → mid depth", color: "#e8b84b" },
    { x: 0.82, y: 0.68, size: 0.04, depth: 0.45, label: "car", cue: "Appears small → distant", color: "#38bdf8" },
  ];

  const draw = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.clearRect(0, 0, w, h);

    if (mode === "cues" || mode === "heatmap") {
      const fy = h * 0.58;
      // sky
      const sky = ctx.createLinearGradient(0, 0, 0, fy);
      sky.addColorStop(0, "#0a1828");
      sky.addColorStop(1, "#1e3a50");
      ctx.fillStyle = sky; ctx.fillRect(0, 0, w, fy);
      // ground
      const gnd = ctx.createLinearGradient(0, fy, 0, h);
      gnd.addColorStop(0, "#1a2c18");
      gnd.addColorStop(1, "#0e1a0d");
      ctx.fillStyle = gnd; ctx.fillRect(0, fy, w, h - fy);

      // perspective lines
      const vanX = w * 0.5, vanY = fy;
      for (let i = 0; i < 5; i++) {
        const t2 = i / 4;
        ctx.save();
        ctx.globalAlpha = 0.15 + t2 * 0.25;
        ctx.strokeStyle = "#e8b84b";
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(vanX, vanY);
        ctx.lineTo(lerp(w * 0.42, w * 0.1, t2), lerp(vanY, h, t2));
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(vanX, vanY);
        ctx.lineTo(lerp(w * 0.58, w * 0.9, t2), lerp(vanY, h, t2));
        ctx.stroke();
        ctx.restore();
      }

      // objects
      const sorted = [...objects].sort((a, b) => b.depth - a.depth);
      sorted.forEach(o => {
        const px = o.x * w;
        const py = fy + (1 - o.depth) * 0.4 * (h - fy);
        const sz = o.size * w * (1 - o.depth * 0.7);

        ctx.save();
        ctx.globalAlpha = 0.3;
        ctx.fillStyle = "#000";
        ctx.beginPath();
        ctx.ellipse(px, py + sz * 0.3, sz * 0.7, sz * 0.15, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        ctx.save();
        if (o.label.includes("tree")) {
          ctx.fillStyle = "#4a3020";
          ctx.fillRect(px - sz * 0.08, py - sz * 0.2, sz * 0.16, sz * 0.5);
          ctx.fillStyle = o.color;
          ctx.beginPath();
          ctx.arc(px, py - sz * 0.4, sz * 0.5, 0, Math.PI * 2);
          ctx.fill();
        } else if (o.label === "person") {
          ctx.fillStyle = o.color;
          ctx.beginPath(); ctx.arc(px, py - sz * 0.7, sz * 0.25, 0, Math.PI * 2); ctx.fill();
          ctx.fillRect(px - sz * 0.15, py - sz * 0.5, sz * 0.3, sz * 0.5);
        } else {
          ctx.fillStyle = o.color;
          ctx.fillRect(px - sz * 0.6, py - sz * 0.3, sz * 1.2, sz * 0.4);
          ctx.fillRect(px - sz * 0.4, py - sz * 0.6, sz * 0.8, sz * 0.35);
        }
        ctx.restore();

        // depth label
        if (mode === "cues") {
          ctx.save();
          ctx.fillStyle = "rgba(10,14,18,0.85)";
          ctx.beginPath();
          ctx.roundRect(px - 40, py - sz - 30, 80, 22, 3);
          ctx.fill();
          ctx.fillStyle = "#e8b84b";
          ctx.font = 'bold 10px monospace';
          ctx.textAlign = "center";
          ctx.fillText(`d≈${(o.depth * 10).toFixed(1)}m`, px, py - sz - 14);
          ctx.restore();
        }
      });

      // heatmap overlay
      if (mode === "heatmap") {
        ctx.save();
        ctx.globalAlpha = 0.6;
        const imgData = ctx.createImageData(w, h);
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const depth = y < fy ? 1 : 1 - (y - fy) / (h - fy);
            let minDist = depth;
            objects.forEach(o => {
              const dx = (x / w - o.x), dy = (y / h - o.y);
              const d = Math.sqrt(dx * dx + dy * dy);
              if (d < o.size * 1.5) minDist = Math.min(minDist, o.depth);
            });
            const idx = (y * w + x) * 4;
            const col = minDist;
            if (col < 0.33) {
              imgData.data[idx] = 255; imgData.data[idx + 1] = lerp(50, 200, col / 0.33) | 0; imgData.data[idx + 2] = 50;
            } else if (col < 0.66) {
              imgData.data[idx] = lerp(220, 50, (col - 0.33) / 0.33) | 0; imgData.data[idx + 1] = 180; imgData.data[idx + 2] = lerp(50, 220, (col - 0.33) / 0.33) | 0;
            } else {
              imgData.data[idx] = 30; imgData.data[idx + 1] = lerp(180, 80, (col - 0.66) / 0.34) | 0; imgData.data[idx + 2] = 180;
            }
            imgData.data[idx + 3] = 120;
          }
        }
        ctx.putImageData(imgData, 0, 0);
        ctx.restore();
      }

      // horizon label
      ctx.fillStyle = "rgba(232,184,75,0.4)";
      ctx.font = '9px monospace'; ctx.textAlign = "right";
      ctx.fillText("— horizon / vanishing point", w - 16, fy + 2);

    } else if (mode === "network") {
      const stages = [
        { label: "RGB Input", sub: "H×W×3", color: "#e8b84b", x: 0.08 },
        { label: "ViT Encoder", sub: "patch tokens", color: "#38bdf8", x: 0.28 },
        { label: "DPT Fusion", sub: "reassemble", color: "#2dd4bf", x: 0.48 },
        { label: "Depth Decoder", sub: "upsample", color: "#a78bfa", x: 0.68 },
        { label: "Depth Map", sub: "H×W×1", color: "#fb7185", x: 0.88 },
      ];
      const activeIdx = Math.floor(t * 0.6) % stages.length;
      const flowT = (t * 0.6) % 1;

      stages.forEach((s, i) => {
        const sx = s.x * w;
        const sy = h / 2;
        const isActive = i === activeIdx;
        const sz = Math.min(52, w * 0.06);
        const pulse = isActive ? (Math.sin(t * 6) * 0.5 + 0.5) : 0;

        ctx.save();
        ctx.shadowColor = s.color; ctx.shadowBlur = isActive ? 25 + pulse * 15 : 5;
        ctx.fillStyle = s.color + "18";
        ctx.strokeStyle = s.color;
        ctx.lineWidth = isActive ? 2 : 1;
        ctx.beginPath();
        ctx.roundRect(sx - sz / 2, sy - sz / 2, sz, sz, 6);
        ctx.fill(); ctx.stroke();
        ctx.shadowBlur = 0;

        ctx.fillStyle = s.color; ctx.textAlign = "center";
        ctx.font = `bold ${Math.min(11, w * 0.013)}px monospace`;
        ctx.fillText(s.label, sx, sy + 4);
        ctx.fillStyle = "#4a6070"; ctx.font = `${Math.min(9, w * 0.01)}px monospace`;
        ctx.fillText(s.sub, sx, sy + sz / 2 + 14);
        ctx.restore();

        if (i < stages.length - 1) {
          const nx = stages[i + 1].x * w;
          const particleX = sx + sz / 2 + (nx - sx - sz) * ((flowT + i / stages.length) % 1);
          ctx.beginPath(); ctx.arc(particleX, h / 2, 3, 0, Math.PI * 2);
          ctx.fillStyle = i === activeIdx ? s.color : s.color + "50";
          ctx.fill();
          ctx.strokeStyle = "#253545"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(sx + sz / 2 + 2, h / 2); ctx.lineTo(nx - sz / 2 - 2, h / 2); ctx.stroke();
        }
      });

      const cues = ["Texture gradient", "Relative size", "Occlusion", "Perspective lines", "Atmospheric haze"];
      cues.forEach((c, i) => {
        const alpha = 0.4 + 0.4 * Math.sin(t * 0.8 + i * 1.2);
        ctx.fillStyle = `rgba(232,184,75,${alpha})`;
        ctx.font = '10px monospace'; ctx.textAlign = "left";
        ctx.fillText("→ " + c, 16, 60 + i * 28);
      });
    }
  }, [mode]);

  const canvasRef = useCanvas(draw, [mode]);

  return (
    <CanvasStage
      label="Monocular Depth"
      modes={[
        { id: "cues", label: "Depth Cues" },
        { id: "heatmap", label: "Depth Heatmap" },
        { id: "network", label: "Network Flow" },
      ]}
      activeMode={mode}
      onModeChange={setMode}
    >
      <canvas ref={canvasRef} className="w-full h-full" />
    </CanvasStage>
  );
}

// ════════════════════════════════
// 2. STEREO VISION CANVAS
// ════════════════════════════════
export function StereoVisionCanvas() {
  const [mode, setMode] = useState("triangulate");
  const [baseline, setBaseline] = useState(60);

  const points3D = [
    { x: 0.3, y: 0.4, z: 80, color: "#e8b84b", label: "P₁" },
    { x: 0.6, y: 0.35, z: 140, color: "#2dd4bf", label: "P₂" },
    { x: 0.5, y: 0.6, z: 50, color: "#fb7185", label: "P₃" },
  ];

  const draw = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.clearRect(0, 0, w, h);
    const B = baseline;
    const f = 80;

    if (mode === "triangulate") {
      const camY = h * 0.82;
      const lCamX = w / 2 - B;
      const rCamX = w / 2 + B;

      // grid
      ctx.strokeStyle = "#1c2a38"; ctx.lineWidth = 0.5;
      for (let x = 0; x < w; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
      for (let y = 0; y < h; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

      // Cameras
      const planeY = camY - 35;
      [[lCamX, "#e8b84b", "Left Camera"], [rCamX, "#2dd4bf", "Right Camera"]].forEach(([cx2, col, lbl]) => {
        const cx = cx2 as number; const c = col as string; const l = lbl as string;
        ctx.save();
        ctx.fillStyle = c + "20"; ctx.strokeStyle = c; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.roundRect(cx - 28, camY - 18, 56, 36, 4); ctx.fill(); ctx.stroke();
        ctx.fillStyle = c; ctx.font = '9px monospace'; ctx.textAlign = "center";
        ctx.fillText(l, cx, camY + 28);
        ctx.restore();
      });

      // baseline
      ctx.strokeStyle = "#253545"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(lCamX, camY); ctx.lineTo(rCamX, camY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#4a6070"; ctx.font = '10px monospace'; ctx.textAlign = "center";
      ctx.fillText("B = " + B + "px", w / 2, camY + 14);

      // Points
      points3D.forEach(p => {
        const worldX = p.x * w;
        const worldY = p.y * h * 0.7 + h * 0.02;
        const Z = p.z;
        const xL = lCamX + (worldX - lCamX) * f / Z;
        const xR = rCamX + (worldX - rCamX) * f / Z;
        const disparity = Math.abs(xL - xR);

        ctx.save();
        ctx.strokeStyle = p.color + "60"; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(lCamX, camY - 5); ctx.lineTo(worldX, worldY); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(rCamX, camY - 5); ctx.lineTo(worldX, worldY); ctx.stroke();
        ctx.restore();

        ctx.save();
        ctx.shadowColor = p.color; ctx.shadowBlur = 14;
        ctx.beginPath(); ctx.arc(worldX, worldY, 7, 0, Math.PI * 2);
        ctx.fillStyle = p.color; ctx.fill();
        ctx.shadowBlur = 0;
        ctx.fillStyle = "#fff"; ctx.font = 'bold 10px monospace'; ctx.textAlign = "center";
        ctx.fillText(p.label, worldX, worldY + 3);

        const Zest = f * B / Math.max(1, disparity);
        ctx.fillStyle = p.color; ctx.font = '9px monospace';
        ctx.fillText(`Z≈${Zest.toFixed(0)}`, worldX + 12, worldY - 8);
        ctx.restore();
      });

      // formula
      ctx.fillStyle = "rgba(10,14,18,0.85)";
      ctx.beginPath(); ctx.roundRect(16, 16, 200, 42, 4); ctx.fill();
      ctx.fillStyle = "#e8b84b"; ctx.font = '12px monospace'; ctx.textAlign = "left";
      ctx.fillText("Z = f · B / d", 28, 36);
      ctx.fillStyle = "#4a6070"; ctx.font = '10px monospace';
      ctx.fillText(`f=${f}  B=${B}px`, 28, 52);

    } else if (mode === "epipolar") {
      const lCamX = w * 0.2, rCamX = w * 0.8;

      [[lCamX, "#e8b84b", "Left"], [rCamX, "#2dd4bf", "Right"]].forEach(([cx2, col, lbl]) => {
        const cx = cx2 as number; const c = col as string; const l = lbl as string;
        ctx.fillStyle = "#111820"; ctx.strokeStyle = c; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.roundRect(cx - 80, 40, 160, h * 0.62, 4); ctx.fill(); ctx.stroke();
        ctx.fillStyle = c; ctx.font = '10px monospace'; ctx.textAlign = "center";
        ctx.fillText(l + " Image", cx, 35);
      });

      const qx = lCamX + Math.cos(t * 0.7) * 50;
      const qy = 40 + 80 + Math.sin(t * 0.9) * 60;

      ctx.save();
      ctx.shadowColor = "#e8b84b"; ctx.shadowBlur = 16;
      ctx.beginPath(); ctx.arc(qx, qy, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#e8b84b"; ctx.fill();
      ctx.shadowBlur = 0; ctx.restore();

      ctx.fillStyle = "#e8b84b"; ctx.font = '9px monospace'; ctx.textAlign = "left";
      ctx.fillText("query p_L", qx + 10, qy);

      // Epipolar line
      ctx.save();
      ctx.strokeStyle = "#2dd4bf"; ctx.lineWidth = 2;
      ctx.shadowColor = "#2dd4bf"; ctx.shadowBlur = 8;
      ctx.setLineDash([5, 3]);
      ctx.beginPath(); ctx.moveTo(rCamX - 78, qy); ctx.lineTo(rCamX + 78, qy); ctx.stroke();
      ctx.setLineDash([]);
      ctx.shadowBlur = 0; ctx.restore();

      ctx.fillStyle = "#2dd4bf"; ctx.font = '9px monospace'; ctx.textAlign = "right";
      ctx.fillText("epipolar line", rCamX + 78, qy - 4);
      ctx.fillText("(search only here!)", rCamX + 78, qy + 12);

      // Candidates
      const candidates = [-50, -15, 20, 45];
      candidates.forEach((dx, ci) => {
        const alpha = 0.3 + 0.5 * Math.abs(Math.sin(t * 2 + ci));
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.beginPath(); ctx.arc(rCamX + dx, qy, 5, 0, Math.PI * 2);
        ctx.fillStyle = ci === 1 ? "#10b981" : "rgba(251,113,133,0.5)";
        ctx.fill();
        if (ci === 1) {
          ctx.strokeStyle = "#10b981"; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(rCamX + dx, qy, 9, 0, Math.PI * 2); ctx.stroke();
          ctx.fillStyle = "#10b981"; ctx.font = '9px monospace'; ctx.textAlign = "center";
          ctx.fillText("match!", rCamX + dx, qy + 20);
        }
        ctx.restore();
      });

      ctx.fillStyle = "rgba(10,14,18,0.9)";
      ctx.beginPath(); ctx.roundRect(w / 2 - 120, h - 80, 240, 52, 4); ctx.fill();
      ctx.fillStyle = "#2dd4bf"; ctx.font = '11px monospace'; ctx.textAlign = "center";
      ctx.fillText("p_R^T · F · p_L = 0", w / 2, h - 56);
      ctx.fillStyle = "#4a6070"; ctx.font = '9px monospace';
      ctx.fillText("Epipolar constraint: 2D→1D search", w / 2, h - 40);

    } else if (mode === "disparity") {
      const cols = 40, rows = 20;
      const cw = w / cols, ch = (h * 0.45) / rows;

      ctx.fillStyle = "#0d1520"; ctx.fillRect(0, 0, w, h * 0.48);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const depth = 0.1 + 0.7 * (
            Math.exp(-((c / cols - 0.3) ** 2 + (r / rows - 0.45) ** 2) / 0.03) * 0.9 +
            Math.exp(-((c / cols - 0.72) ** 2 + (r / rows - 0.5) ** 2) / 0.015) * 0.6 +
            (1 - r / rows) * 0.15
          );
          ctx.fillStyle = depthColor(depth, 0.9);
          ctx.fillRect(c * cw + 1, r * ch + h * 0.02 + 1, cw - 1, ch - 1);
        }
      }

      ctx.fillStyle = "#e8b84b"; ctx.font = '10px monospace'; ctx.textAlign = "left";
      ctx.fillText("Disparity Map (warm=close, cool=far)", 12, h * 0.5 + 18);

      // Cost volume slice
      const cvY = h * 0.55, cvH = h * 0.32;
      const DMAX = 32;
      ctx.fillStyle = "#111820"; ctx.strokeStyle = "#253545"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.roundRect(16, cvY, w - 32, cvH, 4); ctx.fill(); ctx.stroke();

      ctx.fillStyle = "#4a6070"; ctx.font = '10px monospace'; ctx.textAlign = "left";
      ctx.fillText("Cost Volume C[x,y,d] — matching cost vs disparity", 24, cvY + 18);

      const trueD = 19;
      for (let d = 0; d < DMAX; d++) {
        const cost = Math.abs(Math.sin((d - trueD) * 0.5)) * 0.8 + 0.05 * Math.random();
        const bx = 24 + d * ((w - 48) / DMAX);
        const bh = cost * (cvH - 40);
        const isMin = d === trueD;
        if (isMin) { ctx.shadowColor = "#e8b84b"; ctx.shadowBlur = 8; }
        ctx.fillStyle = isMin ? "#e8b84b" : "#253545";
        ctx.fillRect(bx, cvY + cvH - 24 - bh, (w - 48) / DMAX - 2, bh);
        ctx.shadowBlur = 0;
      }
      ctx.fillStyle = "#e8b84b"; ctx.font = '9px monospace'; ctx.textAlign = "center";
      ctx.fillText("▲ best disparity", 24 + trueD * ((w - 48) / DMAX), cvY + cvH - 8);
    }
  }, [mode, baseline]);

  const canvasRef = useCanvas(draw, [mode, baseline]);

  return (
    <CanvasStage
      label="Stereo Vision"
      modes={[
        { id: "triangulate", label: "Triangulation" },
        { id: "epipolar", label: "Epipolar Lines" },
        { id: "disparity", label: "Disparity Map" },
      ]}
      activeMode={mode}
      onModeChange={setMode}
      height={460}
      slider={
        <div className="flex items-center gap-2 text-[10px] font-mono text-muted-foreground">
          <span>Baseline B:</span>
          <input type="range" min={20} max={120} value={baseline} onChange={e => setBaseline(+e.target.value)} className="w-20 h-1 accent-primary" />
          <span>{baseline}px</span>
        </div>
      }
    >
      <canvas ref={canvasRef} className="w-full h-full" />
    </CanvasStage>
  );
}

// ════════════════════════════════
// 3. 2D POSE CANVAS
// ════════════════════════════════

const LIMBS: [number, number][] = [
  [0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
];
const LIMB_COLORS = [
  "#fb7185","#fb7185","#fb7185","#fb7185",
  "#e8b84b","#e8b84b","#e8b84b","#e8b84b","#e8b84b",
  "#2dd4bf","#2dd4bf","#2dd4bf","#38bdf8","#38bdf8","#38bdf8","#38bdf8"
];
const JOINT_NAMES = ["nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist","l_hip","r_hip","l_knee","r_knee","l_ankle","r_ankle"];

function getWalkPose(t: number): number[][] {
  const walk = Math.sin(t * 2);
  const walk2 = Math.cos(t * 2);
  return [
    [0.5, 0.08], [0.47, 0.07], [0.53, 0.07], [0.44, 0.08], [0.56, 0.08],
    [0.42, 0.2 + walk * 0.01], [0.58, 0.2 - walk * 0.01],
    [0.37, 0.32 + walk * 0.04], [0.63, 0.32 - walk * 0.04],
    [0.34, 0.44 + walk2 * 0.05], [0.66, 0.44 - walk2 * 0.05],
    [0.44, 0.45], [0.56, 0.45],
    [0.42, 0.62 + walk * 0.06], [0.58, 0.62 - walk * 0.06],
    [0.43, 0.78 + walk2 * 0.04], [0.57, 0.78 - walk2 * 0.04],
  ];
}

function drawSkeleton(ctx: CanvasRenderingContext2D, joints: number[][], w: number, h: number, scale = 1, offsetX = 0, offsetY = 0, alpha = 1) {
  LIMBS.forEach((limb, li) => {
    const jA = joints[limb[0]], jB = joints[limb[1]];
    ctx.save();
    ctx.globalAlpha = alpha * 0.85;
    ctx.strokeStyle = LIMB_COLORS[li];
    ctx.lineWidth = 3 * scale;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(offsetX + jA[0] * w * scale, offsetY + jA[1] * h * 0.85 * scale);
    ctx.lineTo(offsetX + jB[0] * w * scale, offsetY + jB[1] * h * 0.85 * scale);
    ctx.stroke();
    ctx.restore();
  });
  joints.forEach(j => {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.arc(offsetX + j[0] * w * scale, offsetY + j[1] * h * 0.85 * scale, 5 * scale, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.shadowColor = "#fff"; ctx.shadowBlur = 6 * scale;
    ctx.fill(); ctx.shadowBlur = 0;
    ctx.restore();
  });
}

export function Pose2DCanvas() {
  const [mode, setMode] = useState("skeleton");

  const draw = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.clearRect(0, 0, w, h);
    const joints = getWalkPose(t);

    if (mode === "skeleton") {
      drawSkeleton(ctx, joints, w, h, 1, 0, 0, 1);
      const activeJoint = Math.floor(t * 0.4) % joints.length;
      const jx = joints[activeJoint][0] * w;
      const jy = joints[activeJoint][1] * h * 0.85;
      ctx.save();
      ctx.shadowColor = "#e8b84b"; ctx.shadowBlur = 16;
      ctx.beginPath(); ctx.arc(jx, jy, 8, 0, Math.PI * 2);
      ctx.strokeStyle = "#e8b84b"; ctx.lineWidth = 2; ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "#e8b84b"; ctx.font = '10px monospace'; ctx.textAlign = "center";
      ctx.fillText(JOINT_NAMES[activeJoint], jx, jy - 14);
      ctx.restore();
      ctx.fillStyle = "#253545"; ctx.font = '9px monospace'; ctx.textAlign = "right";
      ctx.fillText("17 COCO keypoints · walking animation", w - 16, h - 10);

    } else if (mode === "heatmaps") {
      const showJoint = Math.floor(t * 0.5) % joints.length;
      const jx = joints[showJoint][0] * w;
      const jy = joints[showJoint][1] * h * 0.85;
      const sigma = 40;
      const imgData = ctx.createImageData(w, h);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const dx = x - jx, dy = y - jy;
          const val = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
          const idx = (y * w + x) * 4;
          imgData.data[idx] = Math.round(lerp(10, 255, val));
          imgData.data[idx + 1] = Math.round(lerp(20, 100, 1 - val));
          imgData.data[idx + 2] = Math.round(lerp(18, 50, 1 - val));
          imgData.data[idx + 3] = Math.round(val * 200 + 30);
        }
      }
      ctx.putImageData(imgData, 0, 0);
      drawSkeleton(ctx, joints, w, h, 1, 0, 0, 0.25);
      ctx.save();
      ctx.shadowColor = "#e8b84b"; ctx.shadowBlur = 24;
      ctx.beginPath(); ctx.arc(jx, jy, 10, 0, Math.PI * 2);
      ctx.fillStyle = "#e8b84b"; ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "#e8b84b"; ctx.font = 'bold 11px monospace'; ctx.textAlign = "center";
      ctx.fillText(JOINT_NAMES[showJoint], jx, jy - 16);
      ctx.restore();
      ctx.fillStyle = "rgba(10,14,18,0.8)";
      ctx.beginPath(); ctx.roundRect(16, h - 50, 300, 36, 4); ctx.fill();
      ctx.fillStyle = "#e8b84b"; ctx.font = '10px monospace'; ctx.textAlign = "left";
      ctx.fillText(`Heatmap for: ${JOINT_NAMES[showJoint]}`, 24, h - 30);

    } else if (mode === "paf") {
      drawSkeleton(ctx, joints, w, h, 1, 0, 0, 0.3);
      const limbIdx = Math.floor(t * 0.3) % LIMBS.length;
      const [a, b] = LIMBS[limbIdx];
      const jA = joints[a], jB = joints[b];
      const ax = jA[0] * w, ay = jA[1] * h * 0.85;
      const bx = jB[0] * w, by = jB[1] * h * 0.85;
      const dirX = bx - ax, dirY = by - ay;
      const len = Math.sqrt(dirX * dirX + dirY * dirY);
      const dx = dirX / len, dy = dirY / len;

      for (let i = 0; i <= 12; i++) {
        const t2 = i / 12;
        const px = ax + (bx - ax) * t2;
        const py = ay + (by - ay) * t2;
        for (let p = -2; p <= 2; p++) {
          const vx = px - dy * p * 30;
          const vy = py + dx * p * 30;
          const onLimb = Math.abs(p) <= 1;
          ctx.save();
          ctx.globalAlpha = onLimb ? 0.9 : 0.25;
          ctx.strokeStyle = LIMB_COLORS[limbIdx];
          ctx.lineWidth = onLimb ? 2 : 1;
          const arrowLen = 14;
          ctx.beginPath();
          ctx.moveTo(vx, vy);
          ctx.lineTo(vx + dx * arrowLen, vy + dy * arrowLen);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(vx + dx * arrowLen, vy + dy * arrowLen);
          ctx.lineTo(vx + dx * arrowLen - dy * 4 - dx * 4, vy + dy * arrowLen + dx * 4 - dy * 4);
          ctx.lineTo(vx + dx * arrowLen + dy * 4 - dx * 4, vy + dy * arrowLen - dx * 4 - dy * 4);
          ctx.closePath(); ctx.fillStyle = LIMB_COLORS[limbIdx]; ctx.fill();
          ctx.restore();
        }
      }

      // endpoints
      [[ax, ay, "J₁"], [bx, by, "J₂"]].forEach(([px, py, lbl]) => {
        ctx.save();
        ctx.shadowColor = LIMB_COLORS[limbIdx]; ctx.shadowBlur = 16;
        ctx.beginPath(); ctx.arc(px as number, py as number, 8, 0, Math.PI * 2);
        ctx.fillStyle = LIMB_COLORS[limbIdx]; ctx.fill();
        ctx.shadowBlur = 0;
        ctx.fillStyle = "#fff"; ctx.font = '9px monospace'; ctx.textAlign = "center";
        ctx.fillText(lbl as string, px as number, (py as number) + 3);
        ctx.restore();
      });

      ctx.fillStyle = "rgba(10,14,18,0.88)";
      ctx.beginPath(); ctx.roundRect(16, h - 60, 350, 46, 4); ctx.fill();
      ctx.fillStyle = LIMB_COLORS[limbIdx]; ctx.font = 'bold 11px monospace'; ctx.textAlign = "left";
      ctx.fillText(`PAF: ${JOINT_NAMES[LIMBS[limbIdx][0]]} → ${JOINT_NAMES[LIMBS[limbIdx][1]]}`, 24, h - 40);
      ctx.fillStyle = "#4a6070"; ctx.font = '9px monospace';
      ctx.fillText("E = ∫₀¹ Lc(p(u)) · (dj2−dj1)/‖dj2−dj1‖ du", 24, h - 22);

    } else if (mode === "topdown") {
      const midX = w / 2;
      // Left: Top-down
      ctx.fillStyle = "rgba(232,184,75,0.04)";
      ctx.beginPath(); ctx.roundRect(8, 8, midX - 16, h - 16, 6); ctx.fill();
      ctx.strokeStyle = "rgba(232,184,75,0.12)"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "#e8b84b"; ctx.font = 'bold 11px monospace'; ctx.textAlign = "center";
      ctx.fillText("TOP-DOWN", midX / 2, 30);

      ctx.strokeStyle = "rgba(232,184,75,0.25)"; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.roundRect(midX / 2 - 60, 45, 120, 140, 4); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(232,184,75,0.4)"; ctx.font = '9px monospace';
      ctx.fillText("1. Detect person box", midX / 2, 60);
      drawSkeleton(ctx, joints, w, h, 0.45, midX / 2 - 100, 45, 0.9);
      ctx.fillStyle = "rgba(232,184,75,0.4)"; ctx.font = '9px monospace'; ctx.textAlign = "center";
      ctx.fillText("2. Estimate joints inside box", midX / 2, h - 30);
      ctx.fillStyle = "#4a6070";
      ctx.fillText("Cost: O(num_persons)", midX / 2, h - 14);

      // Right: Bottom-up
      ctx.fillStyle = "rgba(45,212,191,0.04)";
      ctx.beginPath(); ctx.roundRect(midX + 8, 8, midX - 16, h - 16, 6); ctx.fill();
      ctx.strokeStyle = "rgba(45,212,191,0.12)"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "#2dd4bf"; ctx.font = 'bold 11px monospace'; ctx.textAlign = "center";
      ctx.fillText("BOTTOM-UP", midX + midX / 2, 30);

      joints.forEach((j, ji) => {
        const px = midX + 20 + (j[0] * 0.45 + 0.05) * (midX - 40);
        const py = 50 + j[1] * (h - 120) * 0.85;
        ctx.save();
        ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = LIMB_COLORS[ji % LIMB_COLORS.length];
        ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 6;
        ctx.fill(); ctx.shadowBlur = 0;
        ctx.restore();
      });

      ctx.fillStyle = "rgba(45,212,191,0.4)"; ctx.font = '9px monospace'; ctx.textAlign = "center";
      ctx.fillText("1. Detect all joints globally", midX + midX / 2, h - 30);
      ctx.fillStyle = "#4a6070";
      ctx.fillText("2. Group via PAF matching", midX + midX / 2, h - 14);

      // divider
      ctx.strokeStyle = "#253545"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(midX, 0); ctx.lineTo(midX, h); ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [mode]);

  const canvasRef = useCanvas(draw, [mode]);

  return (
    <CanvasStage
      label="2D Pose Estimation"
      modes={[
        { id: "skeleton", label: "Skeleton" },
        { id: "heatmaps", label: "Heatmaps" },
        { id: "paf", label: "Part Affinity Fields" },
        { id: "topdown", label: "Top-Down vs Bottom-Up" },
      ]}
      activeMode={mode}
      onModeChange={setMode}
      height={460}
    >
      <canvas ref={canvasRef} className="w-full h-full" />
    </CanvasStage>
  );
}

// ════════════════════════════════
// 4. 3D POSE CANVAS
// ════════════════════════════════

function get3DPose(t: number): number[][] {
  const walk = Math.sin(t * 1.5);
  return [
    [0, 1.7, 0], [-0.07, 1.75, 0.03], [0.07, 1.75, 0.03],
    [-0.14, 1.72, 0.02], [0.14, 1.72, 0.02],
    [-0.25, 1.4, 0], [0.25, 1.4, 0],
    [-0.35, 1.0, -0.1 * walk], [0.35, 1.0, 0.1 * walk],
    [-0.35, 0.6, -0.15 * walk], [0.35, 0.6, 0.15 * walk],
    [-0.15, 0.85, 0], [0.15, 0.85, 0],
    [-0.18, 0.4, 0.15 * walk], [0.18, 0.4, -0.15 * walk],
    [-0.18, 0, 0.1 * walk], [0.18, 0, -0.1 * walk],
  ];
}

function project3D(x3: number, y3: number, z3: number, rotY: number, scale: number, w: number, h: number) {
  const cosR = Math.cos(rotY), sinR = Math.sin(rotY);
  const rx = x3 * cosR - z3 * sinR;
  const rz = x3 * sinR + z3 * cosR;
  return { x: w / 2 + rx * scale, y: h * 0.5 - (y3 - 0.85) * scale, z: rz };
}

function draw3DSkeleton(ctx: CanvasRenderingContext2D, joints3D: number[][], rotY: number, scale: number, w: number, h: number, alpha = 1) {
  const projected = joints3D.map(j => project3D(j[0], j[1], j[2], rotY, scale, w, h));
  const limbsZ = LIMBS.map((limb, li) => ({
    li, z: (projected[limb[0]].z + projected[limb[1]].z) / 2
  })).sort((a, b) => a.z - b.z);

  limbsZ.forEach(({ li }) => {
    const [a, b] = LIMBS[li];
    const pA = projected[a], pB = projected[b];
    const df = clamp(0.3 + (pA.z + pB.z) * 0.3, 0.2, 1);
    ctx.save();
    ctx.globalAlpha = alpha * df;
    ctx.strokeStyle = LIMB_COLORS[li];
    ctx.lineWidth = 3.5 * (0.5 + df * 0.5);
    ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(pA.x, pA.y); ctx.lineTo(pB.x, pB.y); ctx.stroke();
    ctx.restore();
  });

  projected.forEach(p => {
    const df = clamp(0.3 + p.z * 0.4, 0.2, 1);
    ctx.save();
    ctx.globalAlpha = alpha * df;
    ctx.beginPath(); ctx.arc(p.x, p.y, 5 * df, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.shadowColor = "#fff"; ctx.shadowBlur = 6;
    ctx.fill(); ctx.shadowBlur = 0;
    ctx.restore();
  });
}

export function Pose3DCanvas() {
  const [mode, setMode] = useState("rotate");
  const [rotAngle, setRotAngle] = useState(30);

  const draw = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.clearRect(0, 0, w, h);
    const joints3D = get3DPose(t);

    if (mode === "rotate") {
      const rotY = rotAngle * Math.PI / 180;
      // ghost trails
      for (let i = 3; i >= 1; i--) {
        const oldJ = get3DPose(t - i * 0.2);
        draw3DSkeleton(ctx, oldJ, rotY, 200, w, h, 0.08 * (4 - i));
      }
      draw3DSkeleton(ctx, joints3D, rotY, 200, w, h, 1);

      // floor grid
      const gridY = h * 0.5 + 0.85 * 200;
      ctx.save(); ctx.globalAlpha = 0.15;
      for (let i = -3; i <= 3; i++) {
        const cosR = Math.cos(rotY);
        const x1 = w / 2 + (i * 0.3 * cosR) * 200;
        ctx.strokeStyle = "#2dd4bf"; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(x1 - 200, gridY); ctx.lineTo(x1 + 200, gridY); ctx.stroke();
      }
      ctx.restore();

      const deg = ((rotY * 180 / Math.PI) % 360 + 360) % 360;
      ctx.fillStyle = "#e8b84b"; ctx.font = '10px monospace'; ctx.textAlign = "right";
      ctx.fillText(`θ = ${deg.toFixed(0)}°`, w - 16, 24);

    } else if (mode === "lifting") {
      const joints2D = getWalkPose(t);
      const splitX = w / 2 - 30;

      // 2D side
      ctx.fillStyle = "rgba(232,184,75,0.04)";
      ctx.beginPath(); ctx.roundRect(8, 8, splitX - 16, h - 16, 6); ctx.fill();
      ctx.fillStyle = "#e8b84b"; ctx.font = 'bold 11px monospace'; ctx.textAlign = "center";
      ctx.fillText("2D INPUT", splitX / 2, 28);

      const scale2d = 0.45;
      LIMBS.forEach((limb, li) => {
        const [a, b] = limb;
        ctx.save();
        ctx.strokeStyle = LIMB_COLORS[li]; ctx.lineWidth = 2.5; ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(20 + joints2D[a][0] * (splitX - 30) * scale2d * 2, 50 + joints2D[a][1] * (h - 80) * 0.8);
        ctx.lineTo(20 + joints2D[b][0] * (splitX - 30) * scale2d * 2, 50 + joints2D[b][1] * (h - 80) * 0.8);
        ctx.stroke(); ctx.restore();
      });

      // arrow
      ctx.save();
      ctx.fillStyle = "rgba(45,212,191,0.08)"; ctx.strokeStyle = "#2dd4bf"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.roundRect(splitX, h / 2 - 22, 60, 44, 6); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#2dd4bf"; ctx.font = 'bold 9px monospace'; ctx.textAlign = "center";
      ctx.fillText("fθ", splitX + 30, h / 2 - 5);
      ctx.fillText("2D→3D", splitX + 30, h / 2 + 8);
      ctx.restore();

      // 3D side
      ctx.fillStyle = "rgba(45,212,191,0.04)";
      ctx.beginPath(); ctx.roundRect(splitX + 68, 8, w - splitX - 76, h - 16, 6); ctx.fill();
      ctx.fillStyle = "#2dd4bf"; ctx.font = 'bold 11px monospace'; ctx.textAlign = "center";
      ctx.fillText("3D OUTPUT", splitX + 68 + (w - splitX - 76) / 2, 28);

      const cx3 = splitX + 68 + (w - splitX - 76) / 2;
      const rotY2 = t * 0.4;
      const projected = joints3D.map(j => {
        const cosR = Math.cos(rotY2), sinR = Math.sin(rotY2);
        const rx = j[0] * cosR - j[2] * sinR;
        return { x: cx3 + rx * 140, y: h / 2 - (j[1] - 0.85) * 140 };
      });
      LIMBS.forEach((limb, li) => {
        ctx.save();
        ctx.strokeStyle = LIMB_COLORS[li]; ctx.lineWidth = 3; ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(projected[limb[0]].x, projected[limb[0]].y);
        ctx.lineTo(projected[limb[1]].x, projected[limb[1]].y); ctx.stroke(); ctx.restore();
      });
      projected.forEach(p => {
        ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#fff"; ctx.fill();
      });

      ctx.fillStyle = "#4a6070"; ctx.font = '9px monospace'; ctx.textAlign = "center";
      ctx.fillText("MPJPE = (1/K) Σ ‖X̂k − Xk‖₂", cx3, h - 14);

    } else if (mode === "ambiguity") {
      const baseJoints = get3DPose(0);
      const pose1 = baseJoints.map((j, i) => {
        if (i === 13 || i === 15) return [j[0], j[1], j[2] + 0.3 * Math.abs(Math.sin(t))];
        return [...j];
      });
      const pose2 = baseJoints.map((j, i) => {
        if (i === 13 || i === 15) return [j[0], j[1], j[2] - 0.3 * Math.abs(Math.sin(t))];
        return [...j];
      });

      const centers = [w * 0.22, w * 0.78];
      [pose1, pose2].forEach((pose, pi) => {
        const projected = pose.map(j => {
          const cosR = Math.cos(0.3), sinR = Math.sin(0.3);
          const rx = j[0] * cosR - j[2] * sinR;
          return { x: centers[pi] + rx * 140, y: h * 0.5 - (j[1] - 0.85) * 140 };
        });
        LIMBS.forEach((limb, li) => {
          ctx.save();
          ctx.strokeStyle = pi === 0 ? "#e8b84b" : "#2dd4bf";
          ctx.lineWidth = 3; ctx.lineCap = "round"; ctx.globalAlpha = 0.8;
          ctx.beginPath(); ctx.moveTo(projected[limb[0]].x, projected[limb[0]].y);
          ctx.lineTo(projected[limb[1]].x, projected[limb[1]].y); ctx.stroke(); ctx.restore();
        });
        projected.forEach(p => {
          ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
          ctx.fillStyle = "#fff"; ctx.fill();
        });
        ctx.fillStyle = pi === 0 ? "#e8b84b" : "#2dd4bf";
        ctx.font = 'bold 10px monospace'; ctx.textAlign = "center";
        ctx.fillText(pi === 0 ? "Pose A" : "Pose B", centers[pi], 22);
      });

      const cx = w / 2;
      ctx.fillStyle = "#fb7185"; ctx.font = 'bold 11px monospace'; ctx.textAlign = "center";
      ctx.fillText("BOTH PROJECT TO", cx, h * 0.5 - 30);
      ctx.fillText("SAME 2D SKELETON", cx, h * 0.5 - 14);
      ctx.fillStyle = "rgba(251,113,133,0.4)"; ctx.font = '9px monospace';
      ctx.fillText("← ambiguity →", cx, h * 0.5 + 4);

      // front-view 2D
      const front2D = getWalkPose(0);
      const frontY = h * 0.55;
      LIMBS.forEach((limb) => {
        ctx.save();
        ctx.strokeStyle = "rgba(251,113,133,0.44)"; ctx.lineWidth = 2; ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(cx - 60 + front2D[limb[0]][0] * 120, frontY + front2D[limb[0]][1] * 140);
        ctx.lineTo(cx - 60 + front2D[limb[1]][0] * 120, frontY + front2D[limb[1]][1] * 140);
        ctx.stroke(); ctx.restore();
      });
      ctx.fillStyle = "#fb7185"; ctx.font = '9px monospace'; ctx.textAlign = "center";
      ctx.fillText("Front-view 2D (indistinguishable)", cx, h - 14);
    }
  }, [mode, rotAngle]);

  const canvasRef = useCanvas(draw, [mode, rotAngle]);

  return (
    <CanvasStage
      label="3D Pose Lifting"
      modes={[
        { id: "rotate", label: "3D Skeleton" },
        { id: "lifting", label: "2D → 3D Lifting" },
        { id: "ambiguity", label: "Depth Ambiguity" },
      ]}
      activeMode={mode}
      onModeChange={setMode}
      height={480}
      slider={mode === "rotate" ? (
        <div className="flex items-center gap-2 text-[10px] font-mono text-muted-foreground">
          <span>Rotate:</span>
          <input type="range" min={0} max={360} value={rotAngle} onChange={e => setRotAngle(+e.target.value)} className="w-20 h-1 accent-primary" />
          <span>{rotAngle}°</span>
        </div>
      ) : undefined}
    >
      <canvas ref={canvasRef} className="w-full h-full" />
    </CanvasStage>
  );
}

// ════════════════════════════════
// 5. SELF-SUPERVISED DEPTH CANVAS
// ════════════════════════════════

export function SelfSupervisedDepthCanvas() {
  const [mode, setMode] = useState("pipeline");

  const sceneVals: number[] = [];
  for (let i = 0; i < 200; i++) {
    const r = Math.floor(i / 20) / 10, c = (i % 20) / 20;
    sceneVals.push(0.3 + 0.5 * Math.exp(-((c - 0.4) ** 2 + (r - 0.4) ** 2) / 0.05) + 0.2 * Math.exp(-((c - 0.7) ** 2 + (r - 0.6) ** 2) / 0.04));
  }

  const draw = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.clearRect(0, 0, w, h);

    function drawPixelGrid(ox: number, oy: number, gw: number, gh: number, cellW: number, vals: number[]) {
      for (let r = 0; r < gh; r++) {
        for (let c = 0; c < gw; c++) {
          const v = vals[r * gw + c] || 0.3;
          ctx.fillStyle = `rgba(${Math.floor(v * 200 + 30)},${Math.floor(v * 120 + 40)},${Math.floor(40 + v * 80)},0.9)`;
          ctx.fillRect(ox + c * cellW, oy + r * cellW, cellW - 1, cellW - 1);
        }
      }
    }

    const COLS = 20, ROWS = 10, CELLW = Math.min(16, w * 0.018);

    if (mode === "pipeline") {
      const step = Math.floor(t * 0.4) % 5;
      const cy = h * 0.45;

      const stages = [
        { x: 0.1, label: "Frame t", color: "#e8b84b" },
        { x: 0.28, label: "DepthNet", color: "#2dd4bf" },
        { x: 0.5, label: "View Synth", color: "#a78bfa" },
        { x: 0.7, label: "Photo Loss", color: "#fb7185" },
        { x: 0.88, label: "Frame t±1", color: "#e8b84b" },
      ];

      // frame t grid
      const gOx = stages[0].x * w - COLS * CELLW / 2;
      drawPixelGrid(gOx, cy - ROWS * CELLW / 2, COLS, ROWS, CELLW, sceneVals);
      ctx.strokeStyle = "#e8b84b"; ctx.lineWidth = 1.5;
      ctx.strokeRect(gOx, cy - ROWS * CELLW / 2, COLS * CELLW, ROWS * CELLW);
      ctx.fillStyle = "#e8b84b"; ctx.font = '10px monospace'; ctx.textAlign = "center";
      ctx.fillText("Frame t", stages[0].x * w, cy + ROWS * CELLW / 2 + 16);

      // network boxes
      stages.slice(1, 4).forEach((s, si) => {
        const sx = s.x * w;
        const isActive = si + 1 === step;
        ctx.save();
        ctx.shadowColor = s.color; ctx.shadowBlur = isActive ? 18 : 5;
        ctx.fillStyle = s.color + "18"; ctx.strokeStyle = s.color; ctx.lineWidth = isActive ? 2 : 1;
        ctx.beginPath(); ctx.roundRect(sx - 45, cy - 22, 90, 44, 5); ctx.fill(); ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.fillStyle = s.color; ctx.font = 'bold 10px monospace'; ctx.textAlign = "center";
        ctx.fillText(s.label, sx, cy + 4);
        ctx.restore();
      });

      // frame t+1 grid
      const gOx2 = stages[4].x * w - COLS * CELLW / 2;
      const warpedVals = sceneVals.map((v, i) => {
        const ni = i + Math.floor(t) % 3 + 1;
        return ni < sceneVals.length ? sceneVals[ni] : v * 0.7;
      });
      drawPixelGrid(gOx2, cy - ROWS * CELLW / 2, COLS, ROWS, CELLW, warpedVals);
      ctx.strokeStyle = "rgba(232,184,75,0.4)"; ctx.lineWidth = 1.5;
      ctx.strokeRect(gOx2, cy - ROWS * CELLW / 2, COLS * CELLW, ROWS * CELLW);
      ctx.fillStyle = "rgba(232,184,75,0.5)"; ctx.font = '10px monospace'; ctx.textAlign = "center";
      ctx.fillText("Frame t+1", stages[4].x * w, cy + ROWS * CELLW / 2 + 16);

      // connection lines
      const arrows: [number, number, number, number, string][] = [
        [stages[0].x * w + COLS * CELLW / 2, cy, stages[1].x * w - 45, cy, "#2dd4bf"],
        [stages[1].x * w + 45, cy, stages[2].x * w - 45, cy, "#a78bfa"],
        [stages[2].x * w + 45, cy, stages[3].x * w - 45, cy, "#fb7185"],
      ];
      arrows.forEach(([x1, y1, x2, y2, col]) => {
        ctx.save();
        ctx.strokeStyle = col + "50"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        ctx.restore();
      });

      // backprop
      ctx.save();
      ctx.strokeStyle = "rgba(251,113,133,0.35)"; ctx.lineWidth = 1.5; ctx.setLineDash([6, 3]);
      const bpY = cy + 80;
      ctx.beginPath(); ctx.moveTo(stages[3].x * w, cy + 22); ctx.lineTo(stages[3].x * w, bpY);
      ctx.lineTo(stages[1].x * w, bpY); ctx.lineTo(stages[1].x * w, cy + 22); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(251,113,133,0.5)"; ctx.font = '9px monospace'; ctx.textAlign = "center";
      ctx.fillText("← backprop (no depth labels!)", (stages[3].x * w + stages[1].x * w) / 2, bpY - 5);
      ctx.restore();

    } else if (mode === "warp") {
      const shift = Math.sin(t * 0.8) * 15;
      const GCOLS = 24, GROWS = 12, CW = Math.min(18, w * 0.02);
      const ox1 = w * 0.12, ox2 = w * 0.62;
      const oy = h / 2 - GROWS * CW / 2;

      for (let r = 0; r < GROWS; r++) for (let c = 0; c < GCOLS; c++) {
        const v = sceneVals[((r % 10) * 20 + (c % 20))] || 0.3;
        ctx.fillStyle = `rgba(${Math.floor(v * 200 + 30)},${Math.floor(v * 120 + 40)},${Math.floor(40 + v * 80)},0.9)`;
        ctx.fillRect(ox1 + c * CW, oy + r * CW, CW - 1, CW - 1);
      }
      ctx.strokeStyle = "#e8b84b"; ctx.lineWidth = 2;
      ctx.strokeRect(ox1, oy, GCOLS * CW, GROWS * CW);
      ctx.fillStyle = "#e8b84b"; ctx.font = '10px monospace'; ctx.textAlign = "center";
      ctx.fillText("Frame t (target)", ox1 + GCOLS * CW / 2, oy - 8);

      for (let r = 0; r < GROWS; r++) for (let c = 0; c < GCOLS; c++) {
        const sc = clamp(c - shift * 0.5, 0, GCOLS - 1);
        const v = sceneVals[(Math.floor(r % 10) * 20 + Math.floor(sc % 20))] || 0.3;
        ctx.fillStyle = `rgba(${Math.floor(v * 200 + 30)},${Math.floor(v * 120 + 40)},${Math.floor(40 + v * 80)},0.9)`;
        ctx.fillRect(ox2 + c * CW, oy + r * CW, CW - 1, CW - 1);
      }
      ctx.strokeStyle = "#a78bfa"; ctx.lineWidth = 2;
      ctx.strokeRect(ox2, oy, GCOLS * CW, GROWS * CW);
      ctx.fillStyle = "#a78bfa"; ctx.font = '10px monospace'; ctx.textAlign = "center";
      ctx.fillText("Synthesized (warped)", ox2 + GCOLS * CW / 2, oy - 8);

      // warp arrows
      for (let i = 0; i < 5; i++) {
        const r = 2 + i * 2, c = 8 + i * 2;
        const srcX = ox1 + c * CW + CW / 2;
        const srcY = oy + r * CW + CW / 2;
        const dstX = ox2 + clamp(c + shift * 0.5, 0, GCOLS - 1) * CW + CW / 2;
        const dstY = oy + r * CW + CW / 2;
        ctx.save();
        ctx.strokeStyle = "rgba(167,139,250,0.56)"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(srcX, srcY); ctx.lineTo(dstX - 4, dstY); ctx.stroke();
        ctx.restore();
      }

      ctx.fillStyle = "#4a6070"; ctx.font = '10px monospace'; ctx.textAlign = "center";
      ctx.fillText("ps ~ K · T(t→s) · Dt(pt) · K⁻¹ · pt", w / 2, h - 20);
      ctx.fillText(`ego-motion shift: ${shift.toFixed(1)}px`, w / 2, h - 6);

    } else if (mode === "loss") {
      const lossHistory = Array.from({ length: 100 }, (_, i) => {
        const epoch = i / 99;
        return 0.8 * Math.exp(-epoch * 4) + 0.05 + 0.02 * Math.sin(epoch * 40) * Math.exp(-epoch * 2);
      });

      const cx = w * 0.62, plotW = w * 0.32, plotH = h * 0.55;
      const plotX = cx - plotW / 2, plotY = h / 2 - plotH / 2;

      ctx.fillStyle = "#111820";
      ctx.beginPath(); ctx.roundRect(plotX, plotY, plotW, plotH, 4); ctx.fill();
      ctx.strokeStyle = "#253545"; ctx.lineWidth = 1; ctx.stroke();

      const maxEpoch = Math.min(99, Math.floor(t * 3));
      ctx.save();
      ctx.strokeStyle = "#fb7185"; ctx.lineWidth = 2;
      ctx.beginPath();
      lossHistory.slice(0, maxEpoch + 1).forEach((v, i) => {
        const x = plotX + (i / 99) * plotW;
        const y = plotY + plotH - v * plotH * 0.9 - 8;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();

      if (maxEpoch < 100) {
        const v = lossHistory[maxEpoch];
        const x = plotX + (maxEpoch / 99) * plotW;
        const y = plotY + plotH - v * plotH * 0.9 - 8;
        ctx.save();
        ctx.shadowColor = "#fb7185"; ctx.shadowBlur = 12;
        ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = "#fb7185"; ctx.fill();
        ctx.shadowBlur = 0;
        ctx.fillStyle = "#fb7185"; ctx.font = '9px monospace'; ctx.textAlign = "left";
        ctx.fillText(`Lp=${v.toFixed(3)}`, x + 8, y);
        ctx.restore();
      }

      ctx.fillStyle = "#4a6070"; ctx.font = '9px monospace';
      ctx.textAlign = "center"; ctx.fillText("Epoch →", cx, plotY + plotH + 14);
      ctx.fillStyle = "#fb7185"; ctx.font = 'bold 11px monospace';
      ctx.fillText("Photometric Loss", cx, plotY - 10);

      // loss components
      const fX = w * 0.15;
      const components = [
        { label: "SSIM term", formula: "α·(1−SSIM)/2", color: "#2dd4bf", val: 0.06 + 0.03 * Math.sin(t * 2) },
        { label: "L1 term", formula: "(1−α)·‖Ia−Ib‖₁", color: "#38bdf8", val: 0.03 + 0.02 * Math.cos(t * 1.5) },
        { label: "Total Lp", formula: "α=0.85", color: "#fb7185", val: 0.09 + 0.04 * Math.sin(t * 2) },
      ];

      ctx.fillStyle = "#e8b84b"; ctx.font = 'bold 11px monospace'; ctx.textAlign = "center";
      ctx.fillText("Loss Components", fX, 50);

      components.forEach((c, i) => {
        const cy = 80 + i * 90;
        ctx.fillStyle = c.color + "20"; ctx.strokeStyle = c.color; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.roundRect(fX - 100, cy, 200, 70, 5); ctx.fill(); ctx.stroke();
        ctx.fillStyle = c.color; ctx.font = 'bold 10px monospace'; ctx.textAlign = "center";
        ctx.fillText(c.label, fX, cy + 18);
        ctx.fillStyle = "#4a6070"; ctx.font = '9px monospace';
        ctx.fillText(c.formula, fX, cy + 34);
        ctx.fillStyle = c.color + "30";
        ctx.beginPath(); ctx.roundRect(fX - 80, cy + 44, 160, 12, 2); ctx.fill();
        ctx.fillStyle = c.color;
        ctx.beginPath(); ctx.roundRect(fX - 80, cy + 44, c.val * 360, 12, 2); ctx.fill();
      });
    }
  }, [mode]);

  const canvasRef = useCanvas(draw, [mode]);

  return (
    <CanvasStage
      label="Self-Supervised Depth"
      modes={[
        { id: "pipeline", label: "Full Pipeline" },
        { id: "warp", label: "View Synthesis" },
        { id: "loss", label: "Photometric Loss" },
      ]}
      activeMode={mode}
      onModeChange={setMode}
      height={460}
    >
      <canvas ref={canvasRef} className="w-full h-full" />
    </CanvasStage>
  );
}
