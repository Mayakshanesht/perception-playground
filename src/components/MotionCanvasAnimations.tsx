import { useEffect, useRef, useState, useCallback } from "react";

/* ═══════════════════════════════════════════
   SHARED UTILS
   ═══════════════════════════════════════════ */

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }
function clamp(v: number, mn: number, mx: number) { return Math.max(mn, Math.min(mx, v)); }

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

    const loop = () => {
      const w = canvas.width;
      const h = canvas.height;
      tRef.current += 0.015;
      ctx.clearRect(0, 0, w, h);
      draw(ctx, w, h, tRef.current);
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(rafRef.current!);
      obs.disconnect();
    };
  }, [draw, ...deps]);

  return canvasRef;
}

function CanvasStage({ label, modes, activeMode, onModeChange, children, height = 440 }: {
  label: string;
  modes: { id: string; label: string }[];
  activeMode: string;
  onModeChange: (m: string) => void;
  children: React.ReactNode;
  height?: number;
}) {
  return (
    <div className="rounded-xl border border-border bg-card/50 overflow-hidden">
      <div className="relative" style={{ height }}>
        {children}
        <div className="absolute top-3 left-3 bg-background/80 backdrop-blur-sm border border-border rounded-md px-2.5 py-1">
          <span className="text-[10px] font-mono font-bold uppercase tracking-wider" style={{ color: "hsl(280, 70%, 55%)" }}>{label}</span>
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
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════
   1. OPTICAL FLOW CANVAS
   ═══════════════════════════════════════════ */

export function OpticalFlowCanvas() {
  const [mode, setMode] = useState("dense");
  const modes = [
    { id: "dense", label: "Dense Flow" },
    { id: "lucas-kanade", label: "Lucas-Kanade" },
    { id: "raft", label: "RAFT Pipeline" },
  ];

  const drawDense = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    // Dark background
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    const cols = 20;
    const rows = 12;
    const cw = w / cols;
    const ch = h / rows;

    // Draw flow vectors as colored arrows
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cx2 = (c + 0.5) * cw;
        const cy2 = (r + 0.5) * ch;

        // Simulate a radial expansion flow + sinusoidal perturbation
        const dx = cx2 - w / 2;
        const dy = cy2 - h / 2;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx) + Math.sin(t + c * 0.3) * 0.3;
        const mag = clamp(dist / (w * 0.4), 0, 1) * 25 + Math.sin(t * 2 + r * 0.5) * 5;

        const u = Math.cos(angle) * mag;
        const v = Math.sin(angle) * mag;

        // HSL color based on flow direction
        const hue = ((angle / Math.PI) * 180 + 360) % 360;
        const sat = clamp(mag / 30, 0.3, 1) * 100;
        ctx.strokeStyle = `hsla(${hue}, ${sat}%, 60%, 0.8)`;
        ctx.fillStyle = `hsla(${hue}, ${sat}%, 60%, 0.8)`;
        ctx.lineWidth = 1.5;

        // Arrow
        ctx.beginPath();
        ctx.moveTo(cx2, cy2);
        ctx.lineTo(cx2 + u, cy2 + v);
        ctx.stroke();

        // Arrow head
        const headLen = 4;
        const headAngle = Math.atan2(v, u);
        ctx.beginPath();
        ctx.moveTo(cx2 + u, cy2 + v);
        ctx.lineTo(cx2 + u - headLen * Math.cos(headAngle - 0.4), cy2 + v - headLen * Math.sin(headAngle - 0.4));
        ctx.lineTo(cx2 + u - headLen * Math.cos(headAngle + 0.4), cy2 + v - headLen * Math.sin(headAngle + 0.4));
        ctx.fill();
      }
    }

    // Labels
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.font = "11px monospace";
    ctx.fillText("F(x,y) = (u(x,y), v(x,y))", w - 220, h - 15);
    ctx.fillText("Color = direction, Length = magnitude", 10, h - 15);
  }, []);

  const drawLK = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    // Draw two frames side by side
    const fw = w * 0.42;
    const fh = h * 0.7;
    const gap = w * 0.16;
    const x1 = (w - 2 * fw - gap) / 2;
    const x2 = x1 + fw + gap;
    const fy = (h - fh) / 2;

    // Frame borders
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    ctx.strokeRect(x1, fy, fw, fh);
    ctx.strokeRect(x2, fy, fw, fh);

    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "10px monospace";
    ctx.fillText("Frame t", x1 + 5, fy - 5);
    ctx.fillText("Frame t+1", x2 + 5, fy - 5);

    // Feature points in frame 1
    const points = [
      { x: 0.3, y: 0.3, dx: 0.05, dy: 0.02 },
      { x: 0.6, y: 0.25, dx: -0.03, dy: 0.04 },
      { x: 0.5, y: 0.6, dx: 0.04, dy: -0.02 },
      { x: 0.2, y: 0.7, dx: 0.06, dy: 0.01 },
      { x: 0.8, y: 0.5, dx: -0.04, dy: 0.03 },
      { x: 0.4, y: 0.45, dx: 0.03, dy: 0.05 },
    ];

    const windowSize = 20;
    const pulse = Math.sin(t * 3) * 0.3 + 0.7;

    points.forEach((p, i) => {
      const px1 = x1 + p.x * fw;
      const py1 = fy + p.y * fh;
      const px2 = x2 + (p.x + p.dx * Math.sin(t + i)) * fw;
      const py2 = fy + (p.y + p.dy * Math.sin(t * 0.8 + i)) * fh;

      // Window in frame 1
      ctx.strokeStyle = `hsla(280, 70%, 55%, ${pulse * 0.6})`;
      ctx.lineWidth = 1;
      ctx.strokeRect(px1 - windowSize, py1 - windowSize, windowSize * 2, windowSize * 2);

      // Point in frame 1
      ctx.fillStyle = "hsla(280, 80%, 65%, 0.9)";
      ctx.beginPath();
      ctx.arc(px1, py1, 4, 0, Math.PI * 2);
      ctx.fill();

      // Point in frame 2
      ctx.fillStyle = "hsla(170, 80%, 50%, 0.9)";
      ctx.beginPath();
      ctx.arc(px2, py2, 4, 0, Math.PI * 2);
      ctx.fill();

      // Correspondence line
      ctx.strokeStyle = "rgba(255,255,255,0.1)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(px1, py1);
      ctx.lineTo(px2, py2);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "10px monospace";
    ctx.fillText("Iₓu + Iᵧv + Iₜ = 0  (one eq, two unknowns)", w / 2 - 150, h - 12);
    ctx.fillText("Window assumption: flow constant in local patch", w / 2 - 160, h - 28);
  }, []);

  const drawRAFT = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    const stages = [
      { label: "Feature\nEncoder", x: 0.08, color: "hsla(280, 70%, 55%, 0.8)" },
      { label: "Correlation\nVolume 4D", x: 0.28, color: "hsla(200, 80%, 55%, 0.8)" },
      { label: "Context\nEncoder", x: 0.08, color: "hsla(340, 70%, 55%, 0.7)" },
      { label: "GRU\nUpdate ×12", x: 0.52, color: "hsla(170, 80%, 50%, 0.8)" },
      { label: "Flow\nΔf", x: 0.75, color: "hsla(45, 90%, 55%, 0.8)" },
      { label: "Final\nFlow", x: 0.9, color: "hsla(120, 70%, 50%, 0.8)" },
    ];

    const cy = h * 0.45;
    const boxW = w * 0.12;
    const boxH = 50;

    stages.forEach((s, i) => {
      const bx = s.x * w;
      const by = i === 2 ? cy + 80 : cy;

      // Glow
      const grad = ctx.createRadialGradient(bx + boxW / 2, by, 5, bx + boxW / 2, by, 60);
      grad.addColorStop(0, s.color.replace("0.8", "0.15").replace("0.7", "0.1"));
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
      ctx.fillRect(bx - 20, by - 30, boxW + 40, boxH + 60);

      // Box
      ctx.fillStyle = s.color.replace("0.8", "0.12").replace("0.7", "0.1");
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(bx, by - boxH / 2, boxW, boxH, 6);
      ctx.fill();
      ctx.stroke();

      // Label
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      const lines = s.label.split("\n");
      lines.forEach((line, li) => {
        ctx.fillText(line, bx + boxW / 2, by - 4 + li * 13);
      });
      ctx.textAlign = "left";
    });

    // Arrows between stages
    const arrows = [
      [0.08 + 0.12, 0.28], // encoder → corr
      [0.28 + 0.12, 0.52], // corr → GRU
      [0.08 + 0.12, 0.52], // context → GRU (offset y)
      [0.52 + 0.12, 0.75], // GRU → delta
      [0.75 + 0.12, 0.9],  // delta → final
    ];

    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 1;
    arrows.forEach(([from, to], i) => {
      const fromX = from * w;
      const toX = to * w;
      const fromY = i === 2 ? cy + 80 : cy;
      const toY = cy;
      ctx.beginPath();
      ctx.moveTo(fromX, fromY);
      ctx.lineTo(toX, toY);
      ctx.stroke();

      // Arrow head
      ctx.fillStyle = "rgba(255,255,255,0.3)";
      const angle = Math.atan2(toY - fromY, toX - fromX);
      ctx.beginPath();
      ctx.moveTo(toX, toY);
      ctx.lineTo(toX - 6 * Math.cos(angle - 0.3), toY - 6 * Math.sin(angle - 0.3));
      ctx.lineTo(toX - 6 * Math.cos(angle + 0.3), toY - 6 * Math.sin(angle + 0.3));
      ctx.fill();
    });

    // Iterative refinement indicator
    const iterPhase = (t * 2) % 12;
    const iterCount = Math.floor(iterPhase);
    ctx.fillStyle = "hsla(170, 80%, 50%, 0.6)";
    ctx.font = "10px monospace";
    ctx.fillText(`Iteration: ${iterCount + 1}/12`, 0.52 * w, cy + 42);

    // Recurrent arrow
    ctx.strokeStyle = "hsla(170, 80%, 50%, 0.3)";
    ctx.setLineDash([3, 3]);
    const gruX = 0.52 * w + boxW / 2;
    ctx.beginPath();
    ctx.arc(gruX, cy - boxH / 2 - 15, 15, Math.PI, 0, true);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "hsla(170, 80%, 50%, 0.4)";
    ctx.beginPath();
    ctx.moveTo(gruX + 15, cy - boxH / 2 - 15);
    ctx.lineTo(gruX + 11, cy - boxH / 2 - 21);
    ctx.lineTo(gruX + 11, cy - boxH / 2 - 9);
    ctx.fill();

    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("All-pairs correlation → iterative GRU refinement", 10, h - 12);
  }, []);

  const draw = mode === "dense" ? drawDense : mode === "lucas-kanade" ? drawLK : drawRAFT;
  const canvasRef = useCanvas(draw, [mode]);

  return (
    <CanvasStage label="Optical Flow" modes={modes} activeMode={mode} onModeChange={setMode}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
    </CanvasStage>
  );
}

/* ═══════════════════════════════════════════
   2. MULTI-OBJECT TRACKING CANVAS
   ═══════════════════════════════════════════ */

export function TrackingCanvas() {
  const [mode, setMode] = useState("sort");
  const modes = [
    { id: "sort", label: "SORT (Kalman+IoU)" },
    { id: "deepsort", label: "DeepSORT" },
    { id: "hungarian", label: "Hungarian Matching" },
  ];

  const drawSORT = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    // Simulate tracked objects moving
    const tracks = [
      { id: 1, color: "hsla(0, 80%, 60%, 0.9)", x0: 0.15, y0: 0.4, vx: 0.08, vy: 0.01 },
      { id: 2, color: "hsla(120, 70%, 50%, 0.9)", x0: 0.7, y0: 0.3, vx: -0.06, vy: 0.02 },
      { id: 3, color: "hsla(220, 80%, 60%, 0.9)", x0: 0.4, y0: 0.7, vx: 0.04, vy: -0.03 },
      { id: 4, color: "hsla(45, 90%, 55%, 0.9)", x0: 0.8, y0: 0.6, vx: -0.05, vy: -0.01 },
    ];

    const period = 8;
    const phase = t % period;

    tracks.forEach(tr => {
      const bw = 45;
      const bh = 55;

      // Trail (past positions)
      for (let i = 0; i < 15; i++) {
        const pt = phase - i * 0.15;
        if (pt < 0) continue;
        const px = (tr.x0 + tr.vx * pt) * w;
        const py = (tr.y0 + tr.vy * pt) * h;
        const alpha = (1 - i / 15) * 0.3;
        ctx.fillStyle = tr.color.replace("0.9", String(alpha));
        ctx.beginPath();
        ctx.arc(px, py, 2, 0, Math.PI * 2);
        ctx.fill();
      }

      // Current position
      const cx = (tr.x0 + tr.vx * phase) * w;
      const cy2 = (tr.y0 + tr.vy * phase) * h;

      // Bounding box
      ctx.strokeStyle = tr.color;
      ctx.lineWidth = 2;
      ctx.strokeRect(cx - bw / 2, cy2 - bh / 2, bw, bh);

      // Kalman predicted position (slightly ahead)
      const predX = cx + tr.vx * w * 0.3;
      const predY = cy2 + tr.vy * h * 0.3;
      ctx.strokeStyle = tr.color.replace("0.9", "0.3");
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(predX - bw / 2, predY - bh / 2, bw, bh);
      ctx.setLineDash([]);

      // Arrow from current to predicted
      ctx.strokeStyle = tr.color.replace("0.9", "0.4");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, cy2);
      ctx.lineTo(predX, predY);
      ctx.stroke();

      // ID label
      ctx.fillStyle = tr.color;
      ctx.font = "bold 10px monospace";
      ctx.fillText(`ID:${tr.id}`, cx - bw / 2, cy2 - bh / 2 - 4);
    });

    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "10px monospace";
    ctx.fillText("Solid = detection  |  Dashed = Kalman prediction", 10, h - 12);
    ctx.fillText(`Frame: ${Math.floor(phase * 30)}`, w - 90, h - 12);
  }, []);

  const drawDeepSORT = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    // Pipeline: Detection → ReID Feature → Kalman → Cost Matrix → Assignment
    const stages = [
      { label: "YOLO\nDetections", x: 0.05, color: "hsla(0, 80%, 55%, 0.8)" },
      { label: "Re-ID\nCNN (128d)", x: 0.25, color: "hsla(280, 70%, 55%, 0.8)" },
      { label: "Kalman\nPredict", x: 0.45, color: "hsla(45, 90%, 55%, 0.8)" },
      { label: "Cost\nMatrix", x: 0.62, color: "hsla(170, 80%, 50%, 0.8)" },
      { label: "Cascade\nMatch", x: 0.82, color: "hsla(120, 70%, 50%, 0.8)" },
    ];

    const cy = h * 0.35;
    const boxW = w * 0.13;
    const boxH = 48;

    stages.forEach((s, i) => {
      const bx = s.x * w;
      ctx.fillStyle = s.color.replace("0.8", "0.1");
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(bx, cy - boxH / 2, boxW, boxH, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      s.label.split("\n").forEach((line, li) => {
        ctx.fillText(line, bx + boxW / 2, cy - 3 + li * 13);
      });
      ctx.textAlign = "left";

      if (i < stages.length - 1) {
        const nextX = stages[i + 1].x * w;
        ctx.strokeStyle = "rgba(255,255,255,0.2)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(bx + boxW, cy);
        ctx.lineTo(nextX, cy);
        ctx.stroke();
      }
    });

    // Cost matrix visualization below
    const matY = h * 0.58;
    const matSize = Math.min(w * 0.35, h * 0.32);
    const matX = (w - matSize) / 2;
    const cells = 4;
    const cellSize = matSize / cells;

    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "10px monospace";
    ctx.fillText("Tracks →", matX, matY - 5);
    ctx.save();
    ctx.translate(matX - 10, matY + matSize / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Detections →", -matSize / 4, 0);
    ctx.restore();

    for (let r = 0; r < cells; r++) {
      for (let c = 0; c < cells; c++) {
        const cost = Math.abs(Math.sin(t + r * 1.5 + c * 2.3)) * 0.8 + 0.1;
        const hue = lerp(120, 0, cost);
        ctx.fillStyle = `hsla(${hue}, 70%, 45%, 0.6)`;
        ctx.fillRect(matX + c * cellSize, matY + r * cellSize, cellSize - 1, cellSize - 1);

        ctx.fillStyle = "rgba(255,255,255,0.7)";
        ctx.font = "9px monospace";
        ctx.textAlign = "center";
        ctx.fillText(cost.toFixed(2), matX + c * cellSize + cellSize / 2, matY + r * cellSize + cellSize / 2 + 3);
        ctx.textAlign = "left";
      }
    }

    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("cᵢⱼ = λ·d_mahal + (1-λ)·d_cosine", 10, h - 12);
  }, []);

  const drawHungarian = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    const leftX = w * 0.2;
    const rightX = w * 0.8;
    const trackCount = 4;
    const detCount = 4;

    // Draw track nodes (left)
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "10px monospace";
    ctx.fillText("Tracks (t-1)", leftX - 35, 30);
    ctx.fillText("Detections (t)", rightX - 40, 30);

    const trackColors = ["hsla(0, 80%, 60%, 0.9)", "hsla(120, 70%, 50%, 0.9)", "hsla(220, 80%, 60%, 0.9)", "hsla(45, 90%, 55%, 0.9)"];
    const spacing = (h - 100) / (trackCount + 1);

    const trackPositions: { x: number; y: number }[] = [];
    const detPositions: { x: number; y: number }[] = [];

    for (let i = 0; i < trackCount; i++) {
      const y = 60 + (i + 1) * spacing;
      trackPositions.push({ x: leftX, y });

      ctx.fillStyle = trackColors[i];
      ctx.beginPath();
      ctx.arc(leftX, y, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#0A0C15";
      ctx.font = "bold 9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(`T${i + 1}`, leftX, y + 3);
    }

    for (let i = 0; i < detCount; i++) {
      const y = 60 + (i + 1) * spacing;
      detPositions.push({ x: rightX, y });

      ctx.fillStyle = "rgba(255,255,255,0.6)";
      ctx.beginPath();
      ctx.arc(rightX, y, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#0A0C15";
      ctx.font = "bold 9px monospace";
      ctx.fillText(`D${i + 1}`, rightX, y + 3);
    }
    ctx.textAlign = "left";

    // All possible connections (faded)
    for (let i = 0; i < trackCount; i++) {
      for (let j = 0; j < detCount; j++) {
        ctx.strokeStyle = "rgba(255,255,255,0.05)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(trackPositions[i].x + 12, trackPositions[i].y);
        ctx.lineTo(detPositions[j].x - 12, detPositions[j].y);
        ctx.stroke();
      }
    }

    // Optimal assignment (animated)
    const assignments = [[0, 1], [1, 0], [2, 3], [3, 2]]; // shuffled matching
    const animProgress = clamp((Math.sin(t * 1.5) + 1) / 2, 0, 1);

    assignments.forEach(([ti, di]) => {
      const from = trackPositions[ti];
      const to = detPositions[di];
      const progress = clamp(animProgress * 1.5 - ti * 0.15, 0, 1);

      const endX = lerp(from.x + 12, to.x - 12, progress);
      const endY = lerp(from.y, to.y, progress);

      ctx.strokeStyle = trackColors[ti];
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(from.x + 12, from.y);
      ctx.lineTo(endX, endY);
      ctx.stroke();
    });

    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("min Σᵢⱼ Cᵢⱼ·xᵢⱼ  s.t. Σⱼxᵢⱼ=1, Σᵢxᵢⱼ=1", 10, h - 12);
    ctx.fillText("O(n³) optimal assignment", w - 200, h - 12);
  }, []);

  const draw = mode === "sort" ? drawSORT : mode === "deepsort" ? drawDeepSORT : drawHungarian;
  const canvasRef = useCanvas(draw, [mode]);

  return (
    <CanvasStage label="Multi-Object Tracking" modes={modes} activeMode={mode} onModeChange={setMode}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
    </CanvasStage>
  );
}

/* ═══════════════════════════════════════════
   3. 3D OBJECT TRACKING CANVAS
   ═══════════════════════════════════════════ */

export function Tracking3DCanvas() {
  const [mode, setMode] = useState("bev");
  const modes = [
    { id: "bev", label: "Bird's Eye View" },
    { id: "3dbox", label: "3D Bounding Boxes" },
    { id: "pipeline", label: "3D MOT Pipeline" },
  ];

  const drawBEV = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    // Road grid
    const cx = w / 2;
    const cy = h * 0.45;
    const scale = Math.min(w, h) * 0.003;

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 0.5;
    for (let i = -10; i <= 10; i++) {
      ctx.beginPath();
      ctx.moveTo(cx + i * 30 * scale, 20);
      ctx.lineTo(cx + i * 30 * scale, h - 20);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(20, cy + i * 30 * scale);
      ctx.lineTo(w - 20, cy + i * 30 * scale);
      ctx.stroke();
    }

    // Ego vehicle (bottom center)
    ctx.fillStyle = "hsla(220, 80%, 55%, 0.8)";
    ctx.beginPath();
    ctx.moveTo(cx, h - 40);
    ctx.lineTo(cx - 10, h - 25);
    ctx.lineTo(cx + 10, h - 25);
    ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.font = "8px monospace";
    ctx.textAlign = "center";
    ctx.fillText("EGO", cx, h - 18);

    // 3D tracked objects (BEV rectangles with velocity vectors)
    const objects3D = [
      { id: 1, x: -2, y: -5, vx: 0.5, vy: -1.5, w: 1.8, l: 4.2, yaw: 0.1, color: "hsla(0, 80%, 60%, 0.8)" },
      { id: 2, x: 3, y: -8, vx: -0.3, vy: -2.0, w: 1.8, l: 4.5, yaw: -0.05, color: "hsla(120, 70%, 50%, 0.8)" },
      { id: 3, x: -4, y: -3, vx: 0.8, vy: 0, w: 2.0, l: 5.0, yaw: 1.5, color: "hsla(45, 90%, 55%, 0.8)" },
      { id: 4, x: 1, y: -12, vx: 0, vy: -2.5, w: 1.8, l: 4.0, yaw: 0, color: "hsla(280, 70%, 55%, 0.8)" },
    ];

    const phase = t * 0.5;

    objects3D.forEach(obj => {
      const ox = cx + (obj.x + obj.vx * (phase % 5)) * 15 * scale;
      const oy = cy + (obj.y + obj.vy * (phase % 5)) * 15 * scale;
      const bw = obj.w * 15 * scale;
      const bl = obj.l * 15 * scale;

      // Trail
      for (let i = 0; i < 8; i++) {
        const tp = phase - i * 0.3;
        const tx = cx + (obj.x + obj.vx * (tp % 5)) * 15 * scale;
        const ty = cy + (obj.y + obj.vy * (tp % 5)) * 15 * scale;
        ctx.fillStyle = obj.color.replace("0.8", String(0.15 - i * 0.015));
        ctx.beginPath();
        ctx.arc(tx, ty, 2, 0, Math.PI * 2);
        ctx.fill();
      }

      // Rotated rectangle
      ctx.save();
      ctx.translate(ox, oy);
      ctx.rotate(obj.yaw);
      ctx.strokeStyle = obj.color;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(-bw / 2, -bl / 2, bw, bl);
      // Direction indicator
      ctx.fillStyle = obj.color;
      ctx.beginPath();
      ctx.moveTo(0, -bl / 2);
      ctx.lineTo(-3, -bl / 2 + 6);
      ctx.lineTo(3, -bl / 2 + 6);
      ctx.fill();
      ctx.restore();

      // Velocity arrow
      const velScale = 12 * scale;
      ctx.strokeStyle = obj.color.replace("0.8", "0.5");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ox + obj.vx * velScale, oy + obj.vy * velScale);
      ctx.stroke();

      // ID
      ctx.fillStyle = obj.color;
      ctx.font = "bold 9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(`#${obj.id}`, ox, oy - bl / 2 - 5);
    });

    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "10px monospace";
    ctx.fillText("Bird's Eye View — 3D positions + velocity vectors", 10, 18);
    ctx.fillText("State: [x, y, z, θ, l, w, h, vₓ, vᵧ]", 10, h - 12);
  }, []);

  const draw3DBox = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    // Perspective 3D boxes
    const boxes = [
      { cx: 0.3, cy: 0.55, size: 0.12, depth: 0.06, color: "hsla(0, 80%, 60%, 0.8)", label: "Car #1" },
      { cx: 0.6, cy: 0.45, size: 0.08, depth: 0.04, color: "hsla(120, 70%, 50%, 0.8)", label: "Car #2" },
      { cx: 0.75, cy: 0.6, size: 0.1, depth: 0.05, color: "hsla(45, 90%, 55%, 0.8)", label: "Truck" },
      { cx: 0.45, cy: 0.35, size: 0.05, depth: 0.025, color: "hsla(280, 70%, 55%, 0.8)", label: "Car #3" },
    ];

    boxes.forEach(box => {
      const x = box.cx * w + Math.sin(t + box.cx * 5) * 10;
      const y = box.cy * h;
      const s = box.size * w;
      const d = box.depth * w;

      // Front face
      ctx.strokeStyle = box.color;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x - s / 2, y - s * 0.6, s, s * 0.6);

      // Back face (offset)
      ctx.strokeStyle = box.color.replace("0.8", "0.4");
      ctx.strokeRect(x - s / 2 + d, y - s * 0.6 - d, s, s * 0.6);

      // Connecting lines
      ctx.beginPath();
      ctx.moveTo(x - s / 2, y - s * 0.6); ctx.lineTo(x - s / 2 + d, y - s * 0.6 - d);
      ctx.moveTo(x + s / 2, y - s * 0.6); ctx.lineTo(x + s / 2 + d, y - s * 0.6 - d);
      ctx.moveTo(x - s / 2, y); ctx.lineTo(x - s / 2 + d, y - d);
      ctx.moveTo(x + s / 2, y); ctx.lineTo(x + s / 2 + d, y - d);
      ctx.stroke();

      // Label
      ctx.fillStyle = box.color;
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(box.label, x, y + 14);
    });

    ctx.textAlign = "left";

    // Coordinate axes
    const axX = 60;
    const axY = h - 50;
    ctx.strokeStyle = "hsla(0, 80%, 60%, 0.6)";
    ctx.beginPath(); ctx.moveTo(axX, axY); ctx.lineTo(axX + 40, axY); ctx.stroke();
    ctx.strokeStyle = "hsla(120, 70%, 50%, 0.6)";
    ctx.beginPath(); ctx.moveTo(axX, axY); ctx.lineTo(axX, axY - 40); ctx.stroke();
    ctx.strokeStyle = "hsla(220, 80%, 60%, 0.6)";
    ctx.beginPath(); ctx.moveTo(axX, axY); ctx.lineTo(axX + 25, axY + 15); ctx.stroke();

    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.font = "9px monospace";
    ctx.fillText("X", axX + 42, axY + 3);
    ctx.fillText("Y", axX - 2, axY - 42);
    ctx.fillText("Z", axX + 28, axY + 18);

    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("3D box: (x,y,z,l,w,h,θ) — 7 DoF per object", 10, 18);
  }, []);

  const drawPipeline = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    const stages = [
      { label: "LiDAR/Camera\nInput", y: 0.12, color: "hsla(220, 80%, 55%, 0.8)" },
      { label: "3D Object\nDetector", y: 0.3, color: "hsla(280, 70%, 55%, 0.8)" },
      { label: "State\nPrediction", y: 0.48, color: "hsla(45, 90%, 55%, 0.8)" },
      { label: "3D IoU\nAssociation", y: 0.66, color: "hsla(170, 80%, 50%, 0.8)" },
      { label: "Track\nManagement", y: 0.84, color: "hsla(0, 80%, 60%, 0.8)" },
    ];

    const cx = w * 0.35;
    const boxW = w * 0.25;
    const boxH = 40;

    stages.forEach((s, i) => {
      const by = s.y * h;
      ctx.fillStyle = s.color.replace("0.8", "0.1");
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(cx - boxW / 2, by - boxH / 2, boxW, boxH, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      s.label.split("\n").forEach((line, li) => {
        ctx.fillText(line, cx, by - 3 + li * 12);
      });

      if (i < stages.length - 1) {
        const nextY = stages[i + 1].y * h - boxH / 2;
        ctx.strokeStyle = "rgba(255,255,255,0.2)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx, by + boxH / 2);
        ctx.lineTo(cx, nextY);
        ctx.stroke();
      }
    });

    // Side annotations
    const annX = w * 0.7;
    const anns = [
      { y: 0.12, text: "Point cloud / multi-cam" },
      { y: 0.3, text: "CenterPoint / PointPillars" },
      { y: 0.48, text: "Kalman / constant velocity" },
      { y: 0.66, text: "3D IoU or center distance" },
      { y: 0.84, text: "Birth / death management" },
    ];

    ctx.textAlign = "left";
    anns.forEach(a => {
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.font = "9px monospace";
      ctx.fillText(a.text, annX, a.y * h + 4);
      ctx.strokeStyle = "rgba(255,255,255,0.1)";
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(cx + boxW / 2 + 5, a.y * h);
      ctx.lineTo(annX - 5, a.y * h);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("AB3DMOT: Simple 3D MOT baseline with Kalman + 3D IoU", 10, h - 12);
  }, []);

  const draw = mode === "bev" ? drawBEV : mode === "3dbox" ? draw3DBox : drawPipeline;
  const canvasRef = useCanvas(draw, [mode]);

  return (
    <CanvasStage label="3D Object Tracking" modes={modes} activeMode={mode} onModeChange={setMode}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
    </CanvasStage>
  );
}

/* ═══════════════════════════════════════════
   4. ACTION RECOGNITION CANVAS
   ═══════════════════════════════════════════ */

export function ActionRecognitionCanvas() {
  const [mode, setMode] = useState("twostream");
  const modes = [
    { id: "twostream", label: "Two-Stream" },
    { id: "3dconv", label: "3D Convolution" },
    { id: "slowfast", label: "SlowFast" },
  ];

  const drawTwoStream = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    const cy = h / 2;

    // Spatial stream (top)
    const spatialY = cy - 70;
    ctx.fillStyle = "hsla(220, 80%, 55%, 0.1)";
    ctx.strokeStyle = "hsla(220, 80%, 55%, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(w * 0.05, spatialY - 40, w * 0.25, 35, 6);
    ctx.fill(); ctx.stroke();
    ctx.beginPath();
    ctx.roundRect(w * 0.35, spatialY - 40, w * 0.2, 35, 6);
    ctx.fill(); ctx.stroke();

    // Temporal stream (bottom)
    const temporalY = cy + 50;
    ctx.fillStyle = "hsla(170, 80%, 50%, 0.1)";
    ctx.strokeStyle = "hsla(170, 80%, 50%, 0.5)";
    ctx.beginPath();
    ctx.roundRect(w * 0.05, temporalY - 15, w * 0.25, 35, 6);
    ctx.fill(); ctx.stroke();
    ctx.beginPath();
    ctx.roundRect(w * 0.35, temporalY - 15, w * 0.2, 35, 6);
    ctx.fill(); ctx.stroke();

    // Labels
    ctx.fillStyle = "hsla(220, 80%, 60%, 0.9)";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText("RGB Frame", w * 0.175, spatialY - 18);
    ctx.fillText("Spatial CNN", w * 0.45, spatialY - 18);

    ctx.fillStyle = "hsla(170, 80%, 55%, 0.9)";
    ctx.fillText("Optical Flow Stack", w * 0.175, temporalY + 7);
    ctx.fillText("Temporal CNN", w * 0.45, temporalY + 7);

    // Arrows
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 1;
    // Spatial
    ctx.beginPath(); ctx.moveTo(w * 0.3, spatialY - 22); ctx.lineTo(w * 0.35, spatialY - 22); ctx.stroke();
    // Temporal
    ctx.beginPath(); ctx.moveTo(w * 0.3, temporalY + 2); ctx.lineTo(w * 0.35, temporalY + 2); ctx.stroke();

    // Fusion
    const fusionX = w * 0.65;
    ctx.fillStyle = "hsla(45, 90%, 55%, 0.1)";
    ctx.strokeStyle = "hsla(45, 90%, 55%, 0.5)";
    ctx.beginPath();
    ctx.roundRect(fusionX, cy - 25, w * 0.12, 50, 6);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = "hsla(45, 90%, 60%, 0.9)";
    ctx.fillText("Fusion", fusionX + w * 0.06, cy + 4);

    // Connect to fusion
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.beginPath();
    ctx.moveTo(w * 0.55, spatialY - 22);
    ctx.lineTo(fusionX, cy - 10);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(w * 0.55, temporalY + 2);
    ctx.lineTo(fusionX, cy + 10);
    ctx.stroke();

    // Output
    const outX = w * 0.82;
    ctx.fillStyle = "hsla(120, 70%, 50%, 0.1)";
    ctx.strokeStyle = "hsla(120, 70%, 50%, 0.5)";
    ctx.beginPath();
    ctx.roundRect(outX, cy - 20, w * 0.12, 40, 6);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = "hsla(120, 70%, 55%, 0.9)";
    ctx.fillText("Action\nClass", outX + w * 0.06, cy + 4);

    ctx.beginPath();
    ctx.moveTo(fusionX + w * 0.12, cy);
    ctx.lineTo(outX, cy);
    ctx.stroke();

    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("P(y|V) = ½·P_spatial(y|Iₜ) + ½·P_temporal(y|Fₜ)", 10, h - 12);
  }, []);

  const draw3DConv = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    // Video clip as stacked frames
    const startX = w * 0.05;
    const frameW = 40;
    const frameH = 30;
    const frames = 8;
    const cy = h * 0.4;

    for (let i = 0; i < frames; i++) {
      const x = startX + i * 8;
      const y = cy - i * 5;
      const alpha = 0.3 + (i / frames) * 0.5;
      ctx.fillStyle = `hsla(220, 60%, 50%, ${alpha * 0.3})`;
      ctx.strokeStyle = `hsla(220, 60%, 55%, ${alpha})`;
      ctx.lineWidth = 0.8;
      ctx.fillRect(x, y, frameW, frameH);
      ctx.strokeRect(x, y, frameW, frameH);
    }

    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Video Clip", startX + frames * 4 + frameW / 2, cy + frameH + 15);

    // 3D kernel visualization
    const kernelX = w * 0.28;
    const kernelY = cy - 10;
    const kSize = 20;
    const kDepth = 3;
    const pulse = Math.sin(t * 3) * 0.2 + 0.8;

    for (let d = 0; d < kDepth; d++) {
      ctx.fillStyle = `hsla(280, 70%, 55%, ${(0.15 + d * 0.1) * pulse})`;
      ctx.strokeStyle = `hsla(280, 70%, 55%, ${(0.5 + d * 0.15) * pulse})`;
      ctx.lineWidth = 1;
      ctx.fillRect(kernelX + d * 6, kernelY - d * 4, kSize, kSize);
      ctx.strokeRect(kernelX + d * 6, kernelY - d * 4, kSize, kSize);
    }

    ctx.fillStyle = "hsla(280, 70%, 60%, 0.8)";
    ctx.fillText("3×3×3", kernelX + kDepth * 3 + kSize / 2, kernelY + kSize + 15);

    // Feature volume output stages
    const stageX = [w * 0.45, w * 0.6, w * 0.75];
    const stageSizes = [{ w: 30, h: 25, d: 6 }, { w: 22, h: 18, d: 4 }, { w: 15, h: 12, d: 3 }];

    stageX.forEach((sx, si) => {
      const ss = stageSizes[si];
      for (let d = 0; d < ss.d; d++) {
        ctx.fillStyle = `hsla(${170 + si * 40}, 70%, 50%, ${0.1 + d * 0.05})`;
        ctx.strokeStyle = `hsla(${170 + si * 40}, 70%, 50%, 0.4)`;
        ctx.lineWidth = 0.8;
        ctx.fillRect(sx + d * 5, cy - ss.h / 2 - d * 3, ss.w, ss.h);
        ctx.strokeRect(sx + d * 5, cy - ss.h / 2 - d * 3, ss.w, ss.h);
      }

      // Arrow
      if (si < stageX.length - 1) {
        ctx.strokeStyle = "rgba(255,255,255,0.15)";
        ctx.beginPath();
        ctx.moveTo(sx + ss.w + ss.d * 5 + 5, cy);
        ctx.lineTo(stageX[si + 1] - 5, cy);
        ctx.stroke();
      }
    });

    // Arrow from clip to kernel
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.beginPath();
    ctx.moveTo(startX + frames * 8 + frameW + 5, cy + frameH / 2 - 10);
    ctx.lineTo(kernelX - 5, kernelY + kSize / 2);
    ctx.stroke();

    // Arrow from kernel to stages
    ctx.beginPath();
    ctx.moveTo(kernelX + kDepth * 6 + kSize + 5, kernelY + kSize / 2);
    ctx.lineTo(stageX[0] - 5, cy);
    ctx.stroke();

    // Final: GAP + FC
    ctx.fillStyle = "hsla(45, 90%, 55%, 0.1)";
    ctx.strokeStyle = "hsla(45, 90%, 55%, 0.5)";
    ctx.beginPath();
    ctx.roundRect(w * 0.88, cy - 18, w * 0.08, 36, 6);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = "hsla(45, 90%, 60%, 0.9)";
    ctx.fillText("FC", w * 0.92, cy + 4);

    ctx.beginPath();
    ctx.moveTo(stageX[2] + stageSizes[2].w + stageSizes[2].d * 5 + 5, cy);
    ctx.lineTo(w * 0.88 - 5, cy);
    ctx.stroke();

    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("(f*g)(x,y,t) = Σᵢⱼₖ f(i,j,k)·g(x-i,y-j,t-k)", 10, h - 12);
  }, []);

  const drawSlowFast = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    const cy = h / 2;

    // Slow pathway (top)
    const slowY = cy - 60;
    ctx.fillStyle = "hsla(220, 80%, 55%, 0.1)";
    ctx.strokeStyle = "hsla(220, 80%, 55%, 0.5)";
    ctx.lineWidth = 1;

    // Slow: few frames, many channels
    const slowFrames = 4;
    for (let i = 0; i < slowFrames; i++) {
      const x = w * 0.08 + i * 25;
      ctx.fillRect(x, slowY - 20, 18, 28);
      ctx.strokeRect(x, slowY - 20, 18, 28);
    }
    ctx.fillStyle = "hsla(220, 80%, 60%, 0.8)";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Slow: T/α frames", w * 0.15, slowY - 25);
    ctx.fillText("High channels", w * 0.15, slowY + 20);

    // Slow ResNet blocks
    for (let i = 0; i < 3; i++) {
      const bx = w * 0.3 + i * w * 0.15;
      ctx.fillStyle = "hsla(220, 80%, 55%, 0.08)";
      ctx.strokeStyle = "hsla(220, 80%, 55%, 0.4)";
      ctx.beginPath();
      ctx.roundRect(bx, slowY - 18, w * 0.1, 36, 4);
      ctx.fill(); ctx.stroke();
    }

    // Fast pathway (bottom)
    const fastY = cy + 55;
    ctx.fillStyle = "hsla(0, 80%, 55%, 0.1)";
    ctx.strokeStyle = "hsla(0, 80%, 55%, 0.5)";

    const fastFrames = 16;
    for (let i = 0; i < fastFrames; i++) {
      const x = w * 0.04 + i * 8;
      ctx.fillRect(x, fastY - 12, 5, 14);
      ctx.strokeRect(x, fastY - 12, 5, 14);
    }
    ctx.fillStyle = "hsla(0, 80%, 60%, 0.8)";
    ctx.font = "9px monospace";
    ctx.fillText("Fast: T frames", w * 0.1, fastY - 18);
    ctx.fillText("β=1/8 channels", w * 0.1, fastY + 15);

    for (let i = 0; i < 3; i++) {
      const bx = w * 0.3 + i * w * 0.15;
      ctx.fillStyle = "hsla(0, 80%, 55%, 0.08)";
      ctx.strokeStyle = "hsla(0, 80%, 55%, 0.4)";
      ctx.beginPath();
      ctx.roundRect(bx, fastY - 14, w * 0.1, 28, 4);
      ctx.fill(); ctx.stroke();
    }

    // Lateral connections (Fast → Slow)
    const lateralPhase = (Math.sin(t * 2) + 1) / 2;
    for (let i = 0; i < 3; i++) {
      const bx = w * 0.35 + i * w * 0.15;
      ctx.strokeStyle = `hsla(45, 90%, 55%, ${0.3 + lateralPhase * 0.4})`;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(bx, fastY - 14);
      ctx.lineTo(bx, slowY + 18);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = "hsla(45, 90%, 60%, 0.6)";
    ctx.fillText("Lateral connections ↑", w * 0.5, cy + 3);

    // Fusion + output
    const fusionX = w * 0.78;
    ctx.fillStyle = "hsla(120, 70%, 50%, 0.1)";
    ctx.strokeStyle = "hsla(120, 70%, 50%, 0.5)";
    ctx.beginPath();
    ctx.roundRect(fusionX, cy - 22, w * 0.12, 44, 6);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = "hsla(120, 70%, 55%, 0.9)";
    ctx.fillText("GAP + FC", fusionX + w * 0.06, cy + 4);

    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("Slow: spatial semantics | Fast: temporal dynamics", 10, h - 12);
  }, []);

  const draw = mode === "twostream" ? drawTwoStream : mode === "3dconv" ? draw3DConv : drawSlowFast;
  const canvasRef = useCanvas(draw, [mode]);

  return (
    <CanvasStage label="Action Recognition" modes={modes} activeMode={mode} onModeChange={setMode}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
    </CanvasStage>
  );
}
