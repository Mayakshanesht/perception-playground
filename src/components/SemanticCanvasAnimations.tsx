import { useEffect, useRef, useState, useCallback } from "react";

/* ══════════════════════════════════════════════════════
   SHARED UTILS
   ══════════════════════════════════════════════════════ */

function useCanvas(draw: (ctx: CanvasRenderingContext2D, t: number, w: number, h: number) => void) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const dpr = window.devicePixelRatio || 1;
      const w = parent.clientWidth;
      const h = parent.clientHeight || 360;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.scale(dpr, dpr);
    };
    resize();

    let raf: number;
    const loop = () => {
      const parent = canvas.parentElement;
      const w = parent?.clientWidth || 600;
      const h = parent?.clientHeight || 360;
      tRef.current += 0.012;
      ctx.clearRect(0, 0, w, h);
      draw(ctx, tRef.current, w, h);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    window.addEventListener("resize", resize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [draw]);

  return canvasRef;
}

function CanvasWrap({ children, height = "360px", label, hint }: { children: React.ReactNode; height?: string; label?: string; hint?: string }) {
  return (
    <div className="relative rounded-lg border border-border overflow-hidden bg-[#0A0C15]" style={{ height }}>
      {children}
      {label && (
        <span className="absolute top-3 left-3 text-[10px] font-mono uppercase tracking-wider text-primary/70 bg-background/80 px-2 py-0.5 rounded border border-border">
          {label}
        </span>
      )}
      {hint && (
        <span className="absolute bottom-2 right-3 text-[8px] font-mono text-muted-foreground/40 uppercase">
          {hint}
        </span>
      )}
    </div>
  );
}

function ModeButtons({ modes, active, onChange }: { modes: { id: string; label: string }[]; active: string; onChange: (id: string) => void }) {
  return (
    <div className="flex gap-1.5 flex-wrap mt-2">
      {modes.map((m) => (
        <button
          key={m.id}
          onClick={() => onChange(m.id)}
          className={`text-[9px] font-mono uppercase tracking-wider px-2.5 py-1.5 rounded border transition-colors ${
            active === m.id
              ? "text-primary border-primary bg-primary/5"
              : "text-muted-foreground border-border hover:text-foreground hover:border-muted-foreground"
          }`}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   1. CNN ARCHITECTURE — 3D Forward Pass
   ══════════════════════════════════════════════════════ */

const CNN_MODES = [
  { id: "full", label: "Full Forward Pass" },
  { id: "conv", label: "Conv Layer" },
  { id: "pool", label: "Pooling" },
  { id: "res", label: "ResNet Skip" },
];

const CNN_LAYERS = [
  { label: "Input\n224×224×3", color: "#00d4ff", w: 24, h: 90, d: 24 },
  { label: "Conv1\n112×112×64", color: "#22d3ee", w: 22, h: 80, d: 50 },
  { label: "Pool\n56×56×64", color: "#6366f1", w: 18, h: 60, d: 46 },
  { label: "Conv2\n56×56×128", color: "#8b5cf6", w: 16, h: 54, d: 60 },
  { label: "Conv3\n28×28×256", color: "#a855f7", w: 14, h: 44, d: 72 },
  { label: "Conv4\n14×14×512", color: "#c084fc", w: 12, h: 30, d: 64 },
  { label: "FC/GAP\n1024", color: "#ff6b35", w: 10, h: 16, d: 14 },
  { label: "Softmax\nK classes", color: "#10b981", w: 8, h: 10, d: 8 },
];

export function CNNArchitectureCanvas() {
  const [mode, setMode] = useState("full");
  const modeRef = useRef(mode);
  modeRef.current = mode;

  const draw = useCallback((ctx: CanvasRenderingContext2D, t: number, W: number, H: number) => {
    const cx = W / 2, cy = H / 2 - 10;
    const m = modeRef.current;

    const totalW = CNN_LAYERS.reduce((s, l) => s + l.d + 22, 0);
    let startX = cx - totalW / 2;

    CNN_LAYERS.forEach((layer, i) => {
      const progress = (t * 0.4) % 1;
      const activeLayer = Math.floor(progress * CNN_LAYERS.length);
      const pulse = m === "full" && i === activeLayer ? Math.sin(t * 4) * 0.5 + 0.5 : 0;
      const alpha = m === "full" ? 1 :
        m === "conv" && (i === 1 || i === 3 || i === 4 || i === 5) ? 1 :
        m === "pool" && i === 2 ? 1 :
        m === "res" && (i === 1 || i === 3) ? 1 : 0.25;

      // Draw stacked rects for 3D illusion
      ctx.save();
      ctx.globalAlpha = alpha;
      const px = pulse * 3;
      for (let j = layer.d; j >= 0; j -= 3) {
        const ox = j * 0.5, oy = -j * 0.5;
        ctx.beginPath();
        ctx.rect(startX + ox - (layer.w + px) / 2, cy + oy - (layer.h + px) / 2, layer.w + px, layer.h + px);
        const fillAlpha = Math.floor(15 + ((layer.d - j) / layer.d) * 30);
        ctx.fillStyle = layer.color + fillAlpha.toString(16).padStart(2, "0");
        ctx.fill();
        ctx.strokeStyle = layer.color + "60";
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }

      // Label
      ctx.fillStyle = "#94a3b8";
      ctx.font = '9px monospace';
      ctx.textAlign = "center";
      const lines = layer.label.split("\n");
      lines.forEach((ln, li) => ctx.fillText(ln, startX + layer.d * 0.25, cy + layer.h / 2 + 16 + li * 11));

      // Arrow to next
      if (i < CNN_LAYERS.length - 1) {
        const nextX = startX + layer.d + 22;
        const arrowY = cy - 5;
        ctx.strokeStyle = m === "res" && i === 1 ? "#7c3aed" : "#2d3f55";
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.6;
        ctx.beginPath();
        ctx.moveTo(startX + layer.d + 4, arrowY);
        ctx.lineTo(nextX - 4, arrowY);
        ctx.stroke();
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath();
        ctx.moveTo(nextX - 8, arrowY - 3);
        ctx.lineTo(nextX, arrowY);
        ctx.lineTo(nextX - 8, arrowY + 3);
        ctx.fill();

        // ResNet skip arc
        if (m === "res" && i === 1) {
          ctx.strokeStyle = "#7c3aed";
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 3]);
          ctx.globalAlpha = 0.8;
          const sx = startX + layer.d / 2;
          const ex = startX + CNN_LAYERS[1].d + 22 + CNN_LAYERS[2].d + 22 + CNN_LAYERS[3].d / 2;
          const arcY = cy - layer.h / 2 - 25;
          ctx.beginPath();
          ctx.moveTo(sx, cy - layer.h / 2);
          ctx.lineTo(sx, arcY);
          ctx.lineTo(ex, arcY);
          ctx.lineTo(ex, cy - CNN_LAYERS[3].h / 2);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = "#7c3aed";
          ctx.font = "bold 10px monospace";
          ctx.textAlign = "center";
          ctx.fillText("skip +", (sx + ex) / 2, arcY - 6);
        }
      }

      ctx.restore();
      startX += layer.d + 22;
    });

    const modeLabels: Record<string, string> = {
      full: "Forward pass sweep", conv: "Convolutional layers", pool: "Pooling / downsampling", res: "ResNet skip connection"
    };
    ctx.fillStyle = "#64748b";
    ctx.font = '10px monospace';
    ctx.textAlign = "center";
    ctx.fillText(modeLabels[m] || "", W / 2, H - 12);
  }, []);

  const canvasRef = useCanvas(draw);

  return (
    <div>
      <CanvasWrap label="Interactive 3D CNN" hint="Auto-animated" height="340px">
        <canvas ref={canvasRef} className="block w-full h-full" />
      </CanvasWrap>
      <ModeButtons modes={CNN_MODES} active={mode} onChange={setMode} />
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   2. CONVOLUTION FILTER SWEEP
   ══════════════════════════════════════════════════════ */

export function ConvFilterCanvas() {
  const colors = useRef(Array.from({ length: 64 }, () => Math.random()));

  const draw = useCallback((ctx: CanvasRenderingContext2D, t: number, W: number, H: number) => {
    const GRID = 8, CELL = Math.min(26, (W - 160) / (GRID + 8));
    const ox = (W - GRID * CELL * 2 - 60) / 2;
    const oy = (H - GRID * CELL) / 2;

    // Input grid
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        const v = colors.current[r * GRID + c];
        const pulse = (Math.sin(r * 0.7 + c * 0.5 + t * 0.5) + 1) / 2 * 0.3;
        ctx.fillStyle = `rgba(0, ${Math.floor(100 + (v + pulse) * 100)}, ${Math.floor(180 + v * 75)}, 0.6)`;
        ctx.fillRect(ox + c * CELL + 1, oy + r * CELL + 1, CELL - 2, CELL - 2);
      }
    }
    ctx.strokeStyle = "#1e2d45";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= GRID; i++) {
      ctx.beginPath(); ctx.moveTo(ox + i * CELL, oy); ctx.lineTo(ox + i * CELL, oy + GRID * CELL); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ox, oy + i * CELL); ctx.lineTo(ox + GRID * CELL, oy + i * CELL); ctx.stroke();
    }

    // Sliding 3×3 kernel
    const speed = GRID - 2;
    const pos = (t * 0.8) % 1;
    const kc = Math.floor(pos * speed * speed) % speed;
    const kr = Math.floor((pos * speed * speed) / speed) % speed;
    ctx.strokeStyle = "#00d4ff";
    ctx.lineWidth = 2;
    ctx.shadowColor = "#00d4ff";
    ctx.shadowBlur = 8;
    ctx.strokeRect(ox + kc * CELL, oy + kr * CELL, CELL * 3, CELL * 3);
    ctx.shadowBlur = 0;
    ctx.fillStyle = "rgba(0,212,255,0.08)";
    ctx.fillRect(ox + kc * CELL, oy + kr * CELL, CELL * 3, CELL * 3);
    ctx.fillStyle = "#00d4ff";
    ctx.font = '9px monospace';
    ctx.textAlign = "center";
    ctx.fillText("3×3 kernel", ox + (kc + 1.5) * CELL, oy + kr * CELL - 5);

    // Arrow
    const ax = ox + GRID * CELL + 18;
    ctx.strokeStyle = "#2d3f55";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ax, H / 2); ctx.lineTo(ax + 16, H / 2); ctx.stroke();
    ctx.fillStyle = "#2d3f55";
    ctx.beginPath(); ctx.moveTo(ax + 12, H / 2 - 4); ctx.lineTo(ax + 20, H / 2); ctx.lineTo(ax + 12, H / 2 + 4); ctx.fill();

    // Output grid (6×6)
    const ox2 = ax + 28;
    const oy2 = oy + CELL * 1;
    const OCELL = CELL;
    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 6; c++) {
        const active = r === kr && c === kc;
        const wave = active ? 1 : (Math.sin(r * 0.5 + c * 0.7 + t * 0.3) + 1) / 2 * 0.5;
        ctx.fillStyle = active ? "rgba(0,212,255,0.4)" : `rgba(99,102,241,${0.15 + wave * 0.3})`;
        ctx.fillRect(ox2 + c * OCELL + 1, oy2 + r * OCELL + 1, OCELL - 2, OCELL - 2);
        ctx.strokeStyle = active ? "#00d4ff" : "#1e2d45";
        ctx.lineWidth = active ? 1.5 : 0.5;
        ctx.strokeRect(ox2 + c * OCELL, oy2 + r * OCELL, OCELL, OCELL);
      }
    }
    ctx.fillStyle = "#6366f1";
    ctx.font = '10px monospace';
    ctx.textAlign = "center";
    ctx.fillText("Feature map out", ox2 + 3 * OCELL, oy2 - 8);
  }, []);

  const canvasRef = useCanvas(draw);

  return (
    <CanvasWrap label="Convolution Filter Sweep" height="280px">
      <canvas ref={canvasRef} className="block w-full h-full" />
    </CanvasWrap>
  );
}

/* ══════════════════════════════════════════════════════
   3. DETECTION PIPELINE — R-CNN / YOLO / FPN / NMS
   ══════════════════════════════════════════════════════ */

const DET_MODES = [
  { id: "rcnn", label: "Faster R-CNN" },
  { id: "yolo", label: "YOLO Grid" },
  { id: "fpn", label: "FPN Pyramid" },
  { id: "nms", label: "NMS Step" },
];

export function DetectionPipelineCanvas() {
  const [mode, setMode] = useState("rcnn");
  const modeRef = useRef(mode);
  modeRef.current = mode;

  const draw = useCallback((ctx: CanvasRenderingContext2D, t: number, W: number, H: number) => {
    const m = modeRef.current;
    if (m === "rcnn") drawRCNN(ctx, t, W, H);
    else if (m === "yolo") drawYOLO(ctx, t, W, H);
    else if (m === "fpn") drawFPN(ctx, t, W, H);
    else drawNMS(ctx, t, W, H);
  }, []);

  const canvasRef = useCanvas(draw);

  return (
    <div>
      <CanvasWrap label="Detector Comparison" hint="Animated" height="360px">
        <canvas ref={canvasRef} className="block w-full h-full" />
      </CanvasWrap>
      <ModeButtons modes={DET_MODES} active={mode} onChange={setMode} />
    </div>
  );
}

function drawRCNN(ctx: CanvasRenderingContext2D, t: number, W: number, H: number) {
  const stages = [
    { x: W * 0.1, label: "Input", color: "#00d4ff", w: 60, h: 70 },
    { x: W * 0.28, label: "Backbone", color: "#22d3ee", w: 50, h: 60 },
    { x: W * 0.44, label: "Feature\nMap", color: "#6366f1", w: 45, h: 55 },
    { x: W * 0.58, label: "RPN", color: "#8b5cf6", w: 40, h: 40 },
    { x: W * 0.72, label: "RoI\nAlign", color: "#ff6b35", w: 40, h: 40 },
    { x: W * 0.88, label: "Cls+Reg", color: "#10b981", w: 50, h: 45 },
  ];
  const cy = H / 2;
  const activeStage = Math.floor((t * 0.3) % stages.length);

  stages.forEach((s, i) => {
    const isActive = i === activeStage;
    ctx.save();
    ctx.shadowColor = s.color;
    ctx.shadowBlur = isActive ? 15 : 3;
    ctx.fillStyle = s.color + (isActive ? "30" : "15");
    ctx.strokeStyle = s.color;
    ctx.lineWidth = isActive ? 2 : 1;
    ctx.beginPath();
    ctx.roundRect(s.x - s.w / 2, cy - s.h / 2, s.w, s.h, 4);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = s.color;
    ctx.textAlign = "center";
    ctx.font = "bold 9px monospace";
    const lines = s.label.split("\n");
    lines.forEach((ln, li) => ctx.fillText(ln, s.x, cy - lines.length * 5 + li * 12 + 4));
    ctx.restore();

    if (i < stages.length - 1) {
      const flow = isActive ? Math.sin(t * 8) * 0.5 + 0.5 : 0;
      ctx.strokeStyle = isActive ? s.color : "#2d3f55";
      ctx.lineWidth = isActive ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(s.x + s.w / 2 + 2, cy);
      ctx.lineTo(stages[i + 1].x - stages[i + 1].w / 2 - 2, cy);
      ctx.stroke();
      if (isActive) {
        const px = s.x + s.w / 2 + (stages[i + 1].x - stages[i + 1].w / 2 - s.x - s.w / 2) * flow;
        ctx.beginPath();
        ctx.arc(px, cy, 3, 0, Math.PI * 2);
        ctx.fillStyle = s.color;
        ctx.fill();
      }
    }
  });

  ctx.fillStyle = "#64748b";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Two-stage: backbone → RPN proposals → RoI classify", W / 2, H - 12);
}

function drawYOLO(ctx: CanvasRenderingContext2D, t: number, W: number, H: number) {
  const S = 7, cellW = Math.min(40, (W - 80) / S), cellH = cellW;
  const gx = W / 2 - (S * cellW) / 2;
  const gy = H / 2 - (S * cellH) / 2;
  const activeCell = Math.floor((t * 0.6) % (S * S));
  const ac = activeCell % S, ar = Math.floor(activeCell / S);

  for (let r = 0; r < S; r++) {
    for (let c = 0; c < S; c++) {
      const isActive = r === ar && c === ac;
      ctx.fillStyle = isActive ? "rgba(255,107,53,0.2)" : "rgba(255,107,53,0.04)";
      ctx.fillRect(gx + c * cellW, gy + r * cellH, cellW - 1, cellH - 1);
      ctx.strokeStyle = isActive ? "#ff6b35" : "#1e2d45";
      ctx.lineWidth = isActive ? 1.5 : 0.5;
      ctx.strokeRect(gx + c * cellW, gy + r * cellH, cellW - 1, cellH - 1);

      if (isActive) {
        ctx.beginPath();
        ctx.arc(gx + c * cellW + cellW / 2, gy + r * cellH + cellH / 2, 3, 0, Math.PI * 2);
        ctx.fillStyle = "#ff6b35";
        ctx.fill();
        const cx2 = gx + c * cellW + cellW / 2;
        const cy2 = gy + r * cellH + cellH / 2;
        [[55, 45, 0.9], [70, 55, 0.7], [40, 35, 0.5]].forEach(([bw, bh, conf], bi) => {
          ctx.save();
          ctx.strokeStyle = `rgba(255,107,53,${(conf as number) * (0.5 + Math.sin(t * 3 + bi) * 0.3)})`;
          ctx.lineWidth = 1;
          ctx.strokeRect(cx2 - (bw as number) / 2, cy2 - (bh as number) / 2, bw as number, bh as number);
          ctx.restore();
        });
        ctx.fillStyle = "#ff6b35";
        ctx.font = "9px monospace";
        ctx.textAlign = "center";
        ctx.fillText(`cell(${ar},${ac})`, cx2, gy + r * cellH - 4);
      }
    }
  }

  ctx.fillStyle = "#64748b";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText(`YOLO ${S}×${S} grid — each cell predicts B=3 boxes × (4+1+K)`, W / 2, H - 12);
}

function drawFPN(ctx: CanvasRenderingContext2D, t: number, W: number, H: number) {
  const levels = [
    { label: "C2 (1/4)", w: 90, h: 70, color: "#00d4ff", scale: "Large" },
    { label: "C3 (1/8)", w: 70, h: 55, color: "#22d3ee", scale: "Medium" },
    { label: "C4 (1/16)", w: 50, h: 40, color: "#6366f1", scale: "Small" },
    { label: "C5 (1/32)", w: 35, h: 28, color: "#8b5cf6", scale: "Tiny" },
  ];
  const cx = W / 2;
  const spacing = Math.min(120, (W - 200) / 4);
  const baseY = H - 70;

  levels.forEach((lv, i) => {
    const x = cx - 1.5 * spacing + i * spacing - 60;
    ctx.fillStyle = lv.color + "20";
    ctx.strokeStyle = lv.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(x - lv.w / 2, baseY - lv.h, lv.w, lv.h, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = lv.color;
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText(lv.label, x, baseY + 14);

    // FPN side
    const fx = cx - 1.5 * spacing + i * spacing + 60;
    const sc = 1 - i * 0.15;
    const pulse = Math.abs(Math.sin(t * 1.5 + i * 0.6));
    ctx.save();
    ctx.shadowColor = lv.color;
    ctx.shadowBlur = pulse * 12;
    ctx.fillStyle = lv.color + "20";
    ctx.strokeStyle = lv.color;
    ctx.beginPath();
    ctx.roundRect(fx - (lv.w * sc) / 2, baseY - lv.h * sc, lv.w * sc, lv.h * sc, 4);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.restore();
    ctx.fillStyle = "#10b981";
    ctx.fillText("P" + (i + 2), fx, baseY + 14);
    ctx.fillStyle = "#64748b";
    ctx.fillText(lv.scale, fx, baseY + 26);
  });

  ctx.fillStyle = "#64748b";
  ctx.font = "10px monospace";
  ctx.textAlign = "left";
  ctx.fillText("← bottom-up backbone", 16, H - 44);
  ctx.fillStyle = "#ff6b35";
  ctx.fillText("top-down FPN →", 16, H - 32);
}

function drawNMS(ctx: CanvasRenderingContext2D, t: number, W: number, H: number) {
  const boxes = [
    { x: W * 0.2, y: H * 0.3, w: 120, h: 100, conf: 0.92, keep: true },
    { x: W * 0.23, y: H * 0.35, w: 110, h: 90, conf: 0.78, keep: false },
    { x: W * 0.18, y: H * 0.28, w: 130, h: 105, conf: 0.65, keep: false },
    { x: W * 0.55, y: H * 0.28, w: 100, h: 85, conf: 0.88, keep: true },
    { x: W * 0.57, y: H * 0.33, w: 90, h: 80, conf: 0.71, keep: false },
    { x: W * 0.78, y: H * 0.35, w: 80, h: 70, conf: 0.82, keep: true },
  ];

  const step = Math.floor(t * 0.7) % 4;
  boxes.forEach((b) => {
    const suppressed = !b.keep && step > 0;
    ctx.save();
    ctx.globalAlpha = suppressed ? 0.3 : 1;
    ctx.strokeStyle = b.keep ? "#10b981" : "#ef4444";
    ctx.lineWidth = b.keep ? 2 : 1;
    if (b.keep) {
      ctx.shadowColor = "#10b981";
      ctx.shadowBlur = 10;
    }
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    ctx.fillStyle = (b.keep ? "#10b981" : "#ef4444") + "15";
    ctx.fillRect(b.x, b.y, b.w, b.h);
    ctx.shadowBlur = 0;
    ctx.fillStyle = b.keep ? "#10b981" : "#ef444480";
    ctx.font = "bold 11px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`${(b.conf * 100).toFixed(0)}%`, b.x + 4, b.y + 14);
    if (suppressed) {
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(b.x, b.y);
      ctx.lineTo(b.x + b.w, b.y + b.h);
      ctx.stroke();
    }
    ctx.restore();
  });

  ctx.fillStyle = "#64748b";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Green = kept (highest conf) │ Red = suppressed (IoU overlap)", W / 2, H - 12);
}

/* ══════════════════════════════════════════════════════
   4. SEGMENTATION ARCHITECTURES — U-Net / DeepLab / Mask R-CNN
   ══════════════════════════════════════════════════════ */

const SEG_MODES = [
  { id: "unet", label: "U-Net" },
  { id: "deeplab", label: "DeepLab ASPP" },
  { id: "maskrcnn", label: "Mask R-CNN" },
];

export function SegmentationArchCanvas() {
  const [mode, setMode] = useState("unet");
  const modeRef = useRef(mode);
  modeRef.current = mode;

  const draw = useCallback((ctx: CanvasRenderingContext2D, t: number, W: number, H: number) => {
    const m = modeRef.current;
    if (m === "unet") drawUNet(ctx, t, W, H);
    else if (m === "deeplab") drawDeepLab(ctx, t, W, H);
    else drawMaskRCNN(ctx, t, W, H);
  }, []);

  const canvasRef = useCanvas(draw);

  return (
    <div>
      <CanvasWrap label="Architecture Visualization" hint="Animated" height="380px">
        <canvas ref={canvasRef} className="block w-full h-full" />
      </CanvasWrap>
      <ModeButtons modes={SEG_MODES} active={mode} onChange={setMode} />
    </div>
  );
}

function drawUNet(ctx: CanvasRenderingContext2D, t: number, W: number, H: number) {
  const levels = [
    { w: 100, h: 65, color: "#00d4ff" },
    { w: 75, h: 50, color: "#22d3ee" },
    { w: 55, h: 38, color: "#6366f1" },
    { w: 42, h: 26, color: "#8b5cf6" },
  ];
  const cx = W / 2;
  const baseY = H - 50;
  const enc_x = cx - W * 0.22;
  const dec_x = cx + W * 0.08;
  const spacing = 72;

  ctx.fillStyle = "#00d4ff80";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText("ENCODER", enc_x - 40, 28);
  ctx.fillStyle = "#10b98180";
  ctx.fillText("DECODER", dec_x + 40, 28);

  levels.forEach((lv, i) => {
    const y = baseY - i * spacing;
    // Encoder
    ctx.fillStyle = lv.color + "20";
    ctx.strokeStyle = lv.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(enc_x - lv.w / 2, y - lv.h / 2, lv.w, lv.h, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = lv.color;
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText("E" + (levels.length - i), enc_x, y + 4);

    // Skip connection
    const pulse = (Math.sin(t * 2 + i * 0.7) + 1) / 2;
    ctx.save();
    ctx.strokeStyle = "#ff6b35";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.globalAlpha = 0.5 + pulse * 0.4;
    ctx.beginPath();
    ctx.moveTo(enc_x + lv.w / 2 + 4, y);
    ctx.lineTo(dec_x - lv.w / 2 - 4, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    if (i === 2) {
      ctx.fillStyle = "#ff6b3580";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText("skip ⊕", cx, y - 8);
    }

    // Decoder
    ctx.fillStyle = lv.color + "20";
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(dec_x - lv.w / 2, y - lv.h / 2, lv.w, lv.h, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#10b981";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText("D" + (levels.length - i), dec_x, y + 4);

    // Arrows between levels
    if (i < levels.length - 1) {
      const ny = baseY - (i + 1) * spacing;
      ctx.strokeStyle = "#00d4ff50";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(enc_x, y - lv.h / 2 - 2);
      ctx.lineTo(enc_x, ny + levels[i + 1].h / 2 + 2);
      ctx.stroke();
    }
  });

  // Bottleneck
  const by = baseY - levels.length * spacing;
  ctx.fillStyle = "#ff6b3530";
  ctx.strokeStyle = "#ff6b35";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(cx - 50, by - 16, 100, 32, 6);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#ff6b35";
  ctx.font = "bold 9px monospace";
  ctx.textAlign = "center";
  ctx.fillText("BOTTLENECK", cx, by + 4);

  ctx.fillStyle = "#64748b";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Dₗ = Conv(Concat(Up(Dₗ₊₁), Eₗ))", W / 2, H - 10);
}

function drawDeepLab(ctx: CanvasRenderingContext2D, t: number, W: number, H: number) {
  const cx = W / 2, cy = H / 2;
  const rates = [1, 6, 12, 18];
  const colors = ["#00d4ff", "#22d3ee", "#6366f1", "#8b5cf6"];

  // Input
  ctx.fillStyle = "#00d4ff20";
  ctx.strokeStyle = "#00d4ff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(cx - W * 0.38 - 30, cy - 22, 60, 44, 4);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#00d4ff";
  ctx.font = "9px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Feature", cx - W * 0.38, cy - 4);
  ctx.fillText("Map", cx - W * 0.38, cy + 8);

  rates.forEach((r, i) => {
    const ax = cx - 40;
    const ay = cy - 50 + i * 34;
    const bw = 75, bh = 24;
    const pulse = (Math.sin(t * 1.5 + i * 0.8) + 1) / 2;
    ctx.save();
    ctx.shadowColor = colors[i];
    ctx.shadowBlur = pulse * 12;
    ctx.fillStyle = colors[i] + "20";
    ctx.strokeStyle = colors[i];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(ax - bw / 2, ay - bh / 2, bw, bh, 4);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = colors[i];
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`Atrous r=${r}`, ax, ay + 3);

    // Lines
    ctx.strokeStyle = colors[i] + "60";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - W * 0.38 + 30, cy);
    ctx.lineTo(ax - bw / 2, ay);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(ax + bw / 2, ay);
    ctx.lineTo(cx + W * 0.12, cy);
    ctx.stroke();
    ctx.restore();
  });

  // Concat box
  ctx.fillStyle = "#10b98120";
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(cx + W * 0.12, cy - 25, 60, 50, 6);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#10b981";
  ctx.font = "bold 9px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Concat", cx + W * 0.12 + 30, cy - 4);
  ctx.fillText("1×1 Conv", cx + W * 0.12 + 30, cy + 8);

  ctx.fillStyle = "#64748b";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText("ASPP: Parallel atrous convolutions at r=1,6,12,18", W / 2, H - 10);
}

function drawMaskRCNN(ctx: CanvasRenderingContext2D, t: number, W: number, H: number) {
  const rois = [
    { x: W * 0.1, y: H * 0.3, w: 100, h: 80, cls: "person", color: "#00d4ff", mask: [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]] },
    { x: W * 0.38, y: H * 0.35, w: 90, h: 72, cls: "car", color: "#ff6b35", mask: [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 0, 0]] },
    { x: W * 0.65, y: H * 0.32, w: 85, h: 78, cls: "dog", color: "#10b981", mask: [[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 0, 0]] },
  ];

  ctx.fillStyle = "#0d1320";
  ctx.fillRect(16, H * 0.2, W - 32, H * 0.45);
  ctx.strokeStyle = "#1e2d45";
  ctx.lineWidth = 1;
  ctx.strokeRect(16, H * 0.2, W - 32, H * 0.45);
  ctx.fillStyle = "#1e2d45";
  ctx.font = "9px monospace";
  ctx.textAlign = "left";
  ctx.fillText("Input + Backbone Feature Map", 24, H * 0.2 + 14);

  rois.forEach((roi, i) => {
    const pulse = (Math.sin(t * 2 + i * 1.2) + 1) / 2;
    ctx.save();
    ctx.shadowColor = roi.color;
    ctx.shadowBlur = pulse * 10;
    ctx.strokeStyle = roi.color;
    ctx.lineWidth = 2;
    ctx.strokeRect(roi.x, roi.y, roi.w, roi.h);

    const mc = 4;
    const mw = roi.w / mc, mh = roi.h / mc;
    roi.mask.forEach((row, r) =>
      row.forEach((v, c) => {
        if (v) {
          ctx.fillStyle = roi.color + "40";
          ctx.fillRect(roi.x + c * mw, roi.y + r * mh, mw - 1, mh - 1);
        }
      })
    );
    ctx.shadowBlur = 0;
    ctx.fillStyle = roi.color;
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText(roi.cls, roi.x + roi.w / 2, roi.y - 5);
    ctx.restore();
  });

  // Loss bars
  const losses = ["Lcls", "Lbox", "Lmask"];
  const lcolors = ["#00d4ff", "#ff6b35", "#10b981"];
  losses.forEach((ln, i) => {
    const x = W * 0.2 + i * W * 0.25;
    const y = H - 55;
    ctx.fillStyle = lcolors[i] + "20";
    ctx.strokeStyle = lcolors[i];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(x - 40, y - 16, 80, 32, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = lcolors[i];
    ctx.font = "bold 11px monospace";
    ctx.textAlign = "center";
    ctx.fillText(ln, x, y + 4);
  });

  ctx.fillStyle = "#64748b";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText("L = Lcls + Lbox + Lmask  (per-pixel BCE on binary mask)", W / 2, H - 10);
}

/* ══════════════════════════════════════════════════════
   5. VISION TRANSFORMER — Patches / Attention / MHA
   ══════════════════════════════════════════════════════ */

const VIT_MODES = [
  { id: "patch", label: "Patch Splitting" },
  { id: "attn", label: "Self-Attention" },
  { id: "mha", label: "Multi-Head" },
];

export function ViTCanvas() {
  const [mode, setMode] = useState("patch");
  const modeRef = useRef(mode);
  modeRef.current = mode;

  const patchColors = useRef(
    Array.from({ length: 16 }, (_, i) => {
      const hue = (i / 16) * 260 + 180;
      return `hsla(${hue}, 70%, 45%, 1)`;
    })
  );

  const draw = useCallback((ctx: CanvasRenderingContext2D, t: number, W: number, H: number) => {
    const m = modeRef.current;
    if (m === "patch") drawPatch(ctx, t, W, H, patchColors.current);
    else if (m === "attn") drawAttention(ctx, t, W, H, patchColors.current);
    else drawMHA(ctx, t, W, H);
  }, []);

  const canvasRef = useCanvas(draw);

  return (
    <div>
      <CanvasWrap label="Vision Transformer" hint="Animated" height="380px">
        <canvas ref={canvasRef} className="block w-full h-full" />
      </CanvasWrap>
      <ModeButtons modes={VIT_MODES} active={mode} onChange={setMode} />
    </div>
  );
}

function drawPatch(ctx: CanvasRenderingContext2D, t: number, W: number, H: number, colors: string[]) {
  const PATCHES = 4;
  const imgS = Math.min(140, H - 100);
  const cellS = imgS / PATCHES;
  const imgX = 40, imgY = (H - imgS) / 2;

  for (let r = 0; r < PATCHES; r++) {
    for (let c = 0; c < PATCHES; c++) {
      ctx.fillStyle = colors[r * PATCHES + c];
      ctx.fillRect(imgX + c * cellS, imgY + r * cellS, cellS - 1, cellS - 1);
    }
  }
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1;
  ctx.strokeRect(imgX, imgY, imgS, imgS);

  // Token sequence
  const seqX = Math.min(W * 0.45, 280);
  const seqTop = 30;
  const tokenH = 18, tokenW = 60, tokenGap = 3;
  const totalTokens = PATCHES * PATCHES + 1;
  const progress = (t * 0.15) % 1;
  const animToken = Math.floor(progress * totalTokens);

  // CLS
  ctx.fillStyle = "#ff6b3530";
  ctx.strokeStyle = "#ff6b35";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(seqX, seqTop, tokenW, tokenH, 3);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#ff6b35";
  ctx.font = "8px monospace";
  ctx.textAlign = "center";
  ctx.fillText("[CLS]", seqX + tokenW / 2, seqTop + tokenH / 2 + 3);

  for (let i = 0; i < PATCHES * PATCHES; i++) {
    const ty = seqTop + (i + 1) * (tokenH + tokenGap);
    if (ty > H - 20) break;
    const isDrawn = i < animToken - 1 || animToken === 0;

    if (isDrawn || animToken === 0) {
      ctx.fillStyle = colors[i].replace("1)", "0.3)");
      ctx.strokeStyle = colors[i];
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(seqX, ty, tokenW, tokenH, 3);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = colors[i];
      ctx.font = "8px monospace";
      ctx.textAlign = "center";
      ctx.fillText(`p${i + 1}`, seqX + tokenW / 2, ty + tokenH / 2 + 3);
    } else {
      ctx.fillStyle = "#1e2d4580";
      ctx.beginPath();
      ctx.roundRect(seqX, ty, tokenW, tokenH, 3);
      ctx.fill();
    }
  }

  // Positional embedding
  const embedX = seqX + tokenW + 20;
  ctx.fillStyle = "#10b98120";
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 1;
  const embedH = Math.min(totalTokens * (tokenH + tokenGap), H - 60);
  ctx.beginPath();
  ctx.roundRect(embedX, seqTop, tokenW, embedH, 4);
  ctx.fill();
  ctx.stroke();
  ctx.save();
  ctx.fillStyle = "#10b981";
  ctx.font = "9px monospace";
  ctx.textAlign = "center";
  ctx.translate(embedX + tokenW / 2, seqTop + embedH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("+ Position Embedding", 0, 0);
  ctx.restore();

  ctx.fillStyle = "#64748b";
  ctx.font = "10px monospace";
  ctx.textAlign = "left";
  ctx.fillText(`N = ${PATCHES}×${PATCHES} = ${PATCHES * PATCHES} patches + [CLS]`, 40, H - 10);
}

function drawAttention(ctx: CanvasRenderingContext2D, t: number, W: number, H: number, pColors: string[]) {
  const N = 8;
  const R = Math.min(130, W / 4, H / 3);
  const cx = W / 2, cy = H / 2 + 10;

  const tokenPos = Array.from({ length: N }, (_, i) => ({
    x: cx + R * Math.cos((i / N) * Math.PI * 2 - Math.PI / 2),
    y: cy + R * Math.sin((i / N) * Math.PI * 2 - Math.PI / 2),
    label: i === 0 ? "CLS" : `p${i}`,
    color: i === 0 ? "#ff6b35" : pColors[i * 2] || "#6366f1",
  }));

  const queryToken = Math.floor(t * 0.4) % N;
  const attnWeights = tokenPos.map((_, j) => {
    if (j === queryToken) return 1;
    return Math.max(0, 0.3 + 0.4 * Math.cos((j - queryToken) * 1.1 + t * 0.2));
  });

  // Attention lines
  tokenPos.forEach((tp, j) => {
    if (j === queryToken) return;
    const w = attnWeights[j];
    ctx.save();
    ctx.globalAlpha = w * 0.7;
    ctx.strokeStyle = tokenPos[queryToken].color;
    ctx.lineWidth = w * 3;
    ctx.beginPath();
    ctx.moveTo(tokenPos[queryToken].x, tokenPos[queryToken].y);
    ctx.lineTo(tp.x, tp.y);
    ctx.stroke();
    ctx.restore();
  });

  // Token circles
  tokenPos.forEach((tp, j) => {
    const isQuery = j === queryToken;
    ctx.save();
    ctx.shadowColor = tp.color;
    ctx.shadowBlur = isQuery ? 15 : 3;
    ctx.beginPath();
    ctx.arc(tp.x, tp.y, isQuery ? 18 : 14, 0, Math.PI * 2);
    ctx.fillStyle = tp.color + (isQuery ? "40" : "20");
    ctx.fill();
    ctx.strokeStyle = tp.color;
    ctx.lineWidth = isQuery ? 2 : 1;
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = isQuery ? "#fff" : tp.color;
    ctx.font = `${isQuery ? "bold " : ""}10px monospace`;
    ctx.textAlign = "center";
    ctx.fillText(tp.label, tp.x, tp.y + 3);
    ctx.restore();
  });

  ctx.fillStyle = tokenPos[queryToken].color;
  ctx.font = "bold 11px monospace";
  ctx.textAlign = "center";
  ctx.fillText(`Query: ${tokenPos[queryToken].label} attending to all keys`, cx, 28);
  ctx.fillStyle = "#64748b";
  ctx.font = "10px monospace";
  ctx.fillText("Attn(Q,K,V) = softmax(QKᵀ/√dₖ)·V", cx, H - 10);
}

function drawMHA(ctx: CanvasRenderingContext2D, t: number, W: number, H: number) {
  const HEADS = 4;
  const headW = (W - 60) / HEADS;
  const headColors = ["#00d4ff", "#ff6b35", "#6366f1", "#10b981"];

  Array.from({ length: HEADS }, (_, h) => {
    const hx = 30 + h * headW + headW / 2;
    const pulse = (Math.sin(t * 1.5 + h * 0.9) + 1) / 2;

    ctx.save();
    ctx.shadowColor = headColors[h];
    ctx.shadowBlur = pulse * 12;
    ctx.fillStyle = headColors[h] + "15";
    ctx.strokeStyle = headColors[h];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(hx - 50, 60, 100, 140, 8);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.fillStyle = headColors[h];
    ctx.font = "bold 10px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`Head ${h + 1}`, hx, 80);

    ["Q", "K", "V"].forEach((lbl, li) => {
      const vy = 100 + li * 28;
      ctx.fillStyle = headColors[h] + "30";
      ctx.beginPath();
      ctx.roundRect(hx - 22, vy, 44, 20, 3);
      ctx.fill();
      ctx.fillStyle = headColors[h];
      ctx.font = "10px monospace";
      ctx.fillText(`W${lbl}`, hx, vy + 14);
    });
    ctx.restore();
  });

  // Concat
  ctx.fillStyle = "#7c3aed20";
  ctx.strokeStyle = "#7c3aed";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(W / 2 - 80, 220, 160, 36, 6);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#a78bfa";
  ctx.font = "bold 10px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Concat(head₁,…,headₕ) · Wᴼ", W / 2, 243);

  // Input tokens
  const N = 6, tokenW = 36, tokGap = 6;
  const tokRowX = (W - N * (tokenW + tokGap)) / 2;
  for (let i = 0; i < N; i++) {
    ctx.fillStyle = "#1e2d45";
    ctx.strokeStyle = "#2d3f55";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(tokRowX + i * (tokenW + tokGap), H - 55, tokenW, 20, 3);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#64748b";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`x${i + 1}`, tokRowX + i * (tokenW + tokGap) + tokenW / 2, H - 42);
  }

  ctx.fillStyle = "#64748b";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Each head learns different attention patterns in parallel", W / 2, H - 10);
}
