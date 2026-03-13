import { useEffect, useRef, useState, useCallback } from "react";

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
      tRef.current += 0.015;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      draw(ctx, canvas.width, canvas.height, tRef.current);
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
          <span className="text-[10px] font-mono font-bold uppercase tracking-wider" style={{ color: "hsl(290, 70%, 55%)" }}>{label}</span>
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
   1. CLIP EMBEDDING SPACE
   ═══════════════════════════════════════════ */

export function CLIPEmbeddingCanvas() {
  const [mode, setMode] = useState("contrastive");
  const modes = [
    { id: "contrastive", label: "Contrastive Learning" },
    { id: "embedding", label: "Shared Embedding" },
    { id: "zeroshot", label: "Zero-Shot Transfer" },
  ];

  const drawContrastive = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    // Image encoder (left), Text encoder (right)
    const imgX = w * 0.12;
    const txtX = w * 0.62;
    const cy = h * 0.4;
    const boxW = w * 0.2;
    const boxH = 50;

    // Image encoder
    ctx.fillStyle = "hsla(220, 80%, 55%, 0.1)";
    ctx.strokeStyle = "hsla(220, 80%, 55%, 0.6)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(imgX, cy - boxH / 2, boxW, boxH, 6);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = "hsla(220, 80%, 60%, 0.9)";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Image Encoder", imgX + boxW / 2, cy - 3);
    ctx.fillText("(ViT / ResNet)", imgX + boxW / 2, cy + 10);

    // Text encoder
    ctx.fillStyle = "hsla(290, 70%, 55%, 0.1)";
    ctx.strokeStyle = "hsla(290, 70%, 55%, 0.6)";
    ctx.beginPath();
    ctx.roundRect(txtX, cy - boxH / 2, boxW, boxH, 6);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = "hsla(290, 70%, 60%, 0.9)";
    ctx.fillText("Text Encoder", txtX + boxW / 2, cy - 3);
    ctx.fillText("(Transformer)", txtX + boxW / 2, cy + 10);

    // Similarity matrix below
    const matY = h * 0.65;
    const matSize = Math.min(w * 0.3, h * 0.28);
    const matX = (w - matSize) / 2;
    const N = 5;
    const cellSize = matSize / N;

    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "9px monospace";
    ctx.fillText("Image embeddings →", matX + matSize / 2, matY - 8);

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const isMatch = i === j;
        const sim = isMatch ? 0.85 + Math.sin(t + i) * 0.1 : 0.1 + Math.abs(Math.sin(t * 0.5 + i + j * 3)) * 0.2;
        const hue = isMatch ? 120 : 0;
        const lightness = isMatch ? 50 : 35;
        ctx.fillStyle = `hsla(${hue}, 70%, ${lightness}%, ${sim})`;
        ctx.fillRect(matX + j * cellSize, matY + i * cellSize, cellSize - 1, cellSize - 1);

        if (isMatch) {
          ctx.strokeStyle = `hsla(120, 70%, 50%, 0.6)`;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(matX + j * cellSize, matY + i * cellSize, cellSize - 1, cellSize - 1);
        }
      }
    }

    // Arrows from encoders to matrix
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(imgX + boxW / 2, cy + boxH / 2);
    ctx.lineTo(matX + matSize / 4, matY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(txtX + boxW / 2, cy + boxH / 2);
    ctx.lineTo(matX + matSize * 3 / 4, matY);
    ctx.stroke();

    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("Diagonal = matching pairs (maximize) | Off-diagonal = negatives (minimize)", 10, h - 12);
  }, []);

  const drawEmbedding = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) * 0.3;

    // Embedding space circle
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();

    // Points: images (blue) and texts (purple) — matched pairs are close
    const pairs = [
      { angle: 0.5, label: "🐱", text: "\"a cat\"" },
      { angle: 1.8, label: "🚗", text: "\"a red car\"" },
      { angle: 3.2, label: "🌅", text: "\"sunset\"" },
      { angle: 4.5, label: "🏠", text: "\"house\"" },
      { angle: 5.5, label: "🐕", text: "\"a dog\"" },
    ];

    pairs.forEach((p, i) => {
      const a = p.angle + Math.sin(t * 0.5 + i) * 0.1;
      const r = radius * (0.6 + Math.sin(t * 0.3 + i * 2) * 0.15);

      const imgPx = cx + Math.cos(a) * r;
      const imgPy = cy + Math.sin(a) * r;

      // Text point (nearby with small offset)
      const offset = 0.08 + Math.sin(t * 2 + i) * 0.03;
      const txtPx = cx + Math.cos(a + offset) * (r + 8);
      const txtPy = cy + Math.sin(a + offset) * (r + 8);

      // Connection line
      ctx.strokeStyle = "hsla(120, 70%, 50%, 0.3)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(imgPx, imgPy);
      ctx.lineTo(txtPx, txtPy);
      ctx.stroke();

      // Image point
      ctx.fillStyle = "hsla(220, 80%, 55%, 0.9)";
      ctx.beginPath();
      ctx.arc(imgPx, imgPy, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = "12px serif";
      ctx.textAlign = "center";
      ctx.fillText(p.label, imgPx, imgPy - 10);

      // Text point
      ctx.fillStyle = "hsla(290, 70%, 55%, 0.9)";
      ctx.beginPath();
      ctx.arc(txtPx, txtPy, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = "8px monospace";
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.fillText(p.text, txtPx, txtPy + 16);
    });

    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("🔵 Image embeddings  🟣 Text embeddings  — aligned via contrastive loss", 10, h - 12);
  }, []);

  const drawZeroShot = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    // Image on left
    const imgX = w * 0.05;
    const imgY = h * 0.2;
    const imgSize = Math.min(w * 0.2, h * 0.4);

    ctx.strokeStyle = "hsla(220, 80%, 55%, 0.5)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(imgX, imgY, imgSize, imgSize);
    ctx.font = "40px serif";
    ctx.textAlign = "center";
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fillText("🐱", imgX + imgSize / 2, imgY + imgSize / 2 + 14);

    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "9px monospace";
    ctx.fillText("Test Image", imgX + imgSize / 2, imgY - 8);

    // Class prompts on right
    const classes = [
      { text: "\"a photo of a cat\"", score: 0.92 },
      { text: "\"a photo of a dog\"", score: 0.05 },
      { text: "\"a photo of a car\"", score: 0.02 },
      { text: "\"a photo of a bird\"", score: 0.01 },
    ];

    const listX = w * 0.45;
    const listY = h * 0.18;
    const rowH = 48;

    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText("Text prompts (class descriptions):", listX, listY - 10);

    classes.forEach((cls, i) => {
      const y = listY + i * rowH;
      const barW = cls.score * w * 0.3;

      // Score bar
      const hue = lerp(0, 120, cls.score);
      ctx.fillStyle = `hsla(${hue}, 70%, 50%, 0.2)`;
      ctx.fillRect(listX, y, barW, 30);
      ctx.strokeStyle = `hsla(${hue}, 70%, 50%, 0.5)`;
      ctx.lineWidth = 1;
      ctx.strokeRect(listX, y, barW, 30);

      // Text
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.font = "9px monospace";
      ctx.fillText(cls.text, listX + 5, y + 13);
      ctx.fillStyle = `hsla(${hue}, 70%, 60%, 0.9)`;
      ctx.font = "bold 11px monospace";
      ctx.fillText(`${(cls.score * 100).toFixed(0)}%`, listX + barW + 8, y + 20);
    });

    // Arrow from image to list
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(imgX + imgSize, imgY + imgSize / 2);
    ctx.lineTo(listX - 10, listY + rowH);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "hsla(290, 70%, 55%, 0.6)";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText("sim(v, t) = v·t / (||v||·||t||)", w / 2, h - 30);

    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.textAlign = "left";
    ctx.fillText("Zero-shot: no task-specific training — just compare embeddings", 10, h - 12);
  }, []);

  const draw = mode === "contrastive" ? drawContrastive : mode === "embedding" ? drawEmbedding : drawZeroShot;
  const canvasRef = useCanvas(draw, [mode]);

  return (
    <CanvasStage label="CLIP & Contrastive Learning" modes={modes} activeMode={mode} onModeChange={setMode}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
    </CanvasStage>
  );
}

/* ═══════════════════════════════════════════
   2. VLM ARCHITECTURE
   ═══════════════════════════════════════════ */

export function VLMArchitectureCanvas() {
  const [mode, setMode] = useState("llava");
  const modes = [
    { id: "llava", label: "LLaVA Pipeline" },
    { id: "florence", label: "Florence-2" },
    { id: "grounding", label: "Visual Grounding" },
  ];

  const drawLLaVA = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    const cy = h * 0.45;

    const blocks = [
      { label: "Image", x: 0.03, w: 0.09, color: "hsla(220, 80%, 55%, 0.7)", sub: "H×W×3" },
      { label: "ViT\nEncoder", x: 0.16, w: 0.1, color: "hsla(220, 80%, 55%, 0.7)", sub: "CLIP ViT-L" },
      { label: "MLP\nProjector", x: 0.3, w: 0.1, color: "hsla(45, 90%, 55%, 0.7)", sub: "Wₚ·z + bₚ" },
      { label: "Visual\nTokens", x: 0.44, w: 0.08, color: "hsla(170, 80%, 50%, 0.7)", sub: "576 tokens" },
      { label: "+", x: 0.54, w: 0.03, color: "rgba(255,255,255,0.3)", sub: "" },
      { label: "Text\nTokens", x: 0.59, w: 0.08, color: "hsla(290, 70%, 55%, 0.7)", sub: "prompt" },
      { label: "LLM\n(LLaMA)", x: 0.72, w: 0.12, color: "hsla(0, 80%, 55%, 0.7)", sub: "7B / 13B" },
      { label: "Response", x: 0.88, w: 0.1, color: "hsla(120, 70%, 50%, 0.7)", sub: "text out" },
    ];

    const boxH = 55;

    blocks.forEach((b, i) => {
      const bx = b.x * w;
      const bw = b.w * w;
      ctx.fillStyle = b.color.replace("0.7", "0.1").replace("0.3", "0.05");
      ctx.strokeStyle = b.color;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.roundRect(bx, cy - boxH / 2, bw, boxH, 5);
      ctx.fill(); ctx.stroke();

      ctx.fillStyle = b.color.replace("0.7", "0.9").replace("0.3", "0.5");
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      b.label.split("\n").forEach((line, li) => {
        ctx.fillText(line, bx + bw / 2, cy - 5 + li * 12);
      });

      if (b.sub) {
        ctx.fillStyle = "rgba(255,255,255,0.3)";
        ctx.font = "7px monospace";
        ctx.fillText(b.sub, bx + bw / 2, cy + boxH / 2 + 12);
      }

      // Arrow to next
      if (i < blocks.length - 1 && blocks[i + 1].label !== "+") {
        const nextX = blocks[i + 1].x * w;
        ctx.strokeStyle = "rgba(255,255,255,0.15)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(bx + bw, cy);
        ctx.lineTo(nextX, cy);
        ctx.stroke();
      }
    });

    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("h = LLM([zᵥ ; zₜ]) — visual + text tokens processed jointly", 10, h - 12);
  }, []);

  const drawFlorence = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    const cy = h * 0.35;

    const blocks = [
      { label: "Image", x: 0.05, color: "hsla(220, 80%, 55%, 0.7)" },
      { label: "DaViT\nEncoder", x: 0.2, color: "hsla(220, 80%, 55%, 0.7)" },
      { label: "Task\nPrompt", x: 0.38, color: "hsla(290, 70%, 55%, 0.7)" },
      { label: "Seq2Seq\nDecoder", x: 0.55, color: "hsla(45, 90%, 55%, 0.7)" },
      { label: "Output\nTokens", x: 0.75, color: "hsla(120, 70%, 50%, 0.7)" },
    ];

    const boxW = w * 0.12;
    const boxH = 48;

    blocks.forEach((b, i) => {
      const bx = b.x * w;
      ctx.fillStyle = b.color.replace("0.7", "0.1");
      ctx.strokeStyle = b.color;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.roundRect(bx, cy - boxH / 2, boxW, boxH, 5);
      ctx.fill(); ctx.stroke();

      ctx.fillStyle = b.color.replace("0.7", "0.9");
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      b.label.split("\n").forEach((line, li) => {
        ctx.fillText(line, bx + boxW / 2, cy - 3 + li * 12);
      });

      if (i < blocks.length - 1) {
        const nextX = blocks[i + 1].x * w;
        ctx.strokeStyle = "rgba(255,255,255,0.15)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(bx + boxW, cy);
        ctx.lineTo(nextX, cy);
        ctx.stroke();
      }
    });

    // Task examples below
    const tasks = [
      { prompt: "<OD>", output: "car [x1,y1,x2,y2]", color: "hsla(0, 80%, 55%, 0.6)" },
      { prompt: "<CAPTION>", output: "A red car on highway", color: "hsla(170, 80%, 50%, 0.6)" },
      { prompt: "<SEG>", output: "polygon coordinates", color: "hsla(280, 70%, 55%, 0.6)" },
      { prompt: "<OCR>", output: "\"STOP\" [x,y]", color: "hsla(45, 90%, 55%, 0.6)" },
    ];

    const taskY = h * 0.62;
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText("Unified task prompts → text outputs:", 10, taskY - 10);

    tasks.forEach((task, i) => {
      const tx = w * 0.05 + i * w * 0.23;
      ctx.fillStyle = task.color.replace("0.6", "0.08");
      ctx.strokeStyle = task.color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(tx, taskY, w * 0.2, 45, 4);
      ctx.fill(); ctx.stroke();

      ctx.fillStyle = task.color.replace("0.6", "0.9");
      ctx.font = "bold 9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(task.prompt, tx + w * 0.1, taskY + 15);
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.font = "8px monospace";
      ctx.fillText(task.output, tx + w * 0.1, taskY + 32);
    });

    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("y = Decoder(Encoder(I, task_prompt)) — one model, many tasks", 10, h - 12);
  }, []);

  const drawGrounding = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    // Scene with objects
    const sceneX = w * 0.05;
    const sceneY = h * 0.1;
    const sceneW = w * 0.4;
    const sceneH = h * 0.65;

    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    ctx.strokeRect(sceneX, sceneY, sceneW, sceneH);

    // Objects in scene
    const objects = [
      { emoji: "🚗", x: 0.2, y: 0.6, w: 0.25, h: 0.15, label: "red car" },
      { emoji: "🌳", x: 0.55, y: 0.3, w: 0.2, h: 0.35, label: "tree" },
      { emoji: "🚗", x: 0.6, y: 0.65, w: 0.2, h: 0.12, label: "blue car" },
      { emoji: "👤", x: 0.35, y: 0.45, w: 0.1, h: 0.2, label: "person" },
    ];

    // The query highlights one object
    const queryIdx = Math.floor((t * 0.4) % objects.length);

    objects.forEach((obj, i) => {
      const ox = sceneX + obj.x * sceneW;
      const oy = sceneY + obj.y * sceneH;
      const ow = obj.w * sceneW;
      const oh = obj.h * sceneH;

      const isTarget = i === queryIdx;
      if (isTarget) {
        ctx.strokeStyle = "hsla(120, 80%, 55%, 0.8)";
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
      } else {
        ctx.strokeStyle = "rgba(255,255,255,0.15)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
      }
      ctx.strokeRect(ox, oy, ow, oh);
      ctx.setLineDash([]);

      ctx.font = isTarget ? "20px serif" : "14px serif";
      ctx.textAlign = "center";
      ctx.fillStyle = isTarget ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.3)";
      ctx.fillText(obj.emoji, ox + ow / 2, oy + oh / 2 + 6);
    });

    // Query text on the right
    const queryX = w * 0.52;
    const queryY = h * 0.2;
    const queries = [
      "\"the red car on the left\"",
      "\"the tall tree\"",
      "\"the blue car behind the tree\"",
      "\"the person in the middle\"",
    ];

    ctx.fillStyle = "hsla(290, 70%, 55%, 0.1)";
    ctx.strokeStyle = "hsla(290, 70%, 55%, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(queryX, queryY, w * 0.42, 40, 6);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = "hsla(290, 70%, 60%, 0.9)";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Query: " + queries[queryIdx], queryX + w * 0.21, queryY + 15);
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "8px monospace";
    ctx.fillText("→ Grounding: localize described object", queryX + w * 0.21, queryY + 30);

    // Output box coordinates
    const outY = h * 0.5;
    const target = objects[queryIdx];
    ctx.fillStyle = "hsla(120, 70%, 50%, 0.1)";
    ctx.strokeStyle = "hsla(120, 70%, 50%, 0.5)";
    ctx.beginPath();
    ctx.roundRect(queryX, outY, w * 0.42, 55, 6);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = "hsla(120, 70%, 55%, 0.9)";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`Output: "${target.label}"`, queryX + w * 0.21, outY + 15);
    ctx.fillText(`bbox: [${(target.x * 100).toFixed(0)}, ${(target.y * 100).toFixed(0)}, ${((target.x + target.w) * 100).toFixed(0)}, ${((target.y + target.h) * 100).toFixed(0)}]`, queryX + w * 0.21, outY + 32);
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "8px monospace";
    ctx.fillText("confidence: 0.95", queryX + w * 0.21, outY + 46);

    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("b̂ = argmax P(b | I, query) — language-guided detection", 10, h - 12);
  }, []);

  const draw = mode === "contrastive" ? drawContrastive : mode === "embedding" ? drawEmbedding : drawGrounding;
  const canvasRef = useCanvas(draw, [mode]);

  return (
    <CanvasStage label="Vision-Language Models" modes={modes} activeMode={mode} onModeChange={setMode}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
    </CanvasStage>
  );
}

/* ═══════════════════════════════════════════
   3. VLM ARCHITECTURE PIPELINE
   ═══════════════════════════════════════════ */

export function VLMPipelineCanvas() {
  const [mode, setMode] = useState("llava");
  const modes = [
    { id: "llava", label: "LLaVA" },
    { id: "florence", label: "Florence-2" },
  ];

  const canvasRef = useCanvas(
    useCallback((ctx, w, h, t) => {
      ctx.fillStyle = "#0A0C15";
      ctx.fillRect(0, 0, w, h);

      if (mode === "llava") {
        // Already covered in CLIP canvas, show complementary: attention flow
        const cy = h * 0.4;
        const tokenCount = 20;
        const tokenW = (w * 0.8) / tokenCount;
        const startX = w * 0.1;

        // Visual tokens
        for (let i = 0; i < 12; i++) {
          const x = startX + i * tokenW;
          const alpha = 0.3 + Math.sin(t * 2 + i * 0.5) * 0.2;
          ctx.fillStyle = `hsla(220, 80%, 55%, ${alpha})`;
          ctx.fillRect(x, cy - 15, tokenW - 2, 30);
        }

        // Text tokens
        for (let i = 12; i < tokenCount; i++) {
          const x = startX + i * tokenW;
          const alpha = 0.3 + Math.sin(t * 2 + i * 0.5) * 0.2;
          ctx.fillStyle = `hsla(290, 70%, 55%, ${alpha})`;
          ctx.fillRect(x, cy - 15, tokenW - 2, 30);
        }

        // Attention lines from output to all tokens
        const outIdx = 18;
        const outX = startX + outIdx * tokenW + tokenW / 2;
        for (let i = 0; i < tokenCount; i++) {
          const srcX = startX + i * tokenW + tokenW / 2;
          const weight = Math.abs(Math.sin(t + i * 0.8)) * 0.4;
          ctx.strokeStyle = `hsla(45, 90%, 55%, ${weight})`;
          ctx.lineWidth = weight * 3;
          ctx.beginPath();
          ctx.moveTo(srcX, cy + 15);
          ctx.quadraticCurveTo((srcX + outX) / 2, cy + 60 + Math.abs(outX - srcX) * 0.15, outX, cy + 15);
          ctx.stroke();
        }

        ctx.fillStyle = "rgba(255,255,255,0.4)";
        ctx.font = "9px monospace";
        ctx.textAlign = "center";
        ctx.fillText("Visual tokens (12)", startX + 6 * tokenW, cy - 22);
        ctx.fillText("Text tokens (8)", startX + 16 * tokenW, cy - 22);

        ctx.fillStyle = "rgba(255,255,255,0.5)";
        ctx.font = "10px monospace";
        ctx.fillText("Cross-attention: output token attends to all visual + text tokens", w / 2, h - 20);
      } else {
        // Florence-2: DaViT multi-scale
        const scales = [
          { label: "1/4 scale", size: 80, y: h * 0.25 },
          { label: "1/8 scale", size: 55, y: h * 0.45 },
          { label: "1/16 scale", size: 35, y: h * 0.65 },
          { label: "1/32 scale", size: 20, y: h * 0.8 },
        ];

        const startX = w * 0.08;
        scales.forEach((s, i) => {
          ctx.fillStyle = `hsla(220, 80%, 55%, ${0.15 + i * 0.05})`;
          ctx.strokeStyle = `hsla(220, 80%, 55%, ${0.4 + i * 0.1})`;
          ctx.lineWidth = 1;
          ctx.fillRect(startX, s.y - s.size / 2, s.size, s.size);
          ctx.strokeRect(startX, s.y - s.size / 2, s.size, s.size);

          ctx.fillStyle = "rgba(255,255,255,0.5)";
          ctx.font = "8px monospace";
          ctx.textAlign = "left";
          ctx.fillText(s.label, startX + s.size + 8, s.y + 3);
        });

        // Decoder
        const decX = w * 0.55;
        ctx.fillStyle = "hsla(45, 90%, 55%, 0.1)";
        ctx.strokeStyle = "hsla(45, 90%, 55%, 0.5)";
        ctx.beginPath();
        ctx.roundRect(decX, h * 0.25, w * 0.15, h * 0.5, 6);
        ctx.fill(); ctx.stroke();

        ctx.fillStyle = "hsla(45, 90%, 60%, 0.9)";
        ctx.font = "10px monospace";
        ctx.textAlign = "center";
        ctx.fillText("Seq2Seq", decX + w * 0.075, h * 0.45);
        ctx.fillText("Decoder", decX + w * 0.075, h * 0.55);

        // Output tokens
        const outX = w * 0.78;
        const outTokens = ["car", "[120", "80", "340", "220]"];
        outTokens.forEach((tok, i) => {
          const y = h * 0.3 + i * 35;
          const alpha = clamp((Math.sin(t * 2 - i * 0.5) + 1) / 2, 0.3, 1);
          ctx.fillStyle = `hsla(120, 70%, 50%, ${alpha * 0.15})`;
          ctx.strokeStyle = `hsla(120, 70%, 50%, ${alpha * 0.5})`;
          ctx.beginPath();
          ctx.roundRect(outX, y, w * 0.12, 25, 4);
          ctx.fill(); ctx.stroke();

          ctx.fillStyle = `hsla(120, 70%, 55%, ${alpha})`;
          ctx.font = "9px monospace";
          ctx.fillText(tok, outX + w * 0.06, y + 15);
        });

        ctx.fillStyle = "rgba(255,255,255,0.35)";
        ctx.textAlign = "left";
        ctx.font = "10px monospace";
        ctx.fillText("DaViT multi-scale → autoregressive token generation", 10, h - 12);
      }

      ctx.textAlign = "left";
    }, [mode]),
    [mode]
  );

  return (
    <CanvasStage label="VLM Architecture" modes={modes} activeMode={mode} onModeChange={setMode}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
    </CanvasStage>
  );
}

/* ═══════════════════════════════════════════
   4. 3D SCENE UNDERSTANDING CANVAS
   ═══════════════════════════════════════════ */

export function Scene3DCanvas() {
  const [mode, setMode] = useState("scenegraph");
  const modes = [
    { id: "scenegraph", label: "3D Scene Graph" },
    { id: "3dqa", label: "3D Visual Q&A" },
    { id: "embodied", label: "Embodied Reasoning" },
  ];

  const drawSceneGraph = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    // Nodes
    const nodes = [
      { id: "room", x: 0.5, y: 0.15, label: "Living Room", color: "hsla(220, 80%, 55%, 0.8)", size: 22 },
      { id: "sofa", x: 0.2, y: 0.4, label: "Sofa", color: "hsla(0, 80%, 55%, 0.8)", size: 16 },
      { id: "table", x: 0.5, y: 0.45, label: "Table", color: "hsla(45, 90%, 55%, 0.8)", size: 16 },
      { id: "tv", x: 0.8, y: 0.35, label: "TV", color: "hsla(170, 80%, 50%, 0.8)", size: 14 },
      { id: "lamp", x: 0.35, y: 0.7, label: "Lamp", color: "hsla(280, 70%, 55%, 0.8)", size: 12 },
      { id: "book", x: 0.65, y: 0.7, label: "Book", color: "hsla(120, 70%, 50%, 0.8)", size: 12 },
      { id: "person", x: 0.3, y: 0.55, label: "Person", color: "hsla(30, 80%, 55%, 0.8)", size: 15 },
    ];

    // Edges with relations
    const edges = [
      { from: "room", to: "sofa", rel: "contains" },
      { from: "room", to: "table", rel: "contains" },
      { from: "room", to: "tv", rel: "contains" },
      { from: "sofa", to: "lamp", rel: "next to" },
      { from: "table", to: "book", rel: "on top" },
      { from: "person", to: "sofa", rel: "sitting on" },
      { from: "person", to: "tv", rel: "looking at" },
    ];

    const nodeMap: Record<string, typeof nodes[0]> = {};
    nodes.forEach(n => { nodeMap[n.id] = n; });

    // Draw edges
    edges.forEach((e, i) => {
      const from = nodeMap[e.from];
      const to = nodeMap[e.to];
      const fx = from.x * w, fy = from.y * h;
      const tx = to.x * w, ty = to.y * h;

      const alpha = 0.15 + Math.sin(t + i) * 0.1;
      ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(fx, fy);
      ctx.lineTo(tx, ty);
      ctx.stroke();

      // Relation label
      const mx = (fx + tx) / 2;
      const my = (fy + ty) / 2;
      ctx.fillStyle = `rgba(255,255,255,${alpha + 0.15})`;
      ctx.font = "7px monospace";
      ctx.textAlign = "center";
      ctx.fillText(e.rel, mx, my - 4);
    });

    // Draw nodes
    nodes.forEach(n => {
      const nx = n.x * w;
      const ny = n.y * h;
      const pulse = 1 + Math.sin(t * 2 + n.x * 5) * 0.08;

      // Glow
      const grad = ctx.createRadialGradient(nx, ny, 0, nx, ny, n.size * 2 * pulse);
      grad.addColorStop(0, n.color.replace("0.8", "0.15"));
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(nx, ny, n.size * 2 * pulse, 0, Math.PI * 2);
      ctx.fill();

      // Node circle
      ctx.fillStyle = n.color.replace("0.8", "0.2");
      ctx.strokeStyle = n.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(nx, ny, n.size * pulse, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Label
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(n.label, nx, ny + n.size + 14);
    });

    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("3D Scene Graph: objects + spatial relationships + attributes", 10, h - 12);
  }, []);

  const draw3DQA = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    // 3D scene representation (left)
    const sceneX = w * 0.03;
    const sceneW = w * 0.38;
    const sceneH = h * 0.55;
    const sceneY = h * 0.1;

    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1;
    ctx.strokeRect(sceneX, sceneY, sceneW, sceneH);

    // Simple 3D room wireframe
    const rx = sceneX + 15;
    const ry = sceneY + 15;
    const rw = sceneW - 30;
    const rh = sceneH - 30;
    const d = 25;

    ctx.strokeStyle = "rgba(255,255,255,0.1)";
    // Front face
    ctx.strokeRect(rx, ry, rw, rh);
    // Back face
    ctx.strokeRect(rx + d, ry - d, rw, rh);
    // Connect
    ctx.beginPath();
    ctx.moveTo(rx, ry); ctx.lineTo(rx + d, ry - d);
    ctx.moveTo(rx + rw, ry); ctx.lineTo(rx + rw + d, ry - d);
    ctx.moveTo(rx, ry + rh); ctx.lineTo(rx + d, ry + rh - d);
    ctx.moveTo(rx + rw, ry + rh); ctx.lineTo(rx + rw + d, ry + rh - d);
    ctx.stroke();

    // Objects inside
    ctx.font = "18px serif";
    ctx.textAlign = "center";
    ctx.fillText("🪑", rx + rw * 0.3, ry + rh * 0.6);
    ctx.fillText("📺", rx + rw * 0.7, ry + rh * 0.3);
    ctx.fillText("🛋️", rx + rw * 0.2, ry + rh * 0.4);

    // Q&A pairs on the right
    const qaX = w * 0.48;
    const qaPairs = [
      { q: "How many chairs are in the room?", a: "There is 1 chair, near the sofa." },
      { q: "What is the TV mounted on?", a: "The TV is on the wall, facing the sofa." },
      { q: "Is there a path from door to sofa?", a: "Yes, 2.3m path, no obstacles." },
    ];

    const qaIdx = Math.floor((t * 0.3) % qaPairs.length);
    const qa = qaPairs[qaIdx];

    // Question
    ctx.fillStyle = "hsla(290, 70%, 55%, 0.1)";
    ctx.strokeStyle = "hsla(290, 70%, 55%, 0.5)";
    ctx.beginPath();
    ctx.roundRect(qaX, h * 0.12, w * 0.48, 40, 6);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = "hsla(290, 70%, 60%, 0.9)";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText("Q: " + qa.q, qaX + 8, h * 0.12 + 25);

    // Answer
    ctx.fillStyle = "hsla(120, 70%, 50%, 0.1)";
    ctx.strokeStyle = "hsla(120, 70%, 50%, 0.5)";
    ctx.beginPath();
    ctx.roundRect(qaX, h * 0.35, w * 0.48, 40, 6);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = "hsla(120, 70%, 55%, 0.9)";
    ctx.fillText("A: " + qa.a, qaX + 8, h * 0.35 + 25);

    // Pipeline note
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "8px monospace";
    ctx.fillText("3D point cloud + language → spatial reasoning", qaX, h * 0.55);

    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("ScanQA / 3D-LLM: answer questions about 3D scenes", 10, h - 12);
  }, []);

  const drawEmbodied = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => {
    ctx.fillStyle = "#0A0C15";
    ctx.fillRect(0, 0, w, h);

    // Robot / agent in center
    const agentX = w * 0.45;
    const agentY = h * 0.45;

    // Environment grid
    const gridSize = 8;
    const cellW = w * 0.06;
    const cellH = h * 0.06;
    const gridX = w * 0.15;
    const gridY = h * 0.15;

    for (let r = 0; r < gridSize; r++) {
      for (let c = 0; c < gridSize; c++) {
        ctx.strokeStyle = "rgba(255,255,255,0.05)";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(gridX + c * cellW, gridY + r * cellH, cellW, cellH);
      }
    }

    // Agent position (animated)
    const path = [
      { r: 4, c: 3 }, { r: 3, c: 3 }, { r: 3, c: 4 }, { r: 3, c: 5 },
      { r: 2, c: 5 }, { r: 2, c: 6 },
    ];
    const pathIdx = Math.floor((t * 0.5) % path.length);
    const pos = path[pathIdx];

    // Draw path
    for (let i = 0; i <= pathIdx; i++) {
      const p = path[i];
      const px = gridX + p.c * cellW + cellW / 2;
      const py = gridY + p.r * cellH + cellH / 2;

      if (i < pathIdx) {
        ctx.fillStyle = `hsla(220, 80%, 55%, ${0.1 + (i / path.length) * 0.2})`;
        ctx.fillRect(gridX + p.c * cellW + 2, gridY + p.r * cellH + 2, cellW - 4, cellH - 4);
      }

      if (i > 0) {
        const prev = path[i - 1];
        ctx.strokeStyle = "hsla(220, 80%, 55%, 0.3)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(gridX + prev.c * cellW + cellW / 2, gridY + prev.r * cellH + cellH / 2);
        ctx.lineTo(px, py);
        ctx.stroke();
      }
    }

    // Agent
    const ax = gridX + pos.c * cellW + cellW / 2;
    const ay = gridY + pos.r * cellH + cellH / 2;
    ctx.fillStyle = "hsla(45, 90%, 55%, 0.9)";
    ctx.beginPath();
    ctx.arc(ax, ay, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "14px serif";
    ctx.textAlign = "center";
    ctx.fillText("🤖", ax, ay + 5);

    // Goal
    ctx.fillStyle = "hsla(120, 70%, 50%, 0.3)";
    ctx.fillRect(gridX + 6 * cellW + 2, gridY + 2 * cellH + 2, cellW - 4, cellH - 4);
    ctx.font = "14px serif";
    ctx.fillText("🎯", gridX + 6 * cellW + cellW / 2, gridY + 2 * cellH + cellH / 2 + 5);

    // Obstacles
    const obstacles = [[1, 4], [4, 5], [5, 3], [2, 2]];
    obstacles.forEach(([r, c]) => {
      ctx.fillStyle = "hsla(0, 80%, 55%, 0.15)";
      ctx.fillRect(gridX + c * cellW + 1, gridY + r * cellH + 1, cellW - 2, cellH - 2);
    });

    // Instruction
    const instrX = w * 0.6;
    ctx.fillStyle = "hsla(290, 70%, 55%, 0.1)";
    ctx.strokeStyle = "hsla(290, 70%, 55%, 0.5)";
    ctx.beginPath();
    ctx.roundRect(instrX, h * 0.15, w * 0.35, 50, 6);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = "hsla(290, 70%, 60%, 0.9)";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText("\"Go to the kitchen and", instrX + 8, h * 0.15 + 20);
    ctx.fillText("pick up the red mug\"", instrX + 8, h * 0.15 + 35);

    // Pipeline
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "8px monospace";
    ctx.fillText("Observe → Plan → Act → Observe", instrX, h * 0.55);

    const steps = ["Perception", "Planning", "Action"];
    steps.forEach((s, i) => {
      const sx = instrX + i * w * 0.12;
      const sy = h * 0.62;
      ctx.fillStyle = `hsla(${120 + i * 60}, 70%, 50%, 0.1)`;
      ctx.strokeStyle = `hsla(${120 + i * 60}, 70%, 50%, 0.4)`;
      ctx.beginPath();
      ctx.roundRect(sx, sy, w * 0.1, 30, 4);
      ctx.fill(); ctx.stroke();

      ctx.fillStyle = `hsla(${120 + i * 60}, 70%, 55%, 0.8)`;
      ctx.font = "8px monospace";
      ctx.textAlign = "center";
      ctx.fillText(s, sx + w * 0.05, sy + 18);
    });

    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px monospace";
    ctx.fillText("Embodied AI: VLMs plan actions from visual observations + language goals", 10, h - 12);
  }, []);

  const draw = mode === "scenegraph" ? drawSceneGraph : mode === "3dqa" ? draw3DQA : drawEmbodied;
  const canvasRef = useCanvas(draw, [mode]);

  return (
    <CanvasStage label="3D Scene Understanding" modes={modes} activeMode={mode} onModeChange={setMode}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
    </CanvasStage>
  );
}
