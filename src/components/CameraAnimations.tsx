import { useRef, useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Camera, Move3D, Focus, Grid3X3, Aperture, Cpu } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

// ─── Utility ─────────────────────────────────────────────────────────
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

// ─── Shared Canvas Drawing Helpers ───────────────────────────────────
function drawArrow(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, color: string, lw = 2) {
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = lw;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  const ang = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 10 * Math.cos(ang - 0.4), y2 - 10 * Math.sin(ang - 0.4));
  ctx.lineTo(x2 - 10 * Math.cos(ang + 0.4), y2 - 10 * Math.sin(ang + 0.4));
  ctx.closePath(); ctx.fill();
}

function drawRay(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, color: string, lw = 1.5) {
  ctx.strokeStyle = color; ctx.lineWidth = lw;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
}

function labelText(ctx: CanvasRenderingContext2D, x: number, y: number, text: string, color: string, align: CanvasTextAlign = "left") {
  ctx.save();
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillStyle = color; ctx.textAlign = align;
  const lines = text.split("\n");
  lines.forEach((l, i) => ctx.fillText(l, x, y + i * 14));
  ctx.restore();
}

// ─── Scene definitions ───────────────────────────────────────────────
const scenes = [
  { id: "projection3d", label: "3D → 2D Projection", icon: Move3D, description: "Watch how a 3D world point travels through space and lands on the camera's image plane." },
  { id: "pinhole", label: "Pinhole Model", icon: Camera, description: "Light rays pass through a single point — the center of projection." },
  { id: "perspective", label: "Perspective Projection", icon: Grid3X3, description: "3D world point → 2D image point via similar triangles." },
  { id: "intrinsic", label: "Intrinsic Matrix K", icon: Cpu, description: "Mapping from camera coordinates to pixel coordinates." },
  { id: "lens", label: "Lens & Depth of Field", icon: Aperture, description: "Real lenses blur out-of-focus points into circles of confusion." },
  { id: "sensor", label: "Sensor & Pixels", icon: Focus, description: "Continuous irradiance → discrete pixel array." },
];

// ═══════════════════════════════════════════════════════════════════════
// SCENE: 3D → 2D Projection (NEW hero animation)
// ═══════════════════════════════════════════════════════════════════════
function Projection3DScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const tRef = useRef(0);
  const [focalLength, setFocalLength] = useState(180);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const cx = W * 0.55, cy = H * 0.5;
    const f = focalLength;
    const t = tRef.current;

    // Camera (pinhole) at center
    const camX = cx, camY = cy;

    // Image plane to the right
    const imgPlaneX = camX + f * 0.7;

    // 3D world points orbiting
    const points3D = [
      { X: -120, Y: -80, Z: 300, color: "hsl(0, 85%, 60%)", label: "P₁" },
      { X: 60, Y: -40, Z: 250 + 50 * Math.sin(t * 0.7), color: "hsl(140, 70%, 55%)", label: "P₂" },
      { X: -30 + 40 * Math.cos(t * 0.5), Y: 60, Z: 350, color: "hsl(220, 80%, 60%)", label: "P₃" },
      { X: 80 * Math.sin(t * 0.3), Y: -60 + 30 * Math.sin(t * 0.6), Z: 200, color: "hsl(45, 90%, 55%)", label: "P₄" },
    ];

    // Draw optical axis
    ctx.strokeStyle = "hsl(var(--primary) / 0.15)";
    ctx.lineWidth = 1;
    ctx.setLineDash([8, 6]);
    ctx.beginPath(); ctx.moveTo(30, cy); ctx.lineTo(W - 30, cy); ctx.stroke();
    ctx.setLineDash([]);

    // Draw camera body
    ctx.fillStyle = "hsl(var(--primary) / 0.08)";
    ctx.strokeStyle = "hsl(var(--primary) / 0.5)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(camX - 20, camY - 16, 40, 32, 4);
    ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.arc(camX, camY, 9, 0, Math.PI * 2);
    ctx.fillStyle = "hsl(var(--primary) / 0.2)"; ctx.fill(); ctx.stroke();
    labelText(ctx, camX, camY + 30, "Camera O", "hsl(var(--primary))", "center");

    // Draw image plane
    ctx.strokeStyle = "hsl(270, 70%, 60%)";
    ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(imgPlaneX, cy - H * 0.38); ctx.lineTo(imgPlaneX, cy + H * 0.38); ctx.stroke();
    labelText(ctx, imgPlaneX + 6, cy - H * 0.36, "Image\nPlane", "hsl(270, 70%, 60%)");

    // Animate each point: draw in world, trace ray, project
    for (const pt of points3D) {
      const worldX = camX - pt.Z * 0.6;
      const worldY = camY - pt.Y * 0.5;

      // World point
      ctx.fillStyle = pt.color;
      ctx.beginPath(); ctx.arc(worldX, worldY, 7, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = pt.color; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(worldX, worldY, 11, 0, Math.PI * 2); ctx.stroke();
      labelText(ctx, worldX - 14, worldY - 16, pt.label, pt.color, "center");

      // Projected image point
      const imgY = camY + (f * 0.7 * pt.Y) / pt.Z;

      // Animated ray: particle traveling along ray
      const rayProgress = (Math.sin(t * 1.2 + pt.Z * 0.01) * 0.5 + 0.5);

      // Full ray (dim)
      ctx.globalAlpha = 0.2;
      drawRay(ctx, worldX, worldY, camX, camY, pt.color, 1);
      drawRay(ctx, camX, camY, imgPlaneX, imgY, pt.color, 1);
      ctx.globalAlpha = 1;

      // Animated particle on the ray
      const particleSegment = rayProgress < 0.5 ? 0 : 1;
      let px: number, py: number;
      if (particleSegment === 0) {
        const p = rayProgress * 2;
        px = lerp(worldX, camX, p);
        py = lerp(worldY, camY, p);
      } else {
        const p = (rayProgress - 0.5) * 2;
        px = lerp(camX, imgPlaneX, p);
        py = lerp(camY, imgY, p);
      }

      // Glowing particle
      const gradient = ctx.createRadialGradient(px, py, 0, px, py, 12);
      gradient.addColorStop(0, pt.color);
      gradient.addColorStop(1, "transparent");
      ctx.fillStyle = gradient;
      ctx.beginPath(); ctx.arc(px, py, 12, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = pt.color;
      ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();

      // Image point
      ctx.fillStyle = pt.color;
      ctx.globalAlpha = 0.8;
      ctx.beginPath(); ctx.arc(imgPlaneX, imgY, 4, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Formula overlay
    ctx.fillStyle = "hsl(var(--background) / 0.85)";
    ctx.fillRect(W - 220, 10, 210, 50);
    ctx.strokeStyle = "hsl(var(--border))";
    ctx.lineWidth = 1;
    ctx.strokeRect(W - 220, 10, 210, 50);
    ctx.font = "12px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsl(var(--primary))";
    ctx.textAlign = "left";
    ctx.fillText("x' = f · X / Z", W - 210, 32);
    ctx.fillText("y' = f · Y / Z", W - 210, 50);

    // f annotation
    ctx.strokeStyle = "hsl(var(--muted-foreground) / 0.4)";
    ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(camX, cy + H * 0.35); ctx.lineTo(imgPlaneX, cy + H * 0.35); ctx.stroke();
    ctx.setLineDash([]);
    labelText(ctx, (camX + imgPlaneX) / 2, cy + H * 0.35 + 14, `f = ${focalLength}`, "hsl(var(--muted-foreground))", "center");
  }, [focalLength]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resizeCanvas = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) { canvas.width = rect.width; canvas.height = 420; }
    };
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const loop = () => {
      tRef.current += 0.02;
      draw();
      rafRef.current = requestAnimationFrame(loop);
    };
    loop();
    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resizeCanvas);
    };
  }, [draw]);

  return (
    <div className="space-y-3">
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border bg-background" style={{ height: 420 }} />
      <div className="flex items-center gap-4 px-1">
        <span className="text-xs text-muted-foreground font-mono whitespace-nowrap">Focal Length (f)</span>
        <Slider min={80} max={320} step={1} value={[focalLength]} onValueChange={([v]) => setFocalLength(v)} className="flex-1" />
        <span className="text-xs font-mono text-primary w-12 text-right">{focalLength}px</span>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// SCENE: Pinhole Camera
// ═══════════════════════════════════════════════════════════════════════
function PinholeScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const tRef = useRef(0);
  const [focalLength, setFocalLength] = useState(200);
  const [objDist, setObjDist] = useState(300);
  const [objHeight, setObjHeight] = useState(120);
  const [showAllRays, setShowAllRays] = useState(false);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const f = focalLength;
    const cxp = W / 2, cy = H / 2;
    const objX = cxp - objDist;
    const imgX = cxp + f;
    const imgH = -(f * objHeight) / objDist;
    const objAnim = Math.sin(tRef.current) * 20;

    // Optical axis
    ctx.strokeStyle = "hsl(var(--primary) / 0.15)";
    ctx.lineWidth = 1; ctx.setLineDash([8, 6]);
    ctx.beginPath(); ctx.moveTo(30, cy); ctx.lineTo(W - 30, cy); ctx.stroke();
    ctx.setLineDash([]);

    // Object arrow
    drawArrow(ctx, objX, cy, objX, cy - objHeight + objAnim, "hsl(20, 90%, 55%)", 3);

    // Image plane
    ctx.strokeStyle = "hsl(270, 70%, 60%)";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(imgX, cy - H * 0.4); ctx.lineTo(imgX, cy + H * 0.4); ctx.stroke();

    // Pinhole wall
    ctx.strokeStyle = "hsl(var(--primary) / 0.5)"; ctx.lineWidth = 3;
    const gap = 12;
    ctx.beginPath(); ctx.moveTo(cxp, cy - H * 0.4); ctx.lineTo(cxp, cy - gap); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cxp, cy + gap); ctx.lineTo(cxp, cy + H * 0.4); ctx.stroke();
    ctx.fillStyle = "hsl(var(--primary) / 0.15)";
    ctx.fillRect(cxp - 3, cy - H * 0.4, 6, H * 0.8);
    ctx.fillStyle = "hsl(var(--primary))";
    ctx.beginPath(); ctx.arc(cxp, cy, 4, 0, Math.PI * 2); ctx.fill();

    // Rays
    const nRays = showAllRays ? 7 : 2;
    for (let i = 0; i < nRays; i++) {
      const t = nRays === 1 ? 0.5 : i / (nRays - 1);
      const srcY = cy - (objHeight - objAnim) + (objHeight - objAnim) * 2 * t;
      const alpha = showAllRays ? 0.25 : 0.7;
      drawRay(ctx, objX, srcY, cxp, cy, `hsl(var(--primary) / ${alpha})`, 1.2);
      const slope = (cy - srcY) / (cxp - objX);
      const dstY = cy + slope * (imgX - cxp);
      drawRay(ctx, cxp, cy, imgX, dstY, `hsl(var(--primary) / ${alpha})`, 1.2);
      if (!showAllRays || i === 0 || i === nRays - 1) {
        ctx.fillStyle = "hsl(270, 70%, 60%)";
        ctx.beginPath(); ctx.arc(imgX, dstY, 3, 0, Math.PI * 2); ctx.fill();
      }
    }

    // Inverted image
    drawArrow(ctx, imgX, cy, imgX, cy + Math.abs(imgH) + objAnim * (f / objDist), "hsl(270, 70%, 60%)", 2);

    // Labels
    labelText(ctx, objX, cy + 20, "Object", "hsl(20, 90%, 55%)");
    labelText(ctx, cxp, cy - H * 0.38, "Pinhole", "hsl(var(--primary))");
    labelText(ctx, imgX + 8, cy - H * 0.38, "Image\nPlane", "hsl(270, 70%, 60%)");

    // Distance annotations
    ctx.strokeStyle = "hsl(var(--muted-foreground) / 0.15)";
    ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(objX, cy + H * 0.35); ctx.lineTo(cxp, cy + H * 0.35); ctx.stroke();
    ctx.setLineDash([]);
    labelText(ctx, (objX + cxp) / 2, cy + H * 0.35 + 15, `d = ${objDist}`, "hsl(var(--muted-foreground))", "center");
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(cxp, cy + H * 0.35); ctx.lineTo(imgX, cy + H * 0.35); ctx.stroke();
    ctx.setLineDash([]);
    labelText(ctx, (cxp + imgX) / 2, cy + H * 0.35 + 15, `f = ${f}`, "hsl(var(--muted-foreground))", "center");

    const m = (f / objDist).toFixed(2);
    labelText(ctx, W - 20, 20, `|m| = ${m}`, "hsl(var(--primary))", "right");
  }, [focalLength, objDist, objHeight, showAllRays]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) { canvas.width = rect.width; canvas.height = 380; }
    };
    resize();
    window.addEventListener("resize", resize);
    const loop = () => { tRef.current += 0.015; draw(); rafRef.current = requestAnimationFrame(loop); };
    loop();
    return () => { cancelAnimationFrame(rafRef.current); window.removeEventListener("resize", resize); };
  }, [draw]);

  return (
    <div className="space-y-3">
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border bg-background" style={{ height: 380 }} />
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 px-1">
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Focal Length</span><span className="text-primary">{focalLength}px</span></div>
          <Slider min={80} max={320} step={1} value={[focalLength]} onValueChange={([v]) => setFocalLength(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Object Dist</span><span className="text-primary">{objDist}px</span></div>
          <Slider min={150} max={500} step={1} value={[objDist]} onValueChange={([v]) => setObjDist(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Object Height</span><span className="text-primary">{objHeight}px</span></div>
          <Slider min={40} max={200} step={1} value={[objHeight]} onValueChange={([v]) => setObjHeight(v)} />
        </div>
      </div>
      <Button variant={showAllRays ? "default" : "outline"} size="sm" className="text-xs" onClick={() => setShowAllRays(!showAllRays)}>
        {showAllRays ? "Hide Extra Rays" : "Show All Rays"}
      </Button>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// SCENE: Intrinsic Matrix
// ═══════════════════════════════════════════════════════════════════════
function IntrinsicScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [fx, setFx] = useState(500);
  const [fy, setFy] = useState(500);
  const [cxOff, setCxOff] = useState(0);
  const [cyOff, setCyOff] = useState(0);
  const [skew, setSkew] = useState(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2;
    const pw = W * 0.7, ph = H * 0.7;
    const px = cx - pw / 2 + cxOff * 0.3, py = cy - ph / 2 + cyOff * 0.3;

    // Image plane
    ctx.fillStyle = "hsl(var(--primary) / 0.03)";
    ctx.strokeStyle = "hsl(var(--primary) / 0.2)";
    ctx.lineWidth = 1;
    ctx.fillRect(px, py, pw, ph);
    ctx.strokeRect(px, py, pw, ph);

    // Pixel grid (skewed)
    const gridN = 12;
    ctx.strokeStyle = "hsl(var(--primary) / 0.07)"; ctx.lineWidth = 1;
    const skewFactor = skew / 500;
    for (let i = 0; i <= gridN; i++) {
      const t = i / gridN;
      const x = px + pw * t;
      const skewDy = skewFactor * pw * t;
      ctx.beginPath(); ctx.moveTo(x + skewDy, py); ctx.lineTo(x + skewDy + skewFactor * ph, py + ph); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(px, py + ph * t); ctx.lineTo(px + pw, py + ph * t + skewFactor * pw); ctx.stroke();
    }

    // Principal point
    const ppx = cx + cxOff * 0.3, ppy = cy + cyOff * 0.3;
    ctx.strokeStyle = "hsl(20, 90%, 55% / 0.7)"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(ppx, py); ctx.lineTo(ppx, py + ph); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px, ppy); ctx.lineTo(px + pw, ppy); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "hsl(20, 90%, 55%)";
    ctx.beginPath(); ctx.arc(ppx, ppy, 7, 0, Math.PI * 2); ctx.fill();
    labelText(ctx, ppx + 10, ppy - 10, `(cx, cy)\n= (${cxOff}, ${cyOff})`, "hsl(20, 90%, 55%)");

    // Image center
    ctx.strokeStyle = "hsl(var(--primary) / 0.4)"; ctx.lineWidth = 1; ctx.setLineDash([2, 4]);
    ctx.beginPath(); ctx.moveTo(cx, py); ctx.lineTo(cx, py + ph); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px, cy); ctx.lineTo(px + pw, cy); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "hsl(var(--primary) / 0.4)";
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
    labelText(ctx, cx + 10, cy - 10, "Image\ncenter", "hsl(var(--primary) / 0.5)");

    // Focal length arrows
    const scaleF = 0.18;
    drawArrow(ctx, cx, cy + ph * 0.45, cx + fx * scaleF, cy + ph * 0.45, "hsl(270, 70%, 60%)", 2);
    labelText(ctx, cx + fx * scaleF * 0.5, cy + ph * 0.45 + 16, `fx = ${fx}px`, "hsl(270, 70%, 60%)", "center");
    drawArrow(ctx, cx + pw * 0.45, cy, cx + pw * 0.45, cy + fy * scaleF, "hsl(270, 70%, 60%)", 2);
    labelText(ctx, cx + pw * 0.45 + 10, cy + fy * scaleF * 0.5, `fy = ${fy}px`, "hsl(270, 70%, 60%)");

    // K matrix display
    ctx.fillStyle = "hsl(var(--background) / 0.85)";
    ctx.fillRect(W - 270, 10, 260, 64);
    ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1;
    ctx.strokeRect(W - 270, 10, 260, 64);
    ctx.font = "12px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsl(var(--primary))"; ctx.textAlign = "left";
    ctx.fillText(`K = [${String(fx).padStart(4)}  ${String(skew).padStart(4)}  ${String(cxOff).padStart(5)} ]`, W - 260, 30);
    ctx.fillText(`    [   0  ${String(fy).padStart(4)}  ${String(cyOff).padStart(5)} ]`, W - 260, 48);
    ctx.fillText(`    [   0     0      1 ]`, W - 260, 66);
  }, [fx, fy, cxOff, cyOff, skew]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) { canvas.width = rect.width; canvas.height = 380; }
      draw();
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [draw]);

  useEffect(() => { draw(); }, [draw]);

  return (
    <div className="space-y-3">
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border bg-background" style={{ height: 380 }} />
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 px-1">
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>fx</span><span className="text-primary">{fx}</span></div>
          <Slider min={200} max={900} step={1} value={[fx]} onValueChange={([v]) => setFx(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>fy</span><span className="text-primary">{fy}</span></div>
          <Slider min={200} max={900} step={1} value={[fy]} onValueChange={([v]) => setFy(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>cx</span><span className="text-primary">{cxOff}</span></div>
          <Slider min={-150} max={150} step={1} value={[cxOff]} onValueChange={([v]) => setCxOff(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>cy</span><span className="text-primary">{cyOff}</span></div>
          <Slider min={-150} max={150} step={1} value={[cyOff]} onValueChange={([v]) => setCyOff(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Skew γ</span><span className="text-primary">{skew}</span></div>
          <Slider min={-100} max={100} step={1} value={[skew]} onValueChange={([v]) => setSkew(v)} />
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// SCENE: Lens & Depth of Field
// ═══════════════════════════════════════════════════════════════════════
function LensScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [fmm, setFmm] = useState(50);
  const [aperture, setAperture] = useState(3);
  const [focusDist, setFocusDist] = useState(200);
  const [objDist2, setObjDist2] = useState(350);

  const thinLensImageDist = (f: number, d: number) => d <= f ? 9999 : (f * d) / (d - f);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2;
    const scale = 0.9;
    const lensX = cx;
    const aperR = (fmm / aperture) * 1.5;
    const halfAper = Math.min(aperR, H * 0.35);

    // Optical axis
    ctx.strokeStyle = "hsl(var(--primary) / 0.1)"; ctx.lineWidth = 1; ctx.setLineDash([8, 6]);
    ctx.beginPath(); ctx.moveTo(20, cy); ctx.lineTo(W - 20, cy); ctx.stroke();
    ctx.setLineDash([]);

    // Lens
    ctx.save();
    ctx.strokeStyle = "hsl(var(--primary) / 0.6)"; ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(lensX, cy - halfAper);
    ctx.bezierCurveTo(lensX + 30, cy - halfAper / 2, lensX + 30, cy + halfAper / 2, lensX, cy + halfAper);
    ctx.bezierCurveTo(lensX - 30, cy + halfAper / 2, lensX - 30, cy - halfAper / 2, lensX, cy - halfAper);
    ctx.fillStyle = "hsl(var(--primary) / 0.05)"; ctx.fill(); ctx.stroke();
    ctx.restore();
    labelText(ctx, lensX + 35, cy - halfAper + 10, "Lens", "hsl(var(--primary))");

    // Sensor
    const di_focus = thinLensImageDist(fmm * scale, focusDist);
    const sensorX = lensX + di_focus;
    ctx.strokeStyle = "hsl(270, 70%, 60%)"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(sensorX, cy - H * 0.35); ctx.lineTo(sensorX, cy + H * 0.35); ctx.stroke();
    labelText(ctx, sensorX + 6, cy - H * 0.33, "Sensor", "hsl(270, 70%, 60%)");

    // In-focus object
    const objFocusX = lensX - focusDist;
    if (objFocusX > 20) {
      drawArrow(ctx, objFocusX, cy, objFocusX, cy - 50, "hsl(var(--primary))", 2.5);
      labelText(ctx, objFocusX, cy + 15, "In focus", "hsl(var(--primary))");
      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = "hsl(var(--primary))"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(objFocusX, cy - 50); ctx.lineTo(lensX, cy - halfAper);
      ctx.lineTo(sensorX, cy + 50); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(objFocusX, cy - 50); ctx.lineTo(lensX, cy + halfAper);
      ctx.lineTo(sensorX, cy + 50); ctx.stroke();
      ctx.globalAlpha = 0.8;
      ctx.strokeStyle = "hsl(var(--primary))"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(objFocusX, cy - 50); ctx.lineTo(sensorX, cy + 50); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = "hsl(var(--primary))";
      ctx.beginPath(); ctx.arc(sensorX, cy + 50, 3, 0, Math.PI * 2); ctx.fill();
    }

    // Out-of-focus object
    const objBlurX = lensX - objDist2;
    if (objBlurX > 20 && objBlurX < lensX - 20) {
      drawArrow(ctx, objBlurX, cy, objBlurX, cy - 60, "hsl(20, 90%, 55%)", 2.5);
      labelText(ctx, objBlurX, cy + 15, "Out of\nfocus", "hsl(20, 90%, 55%)");
      const di_blur = thinLensImageDist(fmm * scale, objDist2);
      const sharpPtX = lensX + di_blur;
      const CoC = halfAper * Math.abs(sensorX - sharpPtX) / Math.max(sharpPtX, 1);
      ctx.globalAlpha = 0.4;
      ctx.strokeStyle = "hsl(20, 90%, 55%)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(objBlurX, cy - 60); ctx.lineTo(lensX, cy - halfAper);
      ctx.lineTo(sensorX, cy + 60 * (di_focus / di_blur)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(objBlurX, cy - 60); ctx.lineTo(lensX, cy + halfAper);
      ctx.lineTo(sensorX, cy + 60 * (di_focus / di_blur)); ctx.stroke();
      ctx.globalAlpha = 1;
      const cocR = clamp(CoC * 0.5, 1, 25);
      const imgYblur = cy + 60 * (di_focus / di_blur);
      ctx.strokeStyle = "hsl(20, 90%, 55%)"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sensorX, imgYblur, cocR, 0, Math.PI * 2); ctx.stroke();
      labelText(ctx, sensorX + cocR + 4, imgYblur, `CoC ≈ ${cocR.toFixed(1)}px`, "hsl(20, 90%, 55%)");
    }

    labelText(ctx, W - 10, 15, `f/${aperture.toFixed(1)}  f=${fmm}mm`, "hsl(var(--muted-foreground))", "right");
  }, [fmm, aperture, focusDist, objDist2]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) { canvas.width = rect.width; canvas.height = 380; }
      draw();
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [draw]);

  useEffect(() => { draw(); }, [draw]);

  return (
    <div className="space-y-3">
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border bg-background" style={{ height: 380 }} />
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 px-1">
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Focal (mm)</span><span className="text-primary">{fmm}</span></div>
          <Slider min={20} max={150} step={1} value={[fmm]} onValueChange={([v]) => setFmm(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Aperture</span><span className="text-primary">f/{aperture.toFixed(1)}</span></div>
          <Slider min={1} max={16} step={0.5} value={[aperture]} onValueChange={([v]) => setAperture(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Focus Dist</span><span className="text-primary">{focusDist}</span></div>
          <Slider min={80} max={450} step={1} value={[focusDist]} onValueChange={([v]) => setFocusDist(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Obj 2 Dist</span><span className="text-primary">{objDist2}</span></div>
          <Slider min={80} max={450} step={1} value={[objDist2]} onValueChange={([v]) => setObjDist2(v)} />
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// SCENE: Sensor & Pixels
// ═══════════════════════════════════════════════════════════════════════
function SensorScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const tRef = useRef(0);
  const [res, setRes] = useState(16);
  const [bits, setBits] = useState(8);
  const [noise, setNoise] = useState(5);
  const [bayerOn, setBayerOn] = useState(true);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const levels = Math.pow(2, bits);
    const noiseF = noise / 100;
    const gridW = Math.min(W * 0.55, H * 0.85);
    const cellSize = gridW / res;
    const startX = W * 0.05;
    const startY = (H - gridW) / 2;
    const t = tRef.current;

    for (let row = 0; row < res; row++) {
      for (let col = 0; col < res; col++) {
        const nx = col / res, ny = row / res;
        const v = 0.5 + 0.5 * Math.sin((nx + t) * Math.PI * 3) * Math.cos((ny + t * 0.7) * Math.PI * 2);
        const q = Math.floor(v * (levels - 1)) / (levels - 1);
        const n = (Math.random() - 0.5) * noiseF;
        const final = clamp(q + n, 0, 1);

        let r: number, g: number, b: number;
        if (bayerOn) {
          const isEvenRow = row % 2 === 0, isEvenCol = col % 2 === 0;
          if (isEvenRow && isEvenCol) { r = 0; g = final; b = 0; }
          else if (isEvenRow && !isEvenCol) { r = final; g = 0; b = 0; }
          else if (!isEvenRow && isEvenCol) { r = 0; g = 0; b = final; }
          else { r = 0; g = final; b = 0; }
        } else {
          r = g = b = final;
        }

        ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
        ctx.fillRect(startX + col * cellSize, startY + row * cellSize, cellSize, cellSize);
      }
    }

    // Grid lines
    ctx.strokeStyle = "rgba(0,0,0,0.4)"; ctx.lineWidth = 0.5;
    for (let i = 0; i <= res; i++) {
      ctx.beginPath(); ctx.moveTo(startX + i * cellSize, startY); ctx.lineTo(startX + i * cellSize, startY + gridW); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(startX, startY + i * cellSize); ctx.lineTo(startX + gridW, startY + i * cellSize); ctx.stroke();
    }
    ctx.strokeStyle = "hsl(var(--primary) / 0.4)"; ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, gridW, gridW);

    // Info panel
    const infoX = startX + gridW + 30;
    const infoY = startY + 10;
    ctx.fillStyle = "hsl(var(--background) / 0.8)";
    ctx.fillRect(infoX - 5, infoY - 5, 220, 100);
    ctx.font = "11px 'JetBrains Mono', monospace"; ctx.textAlign = "left";
    const vals = [`Resolution:  ${res}×${res}`, `Bit depth:   ${bits}  (${levels} levels)`, `Noise:       ${noise}%`, `CFA:         ${bayerOn ? "Bayer RGGB" : "Mono"}`, `Total px:    ${res * res}`];
    vals.forEach((v, i) => {
      ctx.fillStyle = i === 0 ? "hsl(var(--primary))" : "hsl(var(--muted-foreground))";
      ctx.fillText(v, infoX, infoY + 16 * i + 12);
    });

    labelText(ctx, W / 2, H - 10, `${res}×${res} = ${res * res} pixels  ·  ${bits}-bit depth  ·  ${levels} gray levels`, "hsl(var(--muted-foreground))", "center");
  }, [res, bits, noise, bayerOn]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) { canvas.width = rect.width; canvas.height = 380; }
    };
    resize();
    window.addEventListener("resize", resize);
    const loop = () => { tRef.current += 0.008; draw(); rafRef.current = requestAnimationFrame(loop); };
    loop();
    return () => { cancelAnimationFrame(rafRef.current); window.removeEventListener("resize", resize); };
  }, [draw]);

  return (
    <div className="space-y-3">
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border bg-background" style={{ height: 380 }} />
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 px-1">
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Resolution</span><span className="text-primary">{res}×{res}</span></div>
          <Slider min={4} max={32} step={4} value={[res]} onValueChange={([v]) => setRes(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Bit Depth</span><span className="text-primary">{bits}-bit</span></div>
          <Slider min={1} max={8} step={1} value={[bits]} onValueChange={([v]) => setBits(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Noise</span><span className="text-primary">{noise}%</span></div>
          <Slider min={0} max={50} step={1} value={[noise]} onValueChange={([v]) => setNoise(v)} />
        </div>
      </div>
      <Button variant={bayerOn ? "default" : "outline"} size="sm" className="text-xs" onClick={() => setBayerOn(!bayerOn)}>
        {bayerOn ? "Bayer RGGB" : "Monochrome"}
      </Button>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// SCENE: Perspective Projection (3D grid)
// ═══════════════════════════════════════════════════════════════════════
function PerspectiveScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [f, setF] = useState(150);
  const [tilt, setTilt] = useState(0);
  const [depth, setDepth] = useState(5);
  const [showVP, setShowVP] = useState(true);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2;
    const scale = 30;
    const tiltRad = tilt * Math.PI / 180;

    function project(X: number, Y: number, Z: number) {
      const Yr = Y * Math.cos(tiltRad) - Z * Math.sin(tiltRad);
      const Zr = Y * Math.sin(tiltRad) + Z * Math.cos(tiltRad);
      const Zfinal = Zr + depth * scale * 0.5 + f;
      if (Zfinal <= 0) return null;
      return { x: cx + f * X / Zfinal, y: cy + f * Yr / Zfinal };
    }

    // Ground grid
    const cols = 10, rows = depth;
    for (let i = -cols; i <= cols; i++) {
      ctx.beginPath(); ctx.strokeStyle = "hsl(var(--primary) / 0.12)"; ctx.lineWidth = 1;
      let first = true;
      for (let j = 0; j <= rows * 2; j++) {
        const p = project(i * scale, -scale * 2, j * scale - scale);
        if (!p) { first = true; continue; }
        first ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
        first = false;
      }
      ctx.stroke();
    }
    for (let j = 0; j <= rows * 2; j++) {
      ctx.beginPath();
      ctx.strokeStyle = j === rows ? "hsl(20, 90%, 55% / 0.5)" : "hsl(var(--primary) / 0.07)";
      ctx.lineWidth = j === rows ? 2 : 1;
      let first = true;
      for (let i = -cols; i <= cols; i++) {
        const p = project(i * scale, -scale * 2, j * scale - scale);
        if (!p) { first = true; continue; }
        first ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
        first = false;
      }
      ctx.stroke();
    }

    // Vanishing point
    if (showVP) {
      const vp = project(0, -scale * 2, 1e6);
      if (vp) {
        ctx.strokeStyle = "hsl(270, 70%, 60% / 0.4)"; ctx.lineWidth = 1; ctx.setLineDash([6, 4]);
        for (const gi of [-4, -2, 2, 4]) {
          const near = project(gi * scale, -scale * 2, -scale * 2);
          if (near) {
            ctx.beginPath(); ctx.moveTo(near.x, near.y); ctx.lineTo(vp.x, vp.y); ctx.stroke();
          }
        }
        ctx.setLineDash([]);
        ctx.fillStyle = "hsl(270, 70%, 60%)";
        ctx.beginPath(); ctx.arc(vp.x, clamp(vp.y, 5, H - 5), 6, 0, Math.PI * 2); ctx.fill();
        labelText(ctx, vp.x + 10, clamp(vp.y, 15, H - 5), "Vanishing\nPoint", "hsl(270, 70%, 60%)");
      }
    }

    // Camera frustum
    const frustW = 120, frustH = 80;
    ctx.strokeStyle = "hsl(var(--primary) / 0.5)"; ctx.lineWidth = 1.5;
    ctx.strokeRect(cx - frustW / 2, cy - frustH / 2, frustW, frustH);

    // Camera icon
    ctx.fillStyle = "hsl(var(--primary) / 0.15)";
    ctx.strokeStyle = "hsl(var(--primary) / 0.5)"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.roundRect(cx - 18, cy - 12, 36, 24, 3); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI * 2);
    ctx.fillStyle = "hsl(var(--primary) / 0.3)"; ctx.fill(); ctx.stroke();

    labelText(ctx, cx, H - 15, `f = ${f}   tilt = ${tilt}°`, "hsl(var(--muted-foreground))", "center");
  }, [f, tilt, depth, showVP]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) { canvas.width = rect.width; canvas.height = 380; }
      draw();
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [draw]);

  useEffect(() => { draw(); }, [draw]);

  return (
    <div className="space-y-3">
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border bg-background" style={{ height: 380 }} />
      <div className="grid grid-cols-3 gap-3 px-1">
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Focal Length</span><span className="text-primary">{f}</span></div>
          <Slider min={80} max={300} step={1} value={[f]} onValueChange={([v]) => setF(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Tilt (°)</span><span className="text-primary">{tilt}°</span></div>
          <Slider min={-30} max={30} step={1} value={[tilt]} onValueChange={([v]) => setTilt(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Grid Depth</span><span className="text-primary">{depth}</span></div>
          <Slider min={2} max={10} step={1} value={[depth]} onValueChange={([v]) => setDepth(v)} />
        </div>
      </div>
      <Button variant={showVP ? "default" : "outline"} size="sm" className="text-xs" onClick={() => setShowVP(!showVP)}>
        {showVP ? "Hide Vanishing Points" : "Show Vanishing Points"}
      </Button>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════════
const sceneComponents: Record<string, React.FC> = {
  projection3d: Projection3DScene,
  pinhole: PinholeScene,
  perspective: PerspectiveScene,
  intrinsic: IntrinsicScene,
  lens: LensScene,
  sensor: SensorScene,
};

export default function CameraAnimations() {
  const [activeScene, setActiveScene] = useState("projection3d");
  const ActiveComponent = sceneComponents[activeScene];

  return (
    <div className="rounded-xl border border-border bg-card/50 overflow-hidden">
      <div className="p-4 border-b border-border bg-muted/20">
        <div className="flex items-center gap-2 mb-3">
          <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Camera className="h-4 w-4 text-primary" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">Interactive Animations</h2>
            <p className="text-[11px] text-muted-foreground">Explore camera image formation step by step</p>
          </div>
        </div>

        {/* Scene tabs */}
        <div className="flex gap-1 overflow-x-auto pb-1 -mx-1 px-1">
          {scenes.map((scene) => {
            const Icon = scene.icon;
            const isActive = activeScene === scene.id;
            return (
              <button
                key={scene.id}
                onClick={() => setActiveScene(scene.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono whitespace-nowrap transition-colors border ${
                  isActive
                    ? "bg-primary/10 border-primary/30 text-primary"
                    : "bg-transparent border-transparent text-muted-foreground hover:text-foreground hover:bg-muted/40"
                }`}
              >
                <Icon className="h-3 w-3" />
                {scene.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="p-4">
        {/* Scene description */}
        <p className="text-xs text-muted-foreground mb-3 font-mono">
          // {scenes.find(s => s.id === activeScene)?.description}
        </p>

        <AnimatePresence mode="wait">
          <motion.div
            key={activeScene}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
          >
            <ActiveComponent />
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
