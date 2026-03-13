import { useRef, useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Camera, Move3D, Focus, Grid3X3, Aperture, Cpu } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";

// ─── Utility ─────────────────────────────────────────────────────────
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

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

function useCanvasSetup(height: number, animated = false) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const tRef = useRef(0);

  return { canvasRef, rafRef, tRef, height };
}

// ═══════════════════════════════════════════════════════════════════════
// Wrapper for individual animations with title
// ═══════════════════════════════════════════════════════════════════════
function AnimationWrapper({ title, description, icon: Icon, children }: {
  title: string;
  description: string;
  icon: any;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-border bg-card/50 overflow-hidden">
      <div className="flex items-center gap-2.5 p-3 border-b border-border bg-muted/20">
        <div className="h-7 w-7 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
          <Icon className="h-3.5 w-3.5 text-primary" />
        </div>
        <div>
          <h3 className="text-xs font-semibold text-foreground">{title}</h3>
          <p className="text-[10px] text-muted-foreground">{description}</p>
        </div>
      </div>
      <div className="p-4">
        {children}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// SCENE: 3D → 2D Projection
// ═══════════════════════════════════════════════════════════════════════
export function Projection3DScene() {
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
    const camX = cx, camY = cy;
    const imgPlaneX = camX + f * 0.7;

    const points3D = [
      { X: -120, Y: -80, Z: 300, color: "hsl(0, 85%, 60%)", label: "P₁" },
      { X: 60, Y: -40, Z: 250 + 50 * Math.sin(t * 0.7), color: "hsl(140, 70%, 55%)", label: "P₂" },
      { X: -30 + 40 * Math.cos(t * 0.5), Y: 60, Z: 350, color: "hsl(220, 80%, 60%)", label: "P₃" },
      { X: 80 * Math.sin(t * 0.3), Y: -60 + 30 * Math.sin(t * 0.6), Z: 200, color: "hsl(45, 90%, 55%)", label: "P₄" },
    ];

    // Optical axis
    ctx.strokeStyle = "hsl(var(--primary) / 0.15)";
    ctx.lineWidth = 1; ctx.setLineDash([8, 6]);
    ctx.beginPath(); ctx.moveTo(30, cy); ctx.lineTo(W - 30, cy); ctx.stroke();
    ctx.setLineDash([]);

    // Camera body
    ctx.fillStyle = "hsl(var(--primary) / 0.08)";
    ctx.strokeStyle = "hsl(var(--primary) / 0.5)";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.roundRect(camX - 20, camY - 16, 40, 32, 4); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.arc(camX, camY, 9, 0, Math.PI * 2);
    ctx.fillStyle = "hsl(var(--primary) / 0.2)"; ctx.fill(); ctx.stroke();
    labelText(ctx, camX, camY + 30, "Camera O", "hsl(var(--primary))", "center");

    // Image plane
    ctx.strokeStyle = "hsl(270, 70%, 60%)";
    ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(imgPlaneX, cy - H * 0.38); ctx.lineTo(imgPlaneX, cy + H * 0.38); ctx.stroke();
    labelText(ctx, imgPlaneX + 6, cy - H * 0.36, "Image\nPlane", "hsl(270, 70%, 60%)");

    for (const pt of points3D) {
      const worldX = camX - pt.Z * 0.6;
      const worldY = camY - pt.Y * 0.5;
      const imgY = camY + (f * 0.7 * pt.Y) / pt.Z;
      const rayProgress = (Math.sin(t * 1.2 + pt.Z * 0.01) * 0.5 + 0.5);

      ctx.fillStyle = pt.color;
      ctx.beginPath(); ctx.arc(worldX, worldY, 7, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = pt.color; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(worldX, worldY, 11, 0, Math.PI * 2); ctx.stroke();
      labelText(ctx, worldX - 14, worldY - 16, pt.label, pt.color, "center");

      ctx.globalAlpha = 0.2;
      drawRay(ctx, worldX, worldY, camX, camY, pt.color, 1);
      drawRay(ctx, camX, camY, imgPlaneX, imgY, pt.color, 1);
      ctx.globalAlpha = 1;

      const particleSegment = rayProgress < 0.5 ? 0 : 1;
      let px: number, py: number;
      if (particleSegment === 0) {
        const p = rayProgress * 2;
        px = lerp(worldX, camX, p); py = lerp(worldY, camY, p);
      } else {
        const p = (rayProgress - 0.5) * 2;
        px = lerp(camX, imgPlaneX, p); py = lerp(camY, imgY, p);
      }

      const gradient = ctx.createRadialGradient(px, py, 0, px, py, 12);
      gradient.addColorStop(0, pt.color); gradient.addColorStop(1, "transparent");
      ctx.fillStyle = gradient;
      ctx.beginPath(); ctx.arc(px, py, 12, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = pt.color;
      ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();

      ctx.fillStyle = pt.color; ctx.globalAlpha = 0.8;
      ctx.beginPath(); ctx.arc(imgPlaneX, imgY, 4, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Formula
    ctx.fillStyle = "hsl(var(--background) / 0.85)";
    ctx.fillRect(W - 220, 10, 210, 50);
    ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1;
    ctx.strokeRect(W - 220, 10, 210, 50);
    ctx.font = "12px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsl(var(--primary))"; ctx.textAlign = "left";
    ctx.fillText("x' = f · X / Z", W - 210, 32);
    ctx.fillText("y' = f · Y / Z", W - 210, 50);

    ctx.strokeStyle = "hsl(var(--muted-foreground) / 0.4)";
    ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(camX, cy + H * 0.35); ctx.lineTo(imgPlaneX, cy + H * 0.35); ctx.stroke();
    ctx.setLineDash([]);
    labelText(ctx, (camX + imgPlaneX) / 2, cy + H * 0.35 + 14, `f = ${focalLength}`, "hsl(var(--muted-foreground))", "center");
  }, [focalLength]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => { const r = canvas.parentElement?.getBoundingClientRect(); if (r) { canvas.width = r.width; canvas.height = 400; } };
    resize(); window.addEventListener("resize", resize);
    const loop = () => { tRef.current += 0.02; draw(); rafRef.current = requestAnimationFrame(loop); };
    loop();
    return () => { cancelAnimationFrame(rafRef.current); window.removeEventListener("resize", resize); };
  }, [draw]);

  return (
    <AnimationWrapper title="3D → 2D Projection" description="Watch how 3D world points project onto the camera image plane" icon={Move3D}>
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border bg-background" style={{ height: 400 }} />
      <div className="flex items-center gap-4 px-1 mt-3">
        <span className="text-xs text-muted-foreground font-mono whitespace-nowrap">Focal Length (f)</span>
        <Slider min={80} max={320} step={1} value={[focalLength]} onValueChange={([v]) => setFocalLength(v)} className="flex-1" />
        <span className="text-xs font-mono text-primary w-12 text-right">{focalLength}px</span>
      </div>
    </AnimationWrapper>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// SCENE: Pinhole Camera
// ═══════════════════════════════════════════════════════════════════════
export function PinholeScene() {
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

    ctx.strokeStyle = "hsl(var(--primary) / 0.15)";
    ctx.lineWidth = 1; ctx.setLineDash([8, 6]);
    ctx.beginPath(); ctx.moveTo(30, cy); ctx.lineTo(W - 30, cy); ctx.stroke();
    ctx.setLineDash([]);

    drawArrow(ctx, objX, cy, objX, cy - objHeight + objAnim, "hsl(20, 90%, 55%)", 3);

    ctx.strokeStyle = "hsl(270, 70%, 60%)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(imgX, cy - H * 0.4); ctx.lineTo(imgX, cy + H * 0.4); ctx.stroke();

    ctx.strokeStyle = "hsl(var(--primary) / 0.5)"; ctx.lineWidth = 3;
    const gap = 12;
    ctx.beginPath(); ctx.moveTo(cxp, cy - H * 0.4); ctx.lineTo(cxp, cy - gap); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cxp, cy + gap); ctx.lineTo(cxp, cy + H * 0.4); ctx.stroke();
    ctx.fillStyle = "hsl(var(--primary) / 0.15)";
    ctx.fillRect(cxp - 3, cy - H * 0.4, 6, H * 0.8);
    ctx.fillStyle = "hsl(var(--primary))";
    ctx.beginPath(); ctx.arc(cxp, cy, 4, 0, Math.PI * 2); ctx.fill();

    const nRays = showAllRays ? 7 : 2;
    for (let i = 0; i < nRays; i++) {
      const t = nRays <= 1 ? 0.5 : i / (nRays - 1);
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

    drawArrow(ctx, imgX, cy, imgX, cy + Math.abs(imgH) + objAnim * (f / objDist), "hsl(270, 70%, 60%)", 2);

    labelText(ctx, objX, cy + 20, "Object", "hsl(20, 90%, 55%)");
    labelText(ctx, cxp, cy - H * 0.38, "Pinhole", "hsl(var(--primary))");
    labelText(ctx, imgX + 8, cy - H * 0.38, "Image\nPlane", "hsl(270, 70%, 60%)");

    ctx.strokeStyle = "hsl(var(--muted-foreground) / 0.15)"; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
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
    const resize = () => { const r = canvas.parentElement?.getBoundingClientRect(); if (r) { canvas.width = r.width; canvas.height = 380; } };
    resize(); window.addEventListener("resize", resize);
    const loop = () => { tRef.current += 0.015; draw(); rafRef.current = requestAnimationFrame(loop); };
    loop();
    return () => { cancelAnimationFrame(rafRef.current); window.removeEventListener("resize", resize); };
  }, [draw]);

  return (
    <AnimationWrapper title="Pinhole Camera Model" description="Light passes through a single point — the center of projection" icon={Camera}>
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border bg-background" style={{ height: 380 }} />
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 px-1 mt-3">
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
      <Button variant={showAllRays ? "default" : "outline"} size="sm" className="text-xs mt-2" onClick={() => setShowAllRays(!showAllRays)}>
        {showAllRays ? "Hide Extra Rays" : "Show All Rays"}
      </Button>
    </AnimationWrapper>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// SCENE: Perspective Projection — IMPROVED
// Shows a 3D house/scene with clear similar triangles + step-by-step
// ═══════════════════════════════════════════════════════════════════════
export function PerspectiveScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const tRef = useRef(0);
  const [focalLength, setFocalLength] = useState(200);
  const [pointX, setPointX] = useState(80);
  const [pointY, setPointY] = useState(-60);
  const [pointZ, setPointZ] = useState(300);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const t = tRef.current;
    const f = focalLength;

    // Layout: left side = 3D side view, right side = image plane view
    const splitX = W * 0.6;
    const margin = 40;

    // === LEFT: Side view showing similar triangles ===
    const camX = margin + 40;
    const camY = H * 0.5;
    const imgPlaneX = camX + f * 0.5;

    // Animated point
    const animX = pointX + 15 * Math.sin(t * 0.4);
    const animY = pointY + 10 * Math.cos(t * 0.5);
    const Z = pointZ;

    // World point position in side view
    const worldScreenX = camX + Z * 0.8;
    const worldScreenY = camY + animY * 0.6;

    // Projected y on image plane
    const projY = f * animY / Z;
    const imgScreenY = camY + projY * 0.5;

    // Background label
    ctx.fillStyle = "hsl(var(--muted-foreground) / 0.08)";
    ctx.fillRect(0, 0, splitX - 10, H);

    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsl(var(--muted-foreground) / 0.4)";
    ctx.textAlign = "center";
    ctx.fillText("SIDE VIEW — Similar Triangles", (splitX - 10) / 2, 16);

    // Optical axis
    ctx.strokeStyle = "hsl(var(--primary) / 0.12)";
    ctx.lineWidth = 1; ctx.setLineDash([6, 6]);
    ctx.beginPath(); ctx.moveTo(camX - 20, camY); ctx.lineTo(splitX - 20, camY); ctx.stroke();
    ctx.setLineDash([]);

    // Camera center
    ctx.fillStyle = "hsl(var(--primary))";
    ctx.beginPath(); ctx.arc(camX, camY, 6, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "hsl(var(--primary) / 0.3)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(camX, camY, 12, 0, Math.PI * 2); ctx.stroke();
    labelText(ctx, camX, camY + 22, "O (0,0,0)", "hsl(var(--primary))", "center");

    // Image plane line
    ctx.strokeStyle = "hsl(270, 70%, 60%)"; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(imgPlaneX, camY - H * 0.4); ctx.lineTo(imgPlaneX, camY + H * 0.4); ctx.stroke();
    labelText(ctx, imgPlaneX, camY - H * 0.4 - 6, "z = f", "hsl(270, 70%, 60%)", "center");

    // Z axis label
    ctx.strokeStyle = "hsl(var(--muted-foreground) / 0.3)"; ctx.lineWidth = 1;
    drawArrow(ctx, camX + 20, camY + H * 0.38, splitX - 30, camY + H * 0.38, "hsl(var(--muted-foreground) / 0.4)", 1.5);
    labelText(ctx, splitX - 50, camY + H * 0.38 + 14, "Z (depth)", "hsl(var(--muted-foreground) / 0.5)", "right");

    // World point
    const ptColor = "hsl(0, 85%, 60%)";
    ctx.fillStyle = ptColor;
    ctx.beginPath(); ctx.arc(worldScreenX, worldScreenY, 8, 0, Math.PI * 2); ctx.fill();
    const glow = ctx.createRadialGradient(worldScreenX, worldScreenY, 0, worldScreenX, worldScreenY, 20);
    glow.addColorStop(0, "hsl(0, 85%, 60% / 0.3)"); glow.addColorStop(1, "transparent");
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(worldScreenX, worldScreenY, 20, 0, Math.PI * 2); ctx.fill();
    labelText(ctx, worldScreenX + 12, worldScreenY - 12, `P(${Math.round(animX)}, ${Math.round(animY)}, ${Z})`, ptColor);

    // Ray from O through P to image plane
    ctx.globalAlpha = 0.3;
    drawRay(ctx, camX, camY, worldScreenX, worldScreenY, ptColor, 1.5);
    ctx.globalAlpha = 1;

    // Projection ray (O to image plane intersection)
    drawRay(ctx, camX, camY, imgPlaneX, imgScreenY, "hsl(270, 70%, 60%)", 2);

    // Projected point on image plane
    ctx.fillStyle = "hsl(270, 70%, 60%)";
    ctx.beginPath(); ctx.arc(imgPlaneX, imgScreenY, 5, 0, Math.PI * 2); ctx.fill();
    const projLabel = `p(${(f * animX / Z).toFixed(1)}, ${(f * animY / Z).toFixed(1)})`;
    labelText(ctx, imgPlaneX - 8, imgScreenY + 16, projLabel, "hsl(270, 70%, 60%)", "right");

    // Similar triangles visualization
    // Triangle 1: O to image plane
    ctx.strokeStyle = "hsl(220, 80%, 60% / 0.6)"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    // Vertical at image plane
    ctx.beginPath(); ctx.moveTo(imgPlaneX, camY); ctx.lineTo(imgPlaneX, imgScreenY); ctx.stroke();
    // Vertical at world point
    ctx.strokeStyle = "hsl(0, 85%, 60% / 0.4)";
    ctx.beginPath(); ctx.moveTo(worldScreenX, camY); ctx.lineTo(worldScreenX, worldScreenY); ctx.stroke();
    ctx.setLineDash([]);

    // f annotation
    ctx.strokeStyle = "hsl(220, 80%, 60% / 0.5)"; ctx.lineWidth = 1;
    const annY = camY + H * 0.28;
    drawArrow(ctx, camX, annY, imgPlaneX, annY, "hsl(220, 80%, 60% / 0.6)", 1.5);
    labelText(ctx, (camX + imgPlaneX) / 2, annY + 12, `f = ${f}`, "hsl(220, 80%, 60%)", "center");

    // Z annotation
    drawArrow(ctx, camX, annY + 24, worldScreenX, annY + 24, "hsl(0, 85%, 60% / 0.5)", 1.5);
    labelText(ctx, (camX + worldScreenX) / 2, annY + 36, `Z = ${Z}`, "hsl(0, 85%, 60% / 0.5)", "center");

    // Y/Y' annotations
    const yAnnX = imgPlaneX - 14;
    const dy = Math.abs(imgScreenY - camY);
    if (dy > 10) {
      labelText(ctx, yAnnX, (camY + imgScreenY) / 2, "y'", "hsl(270, 70%, 60%)", "right");
    }
    const dyW = Math.abs(worldScreenY - camY);
    if (dyW > 10) {
      labelText(ctx, worldScreenX + 10, (camY + worldScreenY) / 2, "Y", ptColor);
    }

    // === RIGHT: Result panel with formula and image plane view ===
    const rightX = splitX + 10;
    const rightW = W - rightX - 10;

    ctx.fillStyle = "hsl(var(--muted-foreground) / 0.04)";
    ctx.fillRect(rightX, 0, rightW, H);

    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsl(var(--muted-foreground) / 0.4)";
    ctx.textAlign = "center";
    ctx.fillText("IMAGE PLANE VIEW", rightX + rightW / 2, 16);

    // Draw image plane rectangle
    const ipW = rightW * 0.7;
    const ipH = ipW * 0.75;
    const ipX = rightX + (rightW - ipW) / 2;
    const ipY = 30;

    ctx.strokeStyle = "hsl(270, 70%, 60% / 0.5)"; ctx.lineWidth = 2;
    ctx.strokeRect(ipX, ipY, ipW, ipH);
    ctx.fillStyle = "hsl(270, 70%, 60% / 0.03)";
    ctx.fillRect(ipX, ipY, ipW, ipH);

    // Grid on image plane
    ctx.strokeStyle = "hsl(var(--border) / 0.3)"; ctx.lineWidth = 0.5;
    for (let i = 1; i < 8; i++) {
      const gx = ipX + (ipW / 8) * i;
      ctx.beginPath(); ctx.moveTo(gx, ipY); ctx.lineTo(gx, ipY + ipH); ctx.stroke();
      const gy = ipY + (ipH / 6) * (i > 6 ? 6 : i);
      if (i <= 6) { ctx.beginPath(); ctx.moveTo(ipX, gy); ctx.lineTo(ipX + ipW, gy); ctx.stroke(); }
    }

    // Principal point
    const ppX = ipX + ipW / 2;
    const ppY = ipY + ipH / 2;
    ctx.strokeStyle = "hsl(var(--primary) / 0.2)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(ppX, ipY); ctx.lineTo(ppX, ipY + ipH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ipX, ppY); ctx.lineTo(ipX + ipW, ppY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "hsl(var(--primary) / 0.3)";
    ctx.beginPath(); ctx.arc(ppX, ppY, 3, 0, Math.PI * 2); ctx.fill();

    // Projected point on image plane view
    const projPxX = ppX + (f * animX / Z) * 0.4;
    const projPxY = ppY + (f * animY / Z) * 0.4;

    // Clamp to image plane bounds
    const cpx = clamp(projPxX, ipX + 4, ipX + ipW - 4);
    const cpy = clamp(projPxY, ipY + 4, ipY + ipH - 4);

    ctx.fillStyle = ptColor;
    ctx.beginPath(); ctx.arc(cpx, cpy, 6, 0, Math.PI * 2); ctx.fill();
    const glow2 = ctx.createRadialGradient(cpx, cpy, 0, cpx, cpy, 16);
    glow2.addColorStop(0, "hsl(0, 85%, 60% / 0.4)"); glow2.addColorStop(1, "transparent");
    ctx.fillStyle = glow2;
    ctx.beginPath(); ctx.arc(cpx, cpy, 16, 0, Math.PI * 2); ctx.fill();

    // Formula box
    const fBoxY = ipY + ipH + 20;
    ctx.fillStyle = "hsl(var(--background) / 0.9)";
    ctx.fillRect(rightX + 10, fBoxY, rightW - 20, 90);
    ctx.strokeStyle = "hsl(var(--border))"; ctx.lineWidth = 1;
    ctx.strokeRect(rightX + 10, fBoxY, rightW - 20, 90);

    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsl(var(--muted-foreground) / 0.5)"; ctx.textAlign = "left";
    ctx.fillText("Similar Triangles:", rightX + 20, fBoxY + 18);

    ctx.fillStyle = "hsl(var(--primary))";
    ctx.font = "13px 'JetBrains Mono', monospace";
    ctx.fillText("y'/f = Y/Z", rightX + 20, fBoxY + 38);
    ctx.fillText("y' = f·Y/Z", rightX + 20, fBoxY + 56);

    ctx.fillStyle = "hsl(var(--muted-foreground))";
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillText(`= ${f}·${Math.round(animY)}/${Z}`, rightX + 20, fBoxY + 74);
    ctx.fillText(`= ${(f * animY / Z).toFixed(1)} px`, rightX + 120, fBoxY + 74);

    // Similar triangles callout
    const triY = fBoxY + 105;
    ctx.fillStyle = "hsl(220, 80%, 60% / 0.08)";
    ctx.fillRect(rightX + 10, triY, rightW - 20, 36);
    ctx.strokeStyle = "hsl(220, 80%, 60% / 0.3)"; ctx.lineWidth = 1;
    ctx.strokeRect(rightX + 10, triY, rightW - 20, 36);
    ctx.fillStyle = "hsl(220, 80%, 60%)";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("△(O, f, y') ~ △(O, Z, Y)", rightX + rightW / 2, triY + 14);
    ctx.fillStyle = "hsl(var(--muted-foreground))";
    ctx.fillText("Smaller f → wider FOV, larger f → zoom", rightX + rightW / 2, triY + 28);

  }, [focalLength, pointX, pointY, pointZ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => { const r = canvas.parentElement?.getBoundingClientRect(); if (r) { canvas.width = r.width; canvas.height = 440; } };
    resize(); window.addEventListener("resize", resize);
    const loop = () => { tRef.current += 0.015; draw(); rafRef.current = requestAnimationFrame(loop); };
    loop();
    return () => { cancelAnimationFrame(rafRef.current); window.removeEventListener("resize", resize); };
  }, [draw]);

  return (
    <AnimationWrapper title="Perspective Projection — Similar Triangles" description="See how y'/f = Y/Z creates the projection via similar triangles" icon={Grid3X3}>
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border bg-background" style={{ height: 440 }} />
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 px-1 mt-3">
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Focal (f)</span><span className="text-primary">{focalLength}</span></div>
          <Slider min={80} max={400} step={1} value={[focalLength]} onValueChange={([v]) => setFocalLength(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>X</span><span className="text-primary">{pointX}</span></div>
          <Slider min={-150} max={150} step={1} value={[pointX]} onValueChange={([v]) => setPointX(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Y</span><span className="text-primary">{pointY}</span></div>
          <Slider min={-120} max={120} step={1} value={[pointY]} onValueChange={([v]) => setPointY(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Z (depth)</span><span className="text-primary">{pointZ}</span></div>
          <Slider min={100} max={600} step={1} value={[pointZ]} onValueChange={([v]) => setPointZ(v)} />
        </div>
      </div>
    </AnimationWrapper>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// SCENE: Intrinsic Matrix K — IMPROVED
// Shows camera coords → pixel coords pipeline with live K matrix
// ═══════════════════════════════════════════════════════════════════════
export function IntrinsicScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const tRef = useRef(0);
  const [fx, setFx] = useState(500);
  const [fy, setFy] = useState(500);
  const [cxOff, setCxOff] = useState(320);
  const [cyOff, setCyOff] = useState(240);
  const [skew, setSkew] = useState(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const t = tRef.current;

    // Three panels: Normalized coords | K matrix | Pixel coords
    const panelW = (W - 30) / 3;
    const panels = [
      { x: 5, label: "NORMALIZED COORDS" },
      { x: panelW + 15, label: "INTRINSIC MATRIX K" },
      { x: 2 * panelW + 25, label: "PIXEL COORDS" },
    ];

    // Panel backgrounds
    panels.forEach((p, i) => {
      ctx.fillStyle = i === 1 ? "hsl(270, 70%, 60% / 0.04)" : "hsl(var(--muted-foreground) / 0.03)";
      ctx.fillRect(p.x, 0, panelW, H);
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillStyle = "hsl(var(--muted-foreground) / 0.4)";
      ctx.textAlign = "center";
      ctx.fillText(p.label, p.x + panelW / 2, 14);
    });

    // Flow arrows between panels
    const arrowY = H / 2;
    ctx.fillStyle = "hsl(var(--primary) / 0.6)";
    for (const ax of [panelW + 8, 2 * panelW + 18]) {
      ctx.beginPath();
      ctx.moveTo(ax, arrowY - 8);
      ctx.lineTo(ax + 8, arrowY);
      ctx.lineTo(ax, arrowY + 8);
      ctx.closePath(); ctx.fill();
    }

    // === Panel 1: Normalized coordinate plane ===
    const normCx = panels[0].x + panelW / 2;
    const normCy = H * 0.48;
    const normScale = Math.min(panelW * 0.35, H * 0.3);

    // Grid
    ctx.strokeStyle = "hsl(var(--border) / 0.3)"; ctx.lineWidth = 0.5;
    for (let i = -3; i <= 3; i++) {
      const gx = normCx + (i / 3) * normScale;
      ctx.beginPath(); ctx.moveTo(gx, normCy - normScale); ctx.lineTo(gx, normCy + normScale); ctx.stroke();
      const gy = normCy + (i / 3) * normScale;
      ctx.beginPath(); ctx.moveTo(normCx - normScale, gy); ctx.lineTo(normCx + normScale, gy); ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = "hsl(var(--primary) / 0.5)"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(normCx - normScale, normCy); ctx.lineTo(normCx + normScale, normCy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(normCx, normCy - normScale); ctx.lineTo(normCx, normCy + normScale); ctx.stroke();
    labelText(ctx, normCx + normScale + 4, normCy - 4, "x_n", "hsl(var(--primary) / 0.5)");
    labelText(ctx, normCx + 6, normCy - normScale - 4, "y_n", "hsl(var(--primary) / 0.5)");

    // Animated test points in normalized coords
    const testPoints = [
      { xn: 0.4 * Math.cos(t * 0.3), yn: -0.3 * Math.sin(t * 0.4), color: "hsl(0, 85%, 60%)", label: "P₁" },
      { xn: -0.5, yn: 0.5 + 0.2 * Math.sin(t * 0.5), color: "hsl(140, 70%, 55%)", label: "P₂" },
      { xn: 0.6 + 0.1 * Math.cos(t * 0.6), yn: 0.1, color: "hsl(45, 90%, 55%)", label: "P₃" },
    ];

    testPoints.forEach((pt) => {
      const sx = normCx + pt.xn * normScale;
      const sy = normCy + pt.yn * normScale;
      ctx.fillStyle = pt.color;
      ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI * 2); ctx.fill();
      labelText(ctx, sx + 8, sy - 4, `${pt.label}(${pt.xn.toFixed(2)}, ${pt.yn.toFixed(2)})`, pt.color);
    });

    labelText(ctx, normCx, normCy + normScale + 16, "(0,0) = optical axis", "hsl(var(--muted-foreground) / 0.5)", "center");

    // === Panel 2: K matrix visualization ===
    const kCx = panels[1].x + panelW / 2;
    const kTop = 28;

    // Matrix display
    ctx.fillStyle = "hsl(var(--background) / 0.9)";
    const matX = kCx - panelW * 0.42;
    const matW = panelW * 0.84;
    ctx.fillRect(matX, kTop, matW, 76);
    ctx.strokeStyle = "hsl(270, 70%, 60% / 0.5)"; ctx.lineWidth = 2;
    // Left bracket
    ctx.beginPath(); ctx.moveTo(matX + 8, kTop + 4); ctx.lineTo(matX + 2, kTop + 4);
    ctx.lineTo(matX + 2, kTop + 72); ctx.lineTo(matX + 8, kTop + 72); ctx.stroke();
    // Right bracket
    ctx.beginPath(); ctx.moveTo(matX + matW - 8, kTop + 4); ctx.lineTo(matX + matW - 2, kTop + 4);
    ctx.lineTo(matX + matW - 2, kTop + 72); ctx.lineTo(matX + matW - 8, kTop + 72); ctx.stroke();

    ctx.font = "12px 'JetBrains Mono', monospace"; ctx.textAlign = "center";
    // Row 1
    ctx.fillStyle = "hsl(220, 80%, 60%)"; ctx.fillText(`${fx}`, kCx - panelW * 0.25, kTop + 22);
    ctx.fillStyle = "hsl(45, 90%, 55%)"; ctx.fillText(`${skew}`, kCx, kTop + 22);
    ctx.fillStyle = "hsl(20, 90%, 55%)"; ctx.fillText(`${cxOff}`, kCx + panelW * 0.25, kTop + 22);
    // Row 2
    ctx.fillStyle = "hsl(var(--muted-foreground) / 0.3)"; ctx.fillText("0", kCx - panelW * 0.25, kTop + 46);
    ctx.fillStyle = "hsl(220, 80%, 60%)"; ctx.fillText(`${fy}`, kCx, kTop + 46);
    ctx.fillStyle = "hsl(20, 90%, 55%)"; ctx.fillText(`${cyOff}`, kCx + panelW * 0.25, kTop + 46);
    // Row 3
    ctx.fillStyle = "hsl(var(--muted-foreground) / 0.3)";
    ctx.fillText("0", kCx - panelW * 0.25, kTop + 68);
    ctx.fillText("0", kCx, kTop + 68);
    ctx.fillText("1", kCx + panelW * 0.25, kTop + 68);

    // Labels for matrix elements
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsl(220, 80%, 60% / 0.6)"; ctx.fillText("fx", kCx - panelW * 0.25, kTop + 88);
    ctx.fillStyle = "hsl(45, 90%, 55% / 0.6)"; ctx.fillText("γ", kCx, kTop + 88);
    ctx.fillStyle = "hsl(20, 90%, 55% / 0.6)"; ctx.fillText("cx", kCx + panelW * 0.25, kTop + 88);
    ctx.fillStyle = "hsl(220, 80%, 60% / 0.6)"; ctx.fillText("fy", kCx, kTop + 98);
    ctx.fillStyle = "hsl(20, 90%, 55% / 0.6)"; ctx.fillText("cy", kCx + panelW * 0.25, kTop + 98);

    // Equation
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsl(var(--primary))"; ctx.textAlign = "center";
    ctx.fillText("u = fx·xn + γ·yn + cx", kCx, kTop + 128);
    ctx.fillText("v = fy·yn + cy", kCx, kTop + 145);

    // Parameter explanations
    const expY = kTop + 170;
    const params = [
      { label: "fx, fy", desc: "focal lengths (scale)", color: "hsl(220, 80%, 60%)" },
      { label: "cx, cy", desc: "principal point (shift)", color: "hsl(20, 90%, 55%)" },
      { label: "γ", desc: "skew (shear)", color: "hsl(45, 90%, 55%)" },
    ];
    params.forEach((p, i) => {
      ctx.fillStyle = p.color;
      ctx.beginPath(); ctx.arc(panels[1].x + 16, expY + i * 18, 4, 0, Math.PI * 2); ctx.fill();
      ctx.font = "10px 'JetBrains Mono', monospace"; ctx.textAlign = "left";
      ctx.fillText(`${p.label}: ${p.desc}`, panels[1].x + 26, expY + i * 18 + 4);
    });

    // === Panel 3: Pixel coordinate plane ===
    const pixCx = panels[2].x + panelW / 2;
    const pixCy = H * 0.48;
    const imgW = Math.min(panelW * 0.8, H * 0.7);
    const imgH = imgW * 0.75;
    const imgLeft = pixCx - imgW / 2;
    const imgTop = pixCy - imgH / 2;

    // Image rectangle
    ctx.strokeStyle = "hsl(var(--primary) / 0.3)"; ctx.lineWidth = 1.5;
    ctx.strokeRect(imgLeft, imgTop, imgW, imgH);
    ctx.fillStyle = "hsl(var(--primary) / 0.02)";
    ctx.fillRect(imgLeft, imgTop, imgW, imgH);

    // Pixel grid
    ctx.strokeStyle = "hsl(var(--border) / 0.15)"; ctx.lineWidth = 0.5;
    for (let i = 1; i < 10; i++) {
      const gx = imgLeft + (imgW / 10) * i;
      ctx.beginPath(); ctx.moveTo(gx, imgTop); ctx.lineTo(gx, imgTop + imgH); ctx.stroke();
    }
    for (let i = 1; i < 8; i++) {
      const gy = imgTop + (imgH / 8) * i;
      ctx.beginPath(); ctx.moveTo(imgLeft, gy); ctx.lineTo(imgLeft + imgW, gy); ctx.stroke();
    }

    // Origin label (top-left)
    ctx.fillStyle = "hsl(var(--muted-foreground) / 0.4)";
    ctx.font = "9px 'JetBrains Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText("(0,0)", imgLeft + 2, imgTop - 3);
    ctx.fillText(`(${640},${480})`, imgLeft + imgW - 50, imgTop + imgH + 12);

    // Principal point in pixel space
    const ppScreenX = imgLeft + (cxOff / 640) * imgW;
    const ppScreenY = imgTop + (cyOff / 480) * imgH;
    ctx.strokeStyle = "hsl(20, 90%, 55% / 0.3)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(ppScreenX, imgTop); ctx.lineTo(ppScreenX, imgTop + imgH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(imgLeft, ppScreenY); ctx.lineTo(imgLeft + imgW, ppScreenY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "hsl(20, 90%, 55%)";
    ctx.beginPath(); ctx.arc(ppScreenX, ppScreenY, 4, 0, Math.PI * 2); ctx.fill();
    labelText(ctx, ppScreenX + 6, ppScreenY - 8, `(cx,cy)`, "hsl(20, 90%, 55%)");

    // Project test points into pixel space
    testPoints.forEach((pt) => {
      const u = fx * pt.xn + skew * pt.yn + cxOff;
      const v = fy * pt.yn + cyOff;
      // Map to screen
      const sx = imgLeft + (u / 640) * imgW;
      const sy = imgTop + (v / 480) * imgH;
      const csx = clamp(sx, imgLeft + 3, imgLeft + imgW - 3);
      const csy = clamp(sy, imgTop + 3, imgTop + imgH - 3);

      ctx.fillStyle = pt.color;
      ctx.beginPath(); ctx.arc(csx, csy, 6, 0, Math.PI * 2); ctx.fill();
      const gl = ctx.createRadialGradient(csx, csy, 0, csx, csy, 14);
      gl.addColorStop(0, pt.color.replace(")", " / 0.3)").replace("hsl(", "hsla(")); gl.addColorStop(1, "transparent");
      ctx.fillStyle = gl;
      ctx.beginPath(); ctx.arc(csx, csy, 14, 0, Math.PI * 2); ctx.fill();
      labelText(ctx, csx + 8, csy - 4, `${pt.label}(${Math.round(u)}, ${Math.round(v)})`, pt.color);
    });

    // U/V axis labels
    drawArrow(ctx, imgLeft, imgTop + imgH + 6, imgLeft + imgW, imgTop + imgH + 6, "hsl(var(--muted-foreground) / 0.3)", 1);
    labelText(ctx, imgLeft + imgW - 10, imgTop + imgH + 18, "u", "hsl(var(--muted-foreground) / 0.4)", "right");
    drawArrow(ctx, imgLeft - 6, imgTop, imgLeft - 6, imgTop + imgH, "hsl(var(--muted-foreground) / 0.3)", 1);
    labelText(ctx, imgLeft - 16, imgTop + imgH - 10, "v", "hsl(var(--muted-foreground) / 0.4)");

  }, [fx, fy, cxOff, cyOff, skew]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => { const r = canvas.parentElement?.getBoundingClientRect(); if (r) { canvas.width = r.width; canvas.height = 440; } };
    resize(); window.addEventListener("resize", resize);
    const loop = () => { tRef.current += 0.015; draw(); rafRef.current = requestAnimationFrame(loop); };
    loop();
    return () => { cancelAnimationFrame(rafRef.current); window.removeEventListener("resize", resize); };
  }, [draw]);

  return (
    <AnimationWrapper title="Intrinsic Matrix K — Camera to Pixel Coordinates" description="See how K transforms normalized camera coordinates into pixel coordinates" icon={Cpu}>
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border bg-background" style={{ height: 440 }} />
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 px-1 mt-3">
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
          <Slider min={100} max={540} step={1} value={[cxOff]} onValueChange={([v]) => setCxOff(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>cy</span><span className="text-primary">{cyOff}</span></div>
          <Slider min={100} max={380} step={1} value={[cyOff]} onValueChange={([v]) => setCyOff(v)} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Skew γ</span><span className="text-primary">{skew}</span></div>
          <Slider min={-200} max={200} step={1} value={[skew]} onValueChange={([v]) => setSkew(v)} />
        </div>
      </div>
    </AnimationWrapper>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// SCENE: Lens & Depth of Field
// ═══════════════════════════════════════════════════════════════════════
export function LensScene() {
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

    ctx.strokeStyle = "hsl(var(--primary) / 0.1)"; ctx.lineWidth = 1; ctx.setLineDash([8, 6]);
    ctx.beginPath(); ctx.moveTo(20, cy); ctx.lineTo(W - 20, cy); ctx.stroke();
    ctx.setLineDash([]);

    ctx.save();
    ctx.strokeStyle = "hsl(var(--primary) / 0.6)"; ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(lensX, cy - halfAper);
    ctx.bezierCurveTo(lensX + 30, cy - halfAper / 2, lensX + 30, cy + halfAper / 2, lensX, cy + halfAper);
    ctx.bezierCurveTo(lensX - 30, cy + halfAper / 2, lensX - 30, cy - halfAper / 2, lensX, cy - halfAper);
    ctx.fillStyle = "hsl(var(--primary) / 0.05)"; ctx.fill(); ctx.stroke();
    ctx.restore();
    labelText(ctx, lensX + 35, cy - halfAper + 10, "Lens", "hsl(var(--primary))");

    const di_focus = thinLensImageDist(fmm * scale, focusDist);
    const sensorX = lensX + di_focus;
    ctx.strokeStyle = "hsl(270, 70%, 60%)"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(sensorX, cy - H * 0.35); ctx.lineTo(sensorX, cy + H * 0.35); ctx.stroke();
    labelText(ctx, sensorX + 6, cy - H * 0.33, "Sensor", "hsl(270, 70%, 60%)");

    const objFocusX = lensX - focusDist;
    if (objFocusX > 20) {
      drawArrow(ctx, objFocusX, cy, objFocusX, cy - 50, "hsl(var(--primary))", 2.5);
      labelText(ctx, objFocusX, cy + 15, "In focus", "hsl(var(--primary))");
      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = "hsl(var(--primary))"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(objFocusX, cy - 50); ctx.lineTo(lensX, cy - halfAper); ctx.lineTo(sensorX, cy + 50); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(objFocusX, cy - 50); ctx.lineTo(lensX, cy + halfAper); ctx.lineTo(sensorX, cy + 50); ctx.stroke();
      ctx.globalAlpha = 0.8;
      ctx.strokeStyle = "hsl(var(--primary))"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(objFocusX, cy - 50); ctx.lineTo(sensorX, cy + 50); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = "hsl(var(--primary))";
      ctx.beginPath(); ctx.arc(sensorX, cy + 50, 3, 0, Math.PI * 2); ctx.fill();
    }

    const objBlurX = lensX - objDist2;
    if (objBlurX > 20 && objBlurX < lensX - 20) {
      drawArrow(ctx, objBlurX, cy, objBlurX, cy - 60, "hsl(20, 90%, 55%)", 2.5);
      labelText(ctx, objBlurX, cy + 15, "Out of\nfocus", "hsl(20, 90%, 55%)");
      const di_blur = thinLensImageDist(fmm * scale, objDist2);
      const sharpPtX = lensX + di_blur;
      const CoC = halfAper * Math.abs(sensorX - sharpPtX) / Math.max(sharpPtX, 1);
      ctx.globalAlpha = 0.4;
      ctx.strokeStyle = "hsl(20, 90%, 55%)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(objBlurX, cy - 60); ctx.lineTo(lensX, cy - halfAper); ctx.lineTo(sensorX, cy + 60 * (di_focus / di_blur)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(objBlurX, cy - 60); ctx.lineTo(lensX, cy + halfAper); ctx.lineTo(sensorX, cy + 60 * (di_focus / di_blur)); ctx.stroke();
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
    const resize = () => { const r = canvas.parentElement?.getBoundingClientRect(); if (r) { canvas.width = r.width; canvas.height = 380; } draw(); };
    resize(); window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [draw]);

  useEffect(() => { draw(); }, [draw]);

  return (
    <AnimationWrapper title="Lens & Depth of Field" description="Real lenses blur out-of-focus points into circles of confusion" icon={Aperture}>
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border bg-background" style={{ height: 380 }} />
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 px-1 mt-3">
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
    </AnimationWrapper>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// SCENE: Sensor & Pixels
// ═══════════════════════════════════════════════════════════════════════
export function SensorScene() {
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
        const final_val = clamp(q + n, 0, 1);

        let r: number, g: number, b: number;
        if (bayerOn) {
          const isEvenRow = row % 2 === 0, isEvenCol = col % 2 === 0;
          if (isEvenRow && isEvenCol) { r = 0; g = final_val; b = 0; }
          else if (isEvenRow && !isEvenCol) { r = final_val; g = 0; b = 0; }
          else if (!isEvenRow && isEvenCol) { r = 0; g = 0; b = final_val; }
          else { r = 0; g = final_val; b = 0; }
        } else {
          r = g = b = final_val;
        }

        ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
        ctx.fillRect(startX + col * cellSize, startY + row * cellSize, cellSize, cellSize);
      }
    }

    ctx.strokeStyle = "rgba(0,0,0,0.4)"; ctx.lineWidth = 0.5;
    for (let i = 0; i <= res; i++) {
      ctx.beginPath(); ctx.moveTo(startX + i * cellSize, startY); ctx.lineTo(startX + i * cellSize, startY + gridW); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(startX, startY + i * cellSize); ctx.lineTo(startX + gridW, startY + i * cellSize); ctx.stroke();
    }
    ctx.strokeStyle = "hsl(var(--primary) / 0.4)"; ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, gridW, gridW);

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
    const resize = () => { const r = canvas.parentElement?.getBoundingClientRect(); if (r) { canvas.width = r.width; canvas.height = 380; } };
    resize(); window.addEventListener("resize", resize);
    const loop = () => { tRef.current += 0.008; draw(); rafRef.current = requestAnimationFrame(loop); };
    loop();
    return () => { cancelAnimationFrame(rafRef.current); window.removeEventListener("resize", resize); };
  }, [draw]);

  return (
    <AnimationWrapper title="Sensor & Pixels" description="Continuous irradiance → discrete pixel array with Bayer filter" icon={Focus}>
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border bg-background" style={{ height: 380 }} />
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 px-1 mt-3">
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
      <Button variant={bayerOn ? "default" : "outline"} size="sm" className="text-xs mt-2" onClick={() => setBayerOn(!bayerOn)}>
        {bayerOn ? "Bayer RGGB" : "Monochrome"}
      </Button>
    </AnimationWrapper>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Default export: Tab-based viewer (kept for backwards compat)
// ═══════════════════════════════════════════════════════════════════════
const scenes = [
  { id: "projection3d", label: "3D → 2D Projection", icon: Move3D, description: "Watch how a 3D world point travels through space and lands on the camera's image plane." },
  { id: "pinhole", label: "Pinhole Model", icon: Camera, description: "Light rays pass through a single point — the center of projection." },
  { id: "perspective", label: "Perspective Projection", icon: Grid3X3, description: "3D world point → 2D image point via similar triangles." },
  { id: "intrinsic", label: "Intrinsic Matrix K", icon: Cpu, description: "Mapping from camera coordinates to pixel coordinates." },
  { id: "lens", label: "Lens & Depth of Field", icon: Aperture, description: "Real lenses blur out-of-focus points into circles of confusion." },
  { id: "sensor", label: "Sensor & Pixels", icon: Focus, description: "Continuous irradiance → discrete pixel array." },
];

const sceneComponents: Record<string, React.FC> = {
  projection3d: () => <Projection3DScene />,
  pinhole: () => <PinholeScene />,
  perspective: () => <PerspectiveScene />,
  intrinsic: () => <IntrinsicScene />,
  lens: () => <LensScene />,
  sensor: () => <SensorScene />,
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
