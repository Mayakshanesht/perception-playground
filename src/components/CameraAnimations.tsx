import { useRef, useEffect, useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Camera, Move3D, Focus, Grid3X3, Aperture, Cpu, Palette } from "lucide-react";
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
      { X: -120, Y: -80, Z: 300, color: "hsla(0, 85%, 60%, 1)", label: "P₁" },
      { X: 60, Y: -40, Z: 250 + 50 * Math.sin(t * 0.7), color: "hsla(140, 70%, 55%, 1)", label: "P₂" },
      { X: -30 + 40 * Math.cos(t * 0.5), Y: 60, Z: 350, color: "hsla(220, 80%, 60%, 1)", label: "P₃" },
      { X: 80 * Math.sin(t * 0.3), Y: -60 + 30 * Math.sin(t * 0.6), Z: 200, color: "hsla(45, 90%, 55%, 1)", label: "P₄" },
    ];

    // Optical axis
    ctx.strokeStyle = "hsla(220, 70%, 55%, 0.15)";
    ctx.lineWidth = 1; ctx.setLineDash([8, 6]);
    ctx.beginPath(); ctx.moveTo(30, cy); ctx.lineTo(W - 30, cy); ctx.stroke();
    ctx.setLineDash([]);

    // Camera body
    ctx.fillStyle = "hsla(220, 70%, 55%, 0.08)";
    ctx.strokeStyle = "hsla(220, 70%, 55%, 0.5)";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.roundRect(camX - 20, camY - 16, 40, 32, 4); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.arc(camX, camY, 9, 0, Math.PI * 2);
    ctx.fillStyle = "hsla(220, 70%, 55%, 0.2)"; ctx.fill(); ctx.stroke();
    labelText(ctx, camX, camY + 30, "Camera O", "hsla(220, 70%, 55%, 1)", "center");

    // Image plane
    ctx.strokeStyle = "hsla(270, 70%, 60%, 1)";
    ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(imgPlaneX, cy - H * 0.38); ctx.lineTo(imgPlaneX, cy + H * 0.38); ctx.stroke();
    labelText(ctx, imgPlaneX + 6, cy - H * 0.36, "Image\nPlane", "hsla(270, 70%, 60%, 1)");

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
    ctx.fillStyle = "hsla(0, 0%, 10%, 0.85)";
    ctx.fillRect(W - 220, 10, 210, 50);
    ctx.strokeStyle = "hsla(0, 0%, 30%, 0.5)"; ctx.lineWidth = 1;
    ctx.strokeRect(W - 220, 10, 210, 50);
    ctx.font = "12px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsla(220, 70%, 65%, 1)"; ctx.textAlign = "left";
    ctx.fillText("x' = f · X / Z", W - 210, 32);
    ctx.fillText("y' = f · Y / Z", W - 210, 50);

    ctx.strokeStyle = "hsla(0, 0%, 60%, 0.4)";
    ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(camX, cy + H * 0.35); ctx.lineTo(imgPlaneX, cy + H * 0.35); ctx.stroke();
    ctx.setLineDash([]);
    labelText(ctx, (camX + imgPlaneX) / 2, cy + H * 0.35 + 14, `f = ${focalLength}`, "hsla(0, 0%, 60%, 1)", "center");
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

    ctx.strokeStyle = "hsla(220, 70%, 55%, 0.15)";
    ctx.lineWidth = 1; ctx.setLineDash([8, 6]);
    ctx.beginPath(); ctx.moveTo(30, cy); ctx.lineTo(W - 30, cy); ctx.stroke();
    ctx.setLineDash([]);

    drawArrow(ctx, objX, cy, objX, cy - objHeight + objAnim, "hsla(20, 90%, 55%, 1)", 3);

    ctx.strokeStyle = "hsla(270, 70%, 60%, 1)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(imgX, cy - H * 0.4); ctx.lineTo(imgX, cy + H * 0.4); ctx.stroke();

    ctx.strokeStyle = "hsla(220, 70%, 55%, 0.5)"; ctx.lineWidth = 3;
    const gap = 12;
    ctx.beginPath(); ctx.moveTo(cxp, cy - H * 0.4); ctx.lineTo(cxp, cy - gap); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cxp, cy + gap); ctx.lineTo(cxp, cy + H * 0.4); ctx.stroke();
    ctx.fillStyle = "hsla(220, 70%, 55%, 0.15)";
    ctx.fillRect(cxp - 3, cy - H * 0.4, 6, H * 0.8);
    ctx.fillStyle = "hsla(220, 70%, 55%, 1)";
    ctx.beginPath(); ctx.arc(cxp, cy, 4, 0, Math.PI * 2); ctx.fill();

    const nRays = showAllRays ? 7 : 2;
    for (let i = 0; i < nRays; i++) {
      const t = nRays <= 1 ? 0.5 : i / (nRays - 1);
      const srcY = cy - (objHeight - objAnim) + (objHeight - objAnim) * 2 * t;
      const alpha = showAllRays ? 0.25 : 0.7;
      drawRay(ctx, objX, srcY, cxp, cy, `hsla(220, 70%, 55%, ${alpha})`, 1.2);
      const slope = (cy - srcY) / (cxp - objX);
      const dstY = cy + slope * (imgX - cxp);
      drawRay(ctx, cxp, cy, imgX, dstY, `hsla(220, 70%, 55%, ${alpha})`, 1.2);
      if (!showAllRays || i === 0 || i === nRays - 1) {
        ctx.fillStyle = "hsla(270, 70%, 60%, 1)";
        ctx.beginPath(); ctx.arc(imgX, dstY, 3, 0, Math.PI * 2); ctx.fill();
      }
    }

    drawArrow(ctx, imgX, cy, imgX, cy + Math.abs(imgH) + objAnim * (f / objDist), "hsla(270, 70%, 60%, 1)", 2);

    labelText(ctx, objX, cy + 20, "Object", "hsla(20, 90%, 55%, 1)");
    labelText(ctx, cxp, cy - H * 0.38, "Pinhole", "hsla(220, 70%, 55%, 1)");
    labelText(ctx, imgX + 8, cy - H * 0.38, "Image\nPlane", "hsla(270, 70%, 60%, 1)");

    ctx.strokeStyle = "hsla(0, 0%, 60%, 0.15)"; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(objX, cy + H * 0.35); ctx.lineTo(cxp, cy + H * 0.35); ctx.stroke();
    ctx.setLineDash([]);
    labelText(ctx, (objX + cxp) / 2, cy + H * 0.35 + 15, `d = ${objDist}`, "hsla(0, 0%, 60%, 1)", "center");
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(cxp, cy + H * 0.35); ctx.lineTo(imgX, cy + H * 0.35); ctx.stroke();
    ctx.setLineDash([]);
    labelText(ctx, (cxp + imgX) / 2, cy + H * 0.35 + 15, `f = ${f}`, "hsla(0, 0%, 60%, 1)", "center");

    const m = (f / objDist).toFixed(2);
    labelText(ctx, W - 20, 20, `|m| = ${m}`, "hsla(220, 70%, 55%, 1)", "right");
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
// SCENE: Perspective Projection — Similar Triangles
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
    const splitX = W * 0.6;
    const margin = 40;

    // === LEFT: Side view showing similar triangles ===
    const camX = margin + 40;
    const camY = H * 0.5;
    const imgPlaneX = camX + f * 0.5;
    const animX = pointX + 15 * Math.sin(t * 0.4);
    const animY = pointY + 10 * Math.cos(t * 0.5);
    const Z = pointZ;
    const worldScreenX = camX + Z * 0.8;
    const worldScreenY = camY + animY * 0.6;
    const projY = f * animY / Z;
    const imgScreenY = camY + projY * 0.5;

    // Background
    ctx.fillStyle = "hsla(0, 0%, 50%, 0.04)";
    ctx.fillRect(0, 0, splitX - 10, H);
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsla(0, 0%, 60%, 0.4)";
    ctx.textAlign = "center";
    ctx.fillText("SIDE VIEW — Similar Triangles", (splitX - 10) / 2, 16);

    // Optical axis
    ctx.strokeStyle = "hsla(220, 70%, 55%, 0.12)";
    ctx.lineWidth = 1; ctx.setLineDash([6, 6]);
    ctx.beginPath(); ctx.moveTo(camX - 20, camY); ctx.lineTo(splitX - 20, camY); ctx.stroke();
    ctx.setLineDash([]);

    // Camera center
    ctx.fillStyle = "hsla(220, 70%, 55%, 1)";
    ctx.beginPath(); ctx.arc(camX, camY, 6, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "hsla(220, 70%, 55%, 0.3)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(camX, camY, 12, 0, Math.PI * 2); ctx.stroke();
    labelText(ctx, camX, camY + 22, "O (0,0,0)", "hsla(220, 70%, 55%, 1)", "center");

    // Image plane line
    ctx.strokeStyle = "hsla(270, 70%, 60%, 1)"; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(imgPlaneX, camY - H * 0.4); ctx.lineTo(imgPlaneX, camY + H * 0.4); ctx.stroke();
    labelText(ctx, imgPlaneX, camY - H * 0.4 - 6, "z = f", "hsla(270, 70%, 60%, 1)", "center");

    // Z axis
    ctx.strokeStyle = "hsla(0, 0%, 60%, 0.3)"; ctx.lineWidth = 1;
    drawArrow(ctx, camX + 20, camY + H * 0.38, splitX - 30, camY + H * 0.38, "hsla(0, 0%, 60%, 0.4)", 1.5);
    labelText(ctx, splitX - 50, camY + H * 0.38 + 14, "Z (depth)", "hsla(0, 0%, 60%, 0.5)", "right");

    // World point
    const ptColor = "hsla(0, 85%, 60%, 1)";
    ctx.fillStyle = ptColor;
    ctx.beginPath(); ctx.arc(worldScreenX, worldScreenY, 8, 0, Math.PI * 2); ctx.fill();
    const glow = ctx.createRadialGradient(worldScreenX, worldScreenY, 0, worldScreenX, worldScreenY, 20);
    glow.addColorStop(0, "hsla(0, 85%, 60%, 0.3)"); glow.addColorStop(1, "transparent");
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(worldScreenX, worldScreenY, 20, 0, Math.PI * 2); ctx.fill();
    labelText(ctx, worldScreenX + 12, worldScreenY - 12, `P(${Math.round(animX)}, ${Math.round(animY)}, ${Z})`, ptColor);

    // Ray from O through P
    ctx.globalAlpha = 0.3;
    drawRay(ctx, camX, camY, worldScreenX, worldScreenY, ptColor, 1.5);
    ctx.globalAlpha = 1;
    drawRay(ctx, camX, camY, imgPlaneX, imgScreenY, "hsla(270, 70%, 60%, 1)", 2);

    // Projected point
    ctx.fillStyle = "hsla(270, 70%, 60%, 1)";
    ctx.beginPath(); ctx.arc(imgPlaneX, imgScreenY, 5, 0, Math.PI * 2); ctx.fill();
    const projLabel = `p(${(f * animX / Z).toFixed(1)}, ${(f * animY / Z).toFixed(1)})`;
    labelText(ctx, imgPlaneX - 8, imgScreenY + 16, projLabel, "hsla(270, 70%, 60%, 1)", "right");

    // Similar triangles dashes
    ctx.strokeStyle = "hsla(220, 80%, 60%, 0.6)"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(imgPlaneX, camY); ctx.lineTo(imgPlaneX, imgScreenY); ctx.stroke();
    ctx.strokeStyle = "hsla(0, 85%, 60%, 0.4)";
    ctx.beginPath(); ctx.moveTo(worldScreenX, camY); ctx.lineTo(worldScreenX, worldScreenY); ctx.stroke();
    ctx.setLineDash([]);

    // f annotation
    const annY = camY + H * 0.28;
    drawArrow(ctx, camX, annY, imgPlaneX, annY, "hsla(220, 80%, 60%, 0.6)", 1.5);
    labelText(ctx, (camX + imgPlaneX) / 2, annY + 12, `f = ${f}`, "hsla(220, 80%, 60%, 1)", "center");

    // Z annotation
    drawArrow(ctx, camX, annY + 24, worldScreenX, annY + 24, "hsla(0, 85%, 60%, 0.5)", 1.5);
    labelText(ctx, (camX + worldScreenX) / 2, annY + 36, `Z = ${Z}`, "hsla(0, 85%, 60%, 0.5)", "center");

    // Y / Y' labels
    const dy = Math.abs(imgScreenY - camY);
    if (dy > 10) labelText(ctx, imgPlaneX - 14, (camY + imgScreenY) / 2, "y'", "hsla(270, 70%, 60%, 1)", "right");
    const dyW = Math.abs(worldScreenY - camY);
    if (dyW > 10) labelText(ctx, worldScreenX + 10, (camY + worldScreenY) / 2, "Y", ptColor);

    // === RIGHT: Image plane + formula ===
    const rightX = splitX + 10;
    const rightW = W - rightX - 10;

    ctx.fillStyle = "hsla(0, 0%, 50%, 0.02)";
    ctx.fillRect(rightX, 0, rightW, H);
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsla(0, 0%, 60%, 0.4)";
    ctx.textAlign = "center";
    ctx.fillText("IMAGE PLANE VIEW", rightX + rightW / 2, 16);

    const ipW = rightW * 0.7;
    const ipH = ipW * 0.75;
    const ipX = rightX + (rightW - ipW) / 2;
    const ipY = 30;

    ctx.strokeStyle = "hsla(270, 70%, 60%, 0.5)"; ctx.lineWidth = 2;
    ctx.strokeRect(ipX, ipY, ipW, ipH);
    ctx.fillStyle = "hsla(270, 70%, 60%, 0.03)";
    ctx.fillRect(ipX, ipY, ipW, ipH);

    // Grid
    ctx.strokeStyle = "hsla(0, 0%, 50%, 0.15)"; ctx.lineWidth = 0.5;
    for (let i = 1; i < 8; i++) {
      const gx = ipX + (ipW / 8) * i;
      ctx.beginPath(); ctx.moveTo(gx, ipY); ctx.lineTo(gx, ipY + ipH); ctx.stroke();
      if (i <= 6) {
        const gy = ipY + (ipH / 6) * i;
        ctx.beginPath(); ctx.moveTo(ipX, gy); ctx.lineTo(ipX + ipW, gy); ctx.stroke();
      }
    }

    // Principal point
    const ppX = ipX + ipW / 2;
    const ppY = ipY + ipH / 2;
    ctx.strokeStyle = "hsla(220, 70%, 55%, 0.2)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(ppX, ipY); ctx.lineTo(ppX, ipY + ipH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ipX, ppY); ctx.lineTo(ipX + ipW, ppY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "hsla(220, 70%, 55%, 0.3)";
    ctx.beginPath(); ctx.arc(ppX, ppY, 3, 0, Math.PI * 2); ctx.fill();

    // Projected point
    const projPxX = clamp(ppX + (f * animX / Z) * 0.4, ipX + 4, ipX + ipW - 4);
    const projPxY = clamp(ppY + (f * animY / Z) * 0.4, ipY + 4, ipY + ipH - 4);
    ctx.fillStyle = ptColor;
    ctx.beginPath(); ctx.arc(projPxX, projPxY, 6, 0, Math.PI * 2); ctx.fill();
    const glow2 = ctx.createRadialGradient(projPxX, projPxY, 0, projPxX, projPxY, 16);
    glow2.addColorStop(0, "hsla(0, 85%, 60%, 0.4)"); glow2.addColorStop(1, "transparent");
    ctx.fillStyle = glow2;
    ctx.beginPath(); ctx.arc(projPxX, projPxY, 16, 0, Math.PI * 2); ctx.fill();

    // Formula box
    const fBoxY = ipY + ipH + 20;
    ctx.fillStyle = "hsla(0, 0%, 8%, 0.9)";
    ctx.fillRect(rightX + 10, fBoxY, rightW - 20, 90);
    ctx.strokeStyle = "hsla(0, 0%, 30%, 0.5)"; ctx.lineWidth = 1;
    ctx.strokeRect(rightX + 10, fBoxY, rightW - 20, 90);

    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsla(0, 0%, 60%, 0.5)"; ctx.textAlign = "left";
    ctx.fillText("Similar Triangles:", rightX + 20, fBoxY + 18);
    ctx.fillStyle = "hsla(220, 70%, 65%, 1)";
    ctx.font = "13px 'JetBrains Mono', monospace";
    ctx.fillText("y'/f = Y/Z", rightX + 20, fBoxY + 38);
    ctx.fillText("y' = f·Y/Z", rightX + 20, fBoxY + 56);
    ctx.fillStyle = "hsla(0, 0%, 65%, 1)";
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillText(`= ${f}·${Math.round(animY)}/${Z}`, rightX + 20, fBoxY + 74);
    ctx.fillText(`= ${(f * animY / Z).toFixed(1)} px`, rightX + 120, fBoxY + 74);

    // Triangle callout
    const triY = fBoxY + 105;
    ctx.fillStyle = "hsla(220, 80%, 60%, 0.08)";
    ctx.fillRect(rightX + 10, triY, rightW - 20, 36);
    ctx.strokeStyle = "hsla(220, 80%, 60%, 0.3)"; ctx.lineWidth = 1;
    ctx.strokeRect(rightX + 10, triY, rightW - 20, 36);
    ctx.fillStyle = "hsla(220, 80%, 60%, 1)";
    ctx.font = "10px 'JetBrains Mono', monospace"; ctx.textAlign = "center";
    ctx.fillText("△(O, f, y') ~ △(O, Z, Y)", rightX + rightW / 2, triY + 14);
    ctx.fillStyle = "hsla(0, 0%, 65%, 1)";
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
// SCENE: Intrinsic Matrix K
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

    // Three panels
    const panelW = (W - 30) / 3;
    const panels = [
      { x: 5, label: "NORMALIZED COORDS" },
      { x: panelW + 15, label: "INTRINSIC MATRIX K" },
      { x: 2 * panelW + 25, label: "PIXEL COORDS" },
    ];

    panels.forEach((p, i) => {
      ctx.fillStyle = i === 1 ? "hsla(270, 70%, 60%, 0.04)" : "hsla(0, 0%, 50%, 0.03)";
      ctx.fillRect(p.x, 0, panelW, H);
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillStyle = "hsla(0, 0%, 60%, 0.4)";
      ctx.textAlign = "center";
      ctx.fillText(p.label, p.x + panelW / 2, 14);
    });

    // Flow arrows
    const arrowY = H / 2;
    ctx.fillStyle = "hsla(220, 70%, 55%, 0.6)";
    for (const ax of [panelW + 8, 2 * panelW + 18]) {
      ctx.beginPath();
      ctx.moveTo(ax, arrowY - 8); ctx.lineTo(ax + 8, arrowY); ctx.lineTo(ax, arrowY + 8);
      ctx.closePath(); ctx.fill();
    }

    // === Panel 1: Normalized coordinate plane ===
    const normCx = panels[0].x + panelW / 2;
    const normCy = H * 0.48;
    const normScale = Math.min(panelW * 0.35, H * 0.3);

    ctx.strokeStyle = "hsla(0, 0%, 50%, 0.15)"; ctx.lineWidth = 0.5;
    for (let i = -3; i <= 3; i++) {
      const gx = normCx + (i / 3) * normScale;
      ctx.beginPath(); ctx.moveTo(gx, normCy - normScale); ctx.lineTo(gx, normCy + normScale); ctx.stroke();
      const gy = normCy + (i / 3) * normScale;
      ctx.beginPath(); ctx.moveTo(normCx - normScale, gy); ctx.lineTo(normCx + normScale, gy); ctx.stroke();
    }

    ctx.strokeStyle = "hsla(220, 70%, 55%, 0.5)"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(normCx - normScale, normCy); ctx.lineTo(normCx + normScale, normCy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(normCx, normCy - normScale); ctx.lineTo(normCx, normCy + normScale); ctx.stroke();
    labelText(ctx, normCx + normScale + 4, normCy - 4, "x_n", "hsla(220, 70%, 55%, 0.5)");
    labelText(ctx, normCx + 6, normCy - normScale - 4, "y_n", "hsla(220, 70%, 55%, 0.5)");

    const testPoints = [
      { xn: 0.4 * Math.cos(t * 0.3), yn: -0.3 * Math.sin(t * 0.4), color: "hsla(0, 85%, 60%, 1)", label: "P₁" },
      { xn: -0.5, yn: 0.5 + 0.2 * Math.sin(t * 0.5), color: "hsla(140, 70%, 55%, 1)", label: "P₂" },
      { xn: 0.6 + 0.1 * Math.cos(t * 0.6), yn: 0.1, color: "hsla(45, 90%, 55%, 1)", label: "P₃" },
    ];

    testPoints.forEach((pt) => {
      const sx = normCx + pt.xn * normScale;
      const sy = normCy + pt.yn * normScale;
      ctx.fillStyle = pt.color;
      ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI * 2); ctx.fill();
      labelText(ctx, sx + 8, sy - 4, `${pt.label}(${pt.xn.toFixed(2)}, ${pt.yn.toFixed(2)})`, pt.color);
    });

    labelText(ctx, normCx, normCy + normScale + 16, "(0,0) = optical axis", "hsla(0, 0%, 60%, 0.5)", "center");

    // === Panel 2: K matrix ===
    const kCx = panels[1].x + panelW / 2;
    const kTop = 28;

    ctx.fillStyle = "hsla(0, 0%, 8%, 0.9)";
    const matX = kCx - panelW * 0.42;
    const matW = panelW * 0.84;
    ctx.fillRect(matX, kTop, matW, 76);
    ctx.strokeStyle = "hsla(270, 70%, 60%, 0.5)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(matX + 8, kTop + 4); ctx.lineTo(matX + 2, kTop + 4);
    ctx.lineTo(matX + 2, kTop + 72); ctx.lineTo(matX + 8, kTop + 72); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(matX + matW - 8, kTop + 4); ctx.lineTo(matX + matW - 2, kTop + 4);
    ctx.lineTo(matX + matW - 2, kTop + 72); ctx.lineTo(matX + matW - 8, kTop + 72); ctx.stroke();

    ctx.font = "12px 'JetBrains Mono', monospace"; ctx.textAlign = "center";
    ctx.fillStyle = "hsla(220, 80%, 60%, 1)"; ctx.fillText(`${fx}`, kCx - panelW * 0.25, kTop + 22);
    ctx.fillStyle = "hsla(45, 90%, 55%, 1)"; ctx.fillText(`${skew}`, kCx, kTop + 22);
    ctx.fillStyle = "hsla(20, 90%, 55%, 1)"; ctx.fillText(`${cxOff}`, kCx + panelW * 0.25, kTop + 22);
    ctx.fillStyle = "hsla(0, 0%, 40%, 0.3)"; ctx.fillText("0", kCx - panelW * 0.25, kTop + 46);
    ctx.fillStyle = "hsla(220, 80%, 60%, 1)"; ctx.fillText(`${fy}`, kCx, kTop + 46);
    ctx.fillStyle = "hsla(20, 90%, 55%, 1)"; ctx.fillText(`${cyOff}`, kCx + panelW * 0.25, kTop + 46);
    ctx.fillStyle = "hsla(0, 0%, 40%, 0.3)";
    ctx.fillText("0", kCx - panelW * 0.25, kTop + 68);
    ctx.fillText("0", kCx, kTop + 68);
    ctx.fillText("1", kCx + panelW * 0.25, kTop + 68);

    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsla(220, 80%, 60%, 0.6)"; ctx.fillText("fx", kCx - panelW * 0.25, kTop + 88);
    ctx.fillStyle = "hsla(45, 90%, 55%, 0.6)"; ctx.fillText("γ", kCx, kTop + 88);
    ctx.fillStyle = "hsla(20, 90%, 55%, 0.6)"; ctx.fillText("cx", kCx + panelW * 0.25, kTop + 88);
    ctx.fillStyle = "hsla(220, 80%, 60%, 0.6)"; ctx.fillText("fy", kCx, kTop + 98);
    ctx.fillStyle = "hsla(20, 90%, 55%, 0.6)"; ctx.fillText("cy", kCx + panelW * 0.25, kTop + 98);

    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsla(220, 70%, 65%, 1)"; ctx.textAlign = "center";
    ctx.fillText("u = fx·xn + γ·yn + cx", kCx, kTop + 128);
    ctx.fillText("v = fy·yn + cy", kCx, kTop + 145);

    const expY = kTop + 170;
    const params = [
      { label: "fx, fy", desc: "focal lengths (scale)", color: "hsla(220, 80%, 60%, 1)" },
      { label: "cx, cy", desc: "principal point (shift)", color: "hsla(20, 90%, 55%, 1)" },
      { label: "γ", desc: "skew (shear)", color: "hsla(45, 90%, 55%, 1)" },
    ];
    params.forEach((p, i) => {
      ctx.fillStyle = p.color;
      ctx.beginPath(); ctx.arc(panels[1].x + 16, expY + i * 18, 4, 0, Math.PI * 2); ctx.fill();
      ctx.font = "10px 'JetBrains Mono', monospace"; ctx.textAlign = "left";
      ctx.fillText(`${p.label}: ${p.desc}`, panels[1].x + 26, expY + i * 18 + 4);
    });

    // === Panel 3: Pixel coordinates ===
    const pixCx = panels[2].x + panelW / 2;
    const pixCy = H * 0.48;
    const imgW = Math.min(panelW * 0.8, H * 0.7);
    const imgH = imgW * 0.75;
    const imgLeft = pixCx - imgW / 2;
    const imgTop = pixCy - imgH / 2;

    ctx.strokeStyle = "hsla(220, 70%, 55%, 0.3)"; ctx.lineWidth = 1.5;
    ctx.strokeRect(imgLeft, imgTop, imgW, imgH);
    ctx.fillStyle = "hsla(220, 70%, 55%, 0.02)";
    ctx.fillRect(imgLeft, imgTop, imgW, imgH);

    ctx.strokeStyle = "hsla(0, 0%, 50%, 0.1)"; ctx.lineWidth = 0.5;
    for (let i = 1; i < 10; i++) {
      const gx = imgLeft + (imgW / 10) * i;
      ctx.beginPath(); ctx.moveTo(gx, imgTop); ctx.lineTo(gx, imgTop + imgH); ctx.stroke();
    }
    for (let i = 1; i < 8; i++) {
      const gy = imgTop + (imgH / 8) * i;
      ctx.beginPath(); ctx.moveTo(imgLeft, gy); ctx.lineTo(imgLeft + imgW, gy); ctx.stroke();
    }

    ctx.fillStyle = "hsla(0, 0%, 60%, 0.4)";
    ctx.font = "9px 'JetBrains Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText("(0,0)", imgLeft + 2, imgTop - 3);
    ctx.fillText(`(${640},${480})`, imgLeft + imgW - 50, imgTop + imgH + 12);

    // Principal point
    const ppScreenX = imgLeft + (cxOff / 640) * imgW;
    const ppScreenY = imgTop + (cyOff / 480) * imgH;
    ctx.strokeStyle = "hsla(20, 90%, 55%, 0.3)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(ppScreenX, imgTop); ctx.lineTo(ppScreenX, imgTop + imgH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(imgLeft, ppScreenY); ctx.lineTo(imgLeft + imgW, ppScreenY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "hsla(20, 90%, 55%, 1)";
    ctx.beginPath(); ctx.arc(ppScreenX, ppScreenY, 4, 0, Math.PI * 2); ctx.fill();
    labelText(ctx, ppScreenX + 6, ppScreenY - 8, `(cx,cy)`, "hsla(20, 90%, 55%, 1)");

    // Project test points
    testPoints.forEach((pt) => {
      const u = fx * pt.xn + skew * pt.yn + cxOff;
      const v = fy * pt.yn + cyOff;
      const sx = clamp(imgLeft + (u / 640) * imgW, imgLeft + 3, imgLeft + imgW - 3);
      const sy = clamp(imgTop + (v / 480) * imgH, imgTop + 3, imgTop + imgH - 3);
      ctx.fillStyle = pt.color;
      ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2); ctx.fill();
      // Glow - extract hsla values for transparency
      const glowColor = pt.color.replace(", 1)", ", 0.3)");
      const gl = ctx.createRadialGradient(sx, sy, 0, sx, sy, 14);
      gl.addColorStop(0, glowColor); gl.addColorStop(1, "transparent");
      ctx.fillStyle = gl;
      ctx.beginPath(); ctx.arc(sx, sy, 14, 0, Math.PI * 2); ctx.fill();
      labelText(ctx, sx + 8, sy - 4, `${pt.label}(${Math.round(u)}, ${Math.round(v)})`, pt.color);
    });

    drawArrow(ctx, imgLeft, imgTop + imgH + 6, imgLeft + imgW, imgTop + imgH + 6, "hsla(0, 0%, 50%, 0.3)", 1);
    labelText(ctx, imgLeft + imgW - 10, imgTop + imgH + 18, "u", "hsla(0, 0%, 60%, 0.4)", "right");
    drawArrow(ctx, imgLeft - 6, imgTop, imgLeft - 6, imgTop + imgH, "hsla(0, 0%, 50%, 0.3)", 1);
    labelText(ctx, imgLeft - 16, imgTop + imgH - 10, "v", "hsla(0, 0%, 60%, 0.4)");

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

    ctx.strokeStyle = "hsla(220, 70%, 55%, 0.1)"; ctx.lineWidth = 1; ctx.setLineDash([8, 6]);
    ctx.beginPath(); ctx.moveTo(20, cy); ctx.lineTo(W - 20, cy); ctx.stroke();
    ctx.setLineDash([]);

    ctx.save();
    ctx.strokeStyle = "hsla(220, 70%, 55%, 0.6)"; ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(lensX, cy - halfAper);
    ctx.bezierCurveTo(lensX + 30, cy - halfAper / 2, lensX + 30, cy + halfAper / 2, lensX, cy + halfAper);
    ctx.bezierCurveTo(lensX - 30, cy + halfAper / 2, lensX - 30, cy - halfAper / 2, lensX, cy - halfAper);
    ctx.fillStyle = "hsla(220, 70%, 55%, 0.05)"; ctx.fill(); ctx.stroke();
    ctx.restore();
    labelText(ctx, lensX + 35, cy - halfAper + 10, "Lens", "hsla(220, 70%, 55%, 1)");

    const di_focus = thinLensImageDist(fmm * scale, focusDist);
    const sensorX = lensX + di_focus;
    ctx.strokeStyle = "hsla(270, 70%, 60%, 1)"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(sensorX, cy - H * 0.35); ctx.lineTo(sensorX, cy + H * 0.35); ctx.stroke();
    labelText(ctx, sensorX + 6, cy - H * 0.33, "Sensor", "hsla(270, 70%, 60%, 1)");

    const objFocusX = lensX - focusDist;
    if (objFocusX > 20) {
      drawArrow(ctx, objFocusX, cy, objFocusX, cy - 50, "hsla(220, 70%, 55%, 1)", 2.5);
      labelText(ctx, objFocusX, cy + 15, "In focus", "hsla(220, 70%, 55%, 1)");
      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = "hsla(220, 70%, 55%, 1)"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(objFocusX, cy - 50); ctx.lineTo(lensX, cy - halfAper); ctx.lineTo(sensorX, cy + 50); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(objFocusX, cy - 50); ctx.lineTo(lensX, cy + halfAper); ctx.lineTo(sensorX, cy + 50); ctx.stroke();
      ctx.globalAlpha = 0.8;
      ctx.strokeStyle = "hsla(220, 70%, 55%, 1)"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(objFocusX, cy - 50); ctx.lineTo(sensorX, cy + 50); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = "hsla(220, 70%, 55%, 1)";
      ctx.beginPath(); ctx.arc(sensorX, cy + 50, 3, 0, Math.PI * 2); ctx.fill();
    }

    const objBlurX = lensX - objDist2;
    if (objBlurX > 20 && objBlurX < lensX - 20) {
      drawArrow(ctx, objBlurX, cy, objBlurX, cy - 60, "hsla(20, 90%, 55%, 1)", 2.5);
      labelText(ctx, objBlurX, cy + 15, "Out of\nfocus", "hsla(20, 90%, 55%, 1)");
      const di_blur = thinLensImageDist(fmm * scale, objDist2);
      const sharpPtX = lensX + di_blur;
      const CoC = halfAper * Math.abs(sensorX - sharpPtX) / Math.max(sharpPtX, 1);
      ctx.globalAlpha = 0.4;
      ctx.strokeStyle = "hsla(20, 90%, 55%, 1)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(objBlurX, cy - 60); ctx.lineTo(lensX, cy - halfAper); ctx.lineTo(sensorX, cy + 60 * (di_focus / di_blur)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(objBlurX, cy - 60); ctx.lineTo(lensX, cy + halfAper); ctx.lineTo(sensorX, cy + 60 * (di_focus / di_blur)); ctx.stroke();
      ctx.globalAlpha = 1;
      const cocR = clamp(CoC * 0.5, 1, 25);
      const imgYblur = cy + 60 * (di_focus / di_blur);
      ctx.strokeStyle = "hsla(20, 90%, 55%, 1)"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sensorX, imgYblur, cocR, 0, Math.PI * 2); ctx.stroke();
      labelText(ctx, sensorX + cocR + 4, imgYblur, `CoC ≈ ${cocR.toFixed(1)}px`, "hsla(20, 90%, 55%, 1)");
    }

    labelText(ctx, W - 10, 15, `f/${aperture.toFixed(1)}  f=${fmm}mm`, "hsla(0, 0%, 60%, 1)", "right");
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
// SCENE: Sensor & Pixels — FIXED Bayer RGGB pattern + demosaic preview
// ═══════════════════════════════════════════════════════════════════════
export function SensorScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const tRef = useRef(0);
  const [res, setRes] = useState(16);
  const [bits, setBits] = useState(8);
  const [noise, setNoise] = useState(5);
  const [showMode, setShowMode] = useState<"bayer" | "mono" | "demosaic">("bayer");

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const levels = Math.pow(2, bits);
    const noiseF = noise / 100;
    const t = tRef.current;

    // Two panels: raw sensor | info
    const gridSize = Math.min(W * 0.5, H * 0.85);
    const cellSize = gridSize / res;
    const startX = W * 0.04;
    const startY = (H - gridSize) / 2;

    // Store raw values for demosaic
    const rawR: number[][] = [];
    const rawG: number[][] = [];
    const rawB: number[][] = [];

    for (let row = 0; row < res; row++) {
      rawR[row] = []; rawG[row] = []; rawB[row] = [];
      for (let col = 0; col < res; col++) {
        const nx = col / res, ny = row / res;
        // Generate a smooth color scene
        const baseR = 0.5 + 0.4 * Math.sin((nx + t * 0.3) * Math.PI * 2.5);
        const baseG = 0.5 + 0.4 * Math.cos((ny + t * 0.2) * Math.PI * 3);
        const baseB = 0.5 + 0.4 * Math.sin((nx + ny + t * 0.4) * Math.PI * 2);

        // Quantize
        const qR = Math.floor(baseR * (levels - 1)) / (levels - 1);
        const qG = Math.floor(baseG * (levels - 1)) / (levels - 1);
        const qB = Math.floor(baseB * (levels - 1)) / (levels - 1);

        const nv = () => (Math.random() - 0.5) * noiseF;

        rawR[row][col] = clamp(qR + nv(), 0, 1);
        rawG[row][col] = clamp(qG + nv(), 0, 1);
        rawB[row][col] = clamp(qB + nv(), 0, 1);

        let r: number, g: number, b: number;

        if (showMode === "bayer") {
          // Correct RGGB Bayer pattern:
          // Even row: R G R G ...
          // Odd row:  G B G B ...
          const isEvenRow = row % 2 === 0;
          const isEvenCol = col % 2 === 0;
          if (isEvenRow && isEvenCol) { r = rawR[row][col]; g = 0; b = 0; }      // Red
          else if (isEvenRow && !isEvenCol) { r = 0; g = rawG[row][col]; b = 0; } // Green
          else if (!isEvenRow && isEvenCol) { r = 0; g = rawG[row][col]; b = 0; } // Green
          else { r = 0; g = 0; b = rawB[row][col]; }                              // Blue
        } else if (showMode === "demosaic") {
          // Simple bilinear demosaic approximation
          r = rawR[row][col];
          g = rawG[row][col];
          b = rawB[row][col];
        } else {
          // Monochrome
          const lum = 0.299 * rawR[row][col] + 0.587 * rawG[row][col] + 0.114 * rawB[row][col];
          r = g = b = lum;
        }

        ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
        ctx.fillRect(startX + col * cellSize, startY + row * cellSize, cellSize + 0.5, cellSize + 0.5);
      }
    }

    // Grid lines
    if (res <= 24) {
      ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 0.5;
      for (let i = 0; i <= res; i++) {
        ctx.beginPath(); ctx.moveTo(startX + i * cellSize, startY); ctx.lineTo(startX + i * cellSize, startY + gridSize); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(startX, startY + i * cellSize); ctx.lineTo(startX + gridSize, startY + i * cellSize); ctx.stroke();
      }
    }
    ctx.strokeStyle = "hsla(220, 70%, 55%, 0.4)"; ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, gridSize, gridSize);

    // Bayer pattern legend (top-right of grid)
    if (showMode === "bayer" && res <= 20) {
      const lgX = startX + gridSize + 16;
      const lgY = startY;
      ctx.font = "10px 'JetBrains Mono', monospace"; ctx.textAlign = "left";
      ctx.fillStyle = "hsla(0, 0%, 80%, 0.6)";
      ctx.fillText("Bayer RGGB Pattern:", lgX, lgY + 12);

      const cs = 18;
      // Row labels
      const patColors = [
        [{ c: "hsla(0, 80%, 50%, 1)", l: "R" }, { c: "hsla(120, 60%, 45%, 1)", l: "G" }],
        [{ c: "hsla(120, 60%, 45%, 1)", l: "G" }, { c: "hsla(220, 80%, 55%, 1)", l: "B" }],
      ];
      patColors.forEach((row, ri) => {
        row.forEach((cell, ci) => {
          ctx.fillStyle = cell.c;
          ctx.fillRect(lgX + ci * cs, lgY + 20 + ri * cs, cs - 2, cs - 2);
          ctx.fillStyle = "white"; ctx.textAlign = "center";
          ctx.fillText(cell.l, lgX + ci * cs + cs / 2 - 1, lgY + 20 + ri * cs + 13);
        });
      });
      ctx.textAlign = "left";
      ctx.fillStyle = "hsla(0, 0%, 60%, 0.5)";
      ctx.fillText("2× green for human", lgX, lgY + 70);
      ctx.fillText("luminance sensitivity", lgX, lgY + 84);
    }

    // Info panel
    const infoX = startX + gridSize + 16;
    const infoY = startY + (showMode === "bayer" && res <= 20 ? 110 : 10);
    ctx.fillStyle = "hsla(0, 0%, 5%, 0.8)";
    ctx.fillRect(infoX - 5, infoY - 5, 200, 110);
    ctx.strokeStyle = "hsla(0, 0%, 30%, 0.3)"; ctx.lineWidth = 1;
    ctx.strokeRect(infoX - 5, infoY - 5, 200, 110);
    ctx.font = "11px 'JetBrains Mono', monospace"; ctx.textAlign = "left";
    const vals = [
      `Resolution:  ${res}×${res}`,
      `Bit depth:   ${bits}  (${levels} levels)`,
      `Noise:       ${noise}%`,
      `Mode:        ${showMode === "bayer" ? "Bayer RGGB" : showMode === "demosaic" ? "Demosaiced" : "Mono"}`,
      `Total px:    ${res * res}`,
      `Data size:   ${(res * res * bits / 8).toFixed(0)} bytes`,
    ];
    vals.forEach((v, i) => {
      ctx.fillStyle = i === 0 ? "hsla(220, 70%, 65%, 1)" : "hsla(0, 0%, 65%, 1)";
      ctx.fillText(v, infoX, infoY + 16 * i + 12);
    });

    // Bottom label
    labelText(ctx, W / 2, H - 8, `${res}×${res} = ${res * res} pixels  ·  ${bits}-bit  ·  ${levels} levels`, "hsla(0, 0%, 60%, 1)", "center");
  }, [res, bits, noise, showMode]);

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
    <AnimationWrapper title="Sensor & Pixel Digitization" description="Continuous light → discrete pixel array with Bayer color filter" icon={Focus}>
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
      <div className="flex gap-2 mt-2">
        {(["bayer", "demosaic", "mono"] as const).map(mode => (
          <Button key={mode} variant={showMode === mode ? "default" : "outline"} size="sm" className="text-xs" onClick={() => setShowMode(mode)}>
            {mode === "bayer" ? "Bayer RGGB" : mode === "demosaic" ? "Demosaiced" : "Monochrome"}
          </Button>
        ))}
      </div>
    </AnimationWrapper>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// SCENE: Color Spaces & Image Manipulation
// ═══════════════════════════════════════════════════════════════════════
export function ColorSpaceScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const tRef = useRef(0);
  const [gamma, setGamma] = useState(2.2);
  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(1.0);
  const [showSpace, setShowSpace] = useState<"rgb" | "hsv" | "gamma">("rgb");

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const t = tRef.current;

    if (showSpace === "rgb") {
      // RGB color cube slices
      const sliceSize = Math.min(W * 0.28, H * 0.7);
      const gap = 20;
      const totalW = sliceSize * 3 + gap * 2;
      const sx = (W - totalW) / 2;
      const sy = 40;

      const channels = [
        { label: "Red Channel", ch: 0, accent: "hsla(0, 80%, 55%, 1)" },
        { label: "Green Channel", ch: 1, accent: "hsla(120, 60%, 50%, 1)" },
        { label: "Blue Channel", ch: 2, accent: "hsla(220, 80%, 60%, 1)" },
      ];

      channels.forEach((channel, ci) => {
        const cx = sx + ci * (sliceSize + gap);
        const step = 8;
        for (let row = 0; row < sliceSize; row += step) {
          for (let col = 0; col < sliceSize; col += step) {
            const nx = col / sliceSize;
            const ny = row / sliceSize;
            // Generate scene color
            const r = 0.5 + 0.4 * Math.sin((nx + t * 0.2) * Math.PI * 2);
            const g = 0.5 + 0.4 * Math.cos((ny + t * 0.15) * Math.PI * 2.5);
            const b = 0.5 + 0.4 * Math.sin((nx + ny + t * 0.25) * Math.PI * 1.5);

            const vals = [r, g, b];
            const v = clamp(vals[channel.ch], 0, 1);

            if (channel.ch === 0) ctx.fillStyle = `rgb(${Math.round(v * 255)}, 0, 0)`;
            else if (channel.ch === 1) ctx.fillStyle = `rgb(0, ${Math.round(v * 255)}, 0)`;
            else ctx.fillStyle = `rgb(0, 0, ${Math.round(v * 255)})`;

            ctx.fillRect(cx + col, sy + row, step, step);
          }
        }

        ctx.strokeStyle = channel.accent; ctx.lineWidth = 2;
        ctx.strokeRect(cx, sy, sliceSize, sliceSize);
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.fillStyle = channel.accent; ctx.textAlign = "center";
        ctx.fillText(channel.label, cx + sliceSize / 2, sy - 6);
      });

      // Combined color
      const combY = sy + sliceSize + 20;
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.fillStyle = "hsla(0, 0%, 70%, 1)"; ctx.textAlign = "center";
      ctx.fillText("RGB: Each pixel stores 3 channel values (0-255) → 16.7M possible colors", W / 2, combY);
      ctx.fillText("I(x,y) = [R(x,y), G(x,y), B(x,y)]", W / 2, combY + 18);

    } else if (showSpace === "hsv") {
      // HSV color wheel
      const centerX = W * 0.35, centerY = H * 0.45;
      const radius = Math.min(W * 0.22, H * 0.35);

      // Draw hue wheel
      for (let angle = 0; angle < 360; angle += 2) {
        const rad = (angle * Math.PI) / 180;
        const nextRad = ((angle + 2) * Math.PI) / 180;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, rad, nextRad);
        ctx.closePath();
        ctx.fillStyle = `hsla(${angle}, 100%, 50%, 1)`;
        ctx.fill();
      }

      // Center fade to white
      const grad = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
      grad.addColorStop(0, "rgba(255, 255, 255, 1)");
      grad.addColorStop(1, "rgba(255, 255, 255, 0)");
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.arc(centerX, centerY, radius, 0, Math.PI * 2); ctx.fill();

      // Animated marker
      const markerAngle = t * 0.5;
      const markerR = radius * 0.7;
      const mx = centerX + markerR * Math.cos(markerAngle);
      const my = centerY + markerR * Math.sin(markerAngle);
      ctx.strokeStyle = "white"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(mx, my, 6, 0, Math.PI * 2); ctx.stroke();

      const hue = ((markerAngle * 180) / Math.PI) % 360;
      labelText(ctx, centerX, centerY + radius + 20, `H=${Math.round(hue < 0 ? hue + 360 : hue)}° S=70% V=100%`, "hsla(0, 0%, 70%, 1)", "center");

      // Value bar on the right
      const barX = W * 0.65, barY = 40, barW = 30, barH = H - 80;
      for (let y = 0; y < barH; y++) {
        const v = 1 - y / barH;
        ctx.fillStyle = `hsla(${Math.round(hue < 0 ? hue + 360 : hue)}, 70%, ${v * 50}%, 1)`;
        ctx.fillRect(barX, barY + y, barW, 1);
      }
      ctx.strokeStyle = "hsla(0, 0%, 50%, 0.5)"; ctx.lineWidth = 1;
      ctx.strokeRect(barX, barY, barW, barH);
      labelText(ctx, barX + barW + 8, barY + 4, "V=1.0", "hsla(0, 0%, 60%, 1)");
      labelText(ctx, barX + barW + 8, barY + barH - 4, "V=0.0", "hsla(0, 0%, 60%, 1)");
      labelText(ctx, barX + barW / 2, barY - 8, "Value", "hsla(0, 0%, 60%, 1)", "center");

      // Explanation
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.fillStyle = "hsla(0, 0%, 70%, 1)"; ctx.textAlign = "left";
      ctx.fillText("HSV separates color (H,S)", W * 0.62 + 50, H * 0.5);
      ctx.fillText("from brightness (V)", W * 0.62 + 50, H * 0.5 + 16);
      ctx.fillText("→ easier for CV thresholding", W * 0.62 + 50, H * 0.5 + 36);

    } else {
      // Gamma correction curve
      const plotX = 60, plotY = 30;
      const plotW = W * 0.45, plotH = H - 80;

      // Plot background
      ctx.fillStyle = "hsla(0, 0%, 5%, 0.5)";
      ctx.fillRect(plotX, plotY, plotW, plotH);
      ctx.strokeStyle = "hsla(0, 0%, 30%, 0.5)"; ctx.lineWidth = 1;
      ctx.strokeRect(plotX, plotY, plotW, plotH);

      // Grid
      ctx.strokeStyle = "hsla(0, 0%, 30%, 0.2)"; ctx.lineWidth = 0.5;
      for (let i = 1; i < 4; i++) {
        const gx = plotX + (plotW / 4) * i;
        ctx.beginPath(); ctx.moveTo(gx, plotY); ctx.lineTo(gx, plotY + plotH); ctx.stroke();
        const gy = plotY + (plotH / 4) * i;
        ctx.beginPath(); ctx.moveTo(plotX, gy); ctx.lineTo(plotX + plotW, gy); ctx.stroke();
      }

      // Linear line
      ctx.strokeStyle = "hsla(0, 0%, 50%, 0.4)"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(plotX, plotY + plotH); ctx.lineTo(plotX + plotW, plotY); ctx.stroke();
      ctx.setLineDash([]);

      // Gamma curve
      ctx.strokeStyle = "hsla(45, 90%, 55%, 1)"; ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let i = 0; i <= 100; i++) {
        const x = i / 100;
        const y = Math.pow(x, 1 / gamma);
        const px = plotX + x * plotW;
        const py = plotY + plotH - y * plotH;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // Inverse gamma
      ctx.strokeStyle = "hsla(270, 70%, 60%, 1)"; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i <= 100; i++) {
        const x = i / 100;
        const y = Math.pow(x, gamma);
        const px = plotX + x * plotW;
        const py = plotY + plotH - y * plotH;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // Labels
      labelText(ctx, plotX + plotW / 2, plotY + plotH + 16, "Input (Linear)", "hsla(0, 0%, 60%, 1)", "center");
      ctx.save();
      ctx.translate(plotX - 14, plotY + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.fillStyle = "hsla(0, 0%, 60%, 1)"; ctx.textAlign = "center";
      ctx.fillText("Output", 0, 0);
      ctx.restore();

      // Legend
      const lgX = plotX + plotW + 30;
      ctx.fillStyle = "hsla(45, 90%, 55%, 1)";
      ctx.fillRect(lgX, plotY + 10, 16, 3);
      labelText(ctx, lgX + 22, plotY + 14, `γ = 1/${gamma.toFixed(1)} (encode)`, "hsla(45, 90%, 55%, 1)");

      ctx.fillStyle = "hsla(270, 70%, 60%, 1)";
      ctx.fillRect(lgX, plotY + 30, 16, 3);
      labelText(ctx, lgX + 22, plotY + 34, `γ = ${gamma.toFixed(1)} (decode)`, "hsla(270, 70%, 60%, 1)");

      ctx.fillStyle = "hsla(0, 0%, 50%, 0.4)";
      ctx.fillRect(lgX, plotY + 50, 16, 3);
      labelText(ctx, lgX + 22, plotY + 54, "linear (γ = 1)", "hsla(0, 0%, 50%, 0.6)");

      // Gradient bar showing effect
      const barY2 = plotY + 80;
      const barW = W - lgX - 20;
      ctx.font = "9px 'JetBrains Mono', monospace"; ctx.textAlign = "left";
      ctx.fillStyle = "hsla(0, 0%, 60%, 0.5)";
      ctx.fillText("Linear:", lgX, barY2);
      ctx.fillText("Gamma corrected:", lgX, barY2 + 30);

      for (let i = 0; i < barW; i++) {
        const v = i / barW;
        const lin = Math.round(v * 255);
        ctx.fillStyle = `rgb(${lin},${lin},${lin})`;
        ctx.fillRect(lgX + i, barY2 + 4, 1, 16);

        const gc = Math.round(Math.pow(v, 1 / gamma) * 255);
        ctx.fillStyle = `rgb(${gc},${gc},${gc})`;
        ctx.fillRect(lgX + i, barY2 + 34, 1, 16);
      }

      // Brightness/contrast preview
      const prevY = barY2 + 70;
      ctx.fillStyle = "hsla(0, 0%, 60%, 0.5)";
      ctx.fillText("Brightness/Contrast:", lgX, prevY);
      ctx.fillText(`B=${brightness}  C=${contrast.toFixed(1)}`, lgX, prevY + 14);
      ctx.fillText("I' = contrast × I + brightness", lgX, prevY + 30);

      for (let i = 0; i < barW; i++) {
        const v = i / barW;
        const adj = clamp(contrast * v + brightness / 255, 0, 1);
        const val = Math.round(adj * 255);
        ctx.fillStyle = `rgb(${val},${val},${val})`;
        ctx.fillRect(lgX + i, prevY + 38, 1, 16);
      }
    }
  }, [gamma, brightness, contrast, showSpace]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => { const r = canvas.parentElement?.getBoundingClientRect(); if (r) { canvas.width = r.width; canvas.height = 420; } };
    resize(); window.addEventListener("resize", resize);
    const loop = () => { tRef.current += 0.01; draw(); rafRef.current = requestAnimationFrame(loop); };
    loop();
    return () => { cancelAnimationFrame(rafRef.current); window.removeEventListener("resize", resize); };
  }, [draw]);

  return (
    <AnimationWrapper title="Color Spaces & Image Manipulation" description="Explore RGB channels, HSV separation, gamma correction, and pixel transforms" icon={Palette}>
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border bg-background" style={{ height: 420 }} />
      <div className="flex gap-2 mt-3 flex-wrap">
        {(["rgb", "hsv", "gamma"] as const).map(mode => (
          <Button key={mode} variant={showSpace === mode ? "default" : "outline"} size="sm" className="text-xs"
            onClick={() => setShowSpace(mode)}>
            {mode === "rgb" ? "RGB Channels" : mode === "hsv" ? "HSV Color Wheel" : "Gamma & Transforms"}
          </Button>
        ))}
      </div>
      {showSpace === "gamma" && (
        <div className="grid grid-cols-3 gap-3 px-1 mt-3">
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Gamma (γ)</span><span className="text-primary">{gamma.toFixed(1)}</span></div>
            <Slider min={0.5} max={4.0} step={0.1} value={[gamma]} onValueChange={([v]) => setGamma(v)} />
          </div>
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Brightness</span><span className="text-primary">{brightness}</span></div>
            <Slider min={-100} max={100} step={1} value={[brightness]} onValueChange={([v]) => setBrightness(v)} />
          </div>
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground font-mono"><span>Contrast</span><span className="text-primary">{contrast.toFixed(1)}</span></div>
            <Slider min={0.2} max={3.0} step={0.1} value={[contrast]} onValueChange={([v]) => setContrast(v)} />
          </div>
        </div>
      )}
    </AnimationWrapper>
  );
}
