import { useEffect, useState, useRef } from "react";
import { motion } from "framer-motion";
import { Upload, Play, Loader2, ImageIcon, AlertCircle } from "lucide-react";

interface PlaygroundProps {
  title: string;
  description: string;
  taskType: string;
  acceptVideo?: boolean;
  acceptImage?: boolean;
  modelName?: string;
  learningFocus?: string;
}

export default function Playground({
  title,
  description,
  taskType,
  acceptVideo = false,
  acceptImage = true,
  modelName,
  learningFocus,
}: PlaygroundProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.3);
  const [inferencePreview, setInferencePreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = [
    ...(acceptImage ? ["image/jpeg", "image/png", "image/webp"] : []),
    ...(acceptVideo ? ["video/mp4", "video/webm"] : []),
  ].join(",");
  const uploadLabel = acceptImage && acceptVideo ? "image or video" : acceptVideo ? "a video" : "an image";

  useEffect(() => {
    return () => {
      if (preview?.startsWith("blob:")) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const isImage = f.type.startsWith("image/");
    const isVideo = f.type.startsWith("video/");

    if ((isImage && !acceptImage) || (isVideo && !acceptVideo) || (!isImage && !isVideo)) {
      setError("Unsupported file type for this playground.");
      return;
    }

    const maxSizeBytes = isVideo ? 40 * 1024 * 1024 : 8 * 1024 * 1024;
    if (f.size > maxSizeBytes) {
      setError(`File too large. Max allowed is ${isVideo ? "40MB" : "8MB"}.`);
      return;
    }

    if (preview?.startsWith("blob:")) {
      URL.revokeObjectURL(preview);
    }

    setFile(f);
    setResult(null);
    setError(null);
    if (isImage) {
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result as string);
      reader.readAsDataURL(f);
    } else {
      setPreview(URL.createObjectURL(f));
    }
  };

  const runInference = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const isImageTask = ["object-detection", "image-segmentation", "pose-estimation", "depth-estimation"].includes(taskType);
      const isVideoInput = file.type.startsWith("video/");

      let payloadDataUrl: string;
      let requestMimeType = file.type;

      if (isImageTask && isVideoInput) {
        payloadDataUrl = await extractFirstFrameDataUrl(file);
        requestMimeType = "image/jpeg";
      } else {
        const reader = new FileReader();
        payloadDataUrl = await new Promise<string>((resolve, reject) => {
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = () => reject(new Error("Could not read the uploaded file."));
          reader.readAsDataURL(file);
        });
      }

      const base64 = payloadDataUrl.split(",")[1];
      setInferencePreview(payloadDataUrl);

      const endpoint = import.meta.env.VITE_INFERENCE_API_URL || "/api/hf-inference";
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: isImageTask ? base64 : (file.type.startsWith("image/") ? base64 : undefined),
          video: isImageTask ? undefined : (file.type.startsWith("video/") ? base64 : undefined),
          payloadBase64: base64,
          task: taskType,
          mimeType: requestMimeType,
          options: {
            threshold,

          },
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.error || "Inference request failed.");
      }
      if (data?.error) throw new Error(data.error);
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Inference failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-xl border border-border bg-card p-6">
      <div className="flex items-center gap-2 mb-2">
        <ImageIcon className="h-4 w-4 text-primary" />
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      </div>
      <p className="text-xs text-muted-foreground mb-4">{description}</p>
      {(modelName || learningFocus) && (
        <div className="mb-4 rounded-lg border border-border bg-muted/40 p-3">
          {modelName && <p className="text-[11px] text-foreground font-mono mb-1">Model: {modelName}</p>}
          {learningFocus && <p className="text-xs text-muted-foreground">{learningFocus}</p>}
        </div>
      )}

      <div className="grid md:grid-cols-2 gap-4">
        {/* Input */}
        <div>
          <div
            onClick={() => inputRef.current?.click()}
            className="border-2 border-dashed border-border rounded-lg p-8 text-center cursor-pointer hover:border-primary/40 transition-colors"
          >
            {preview ? (
              file?.type.startsWith("video/") ? (
                <video src={preview} className="max-h-48 mx-auto rounded" controls />
              ) : (
                <img src={preview} alt="Input" className="max-h-48 mx-auto rounded" />
              )
            ) : (
              <div className="flex flex-col items-center gap-2 text-muted-foreground">
                <Upload className="h-8 w-8" />
                <p className="text-xs">Click to upload {uploadLabel}</p>
              </div>
            )}
          </div>
          <input ref={inputRef} type="file" accept={accept} className="hidden" onChange={handleFile} />

          <button
            onClick={runInference}
            disabled={!file || loading}
            className="mt-3 w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium disabled:opacity-50 hover:bg-primary/90 transition-colors"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {loading ? "Running inference..." : "Run Model"}
          </button>

          {["object-detection", "image-segmentation", "pose-estimation", "velocity-estimation", "sam2-segmentation"].includes(taskType) && (
            <div className="mt-3 rounded-lg border border-border bg-muted/40 p-3">
              <div className="flex items-center justify-between mb-1">
                <p className="text-[11px] text-muted-foreground">Confidence threshold</p>
                <p className="text-[11px] text-foreground font-mono">{(threshold * 100).toFixed(0)}%</p>
              </div>
              <input
                type="range"
                min={0}
                max={0.95}
                step={0.05}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full accent-primary"
              />
            </div>
          )}

          {taskType === "sam2-segmentation" && (
            <div className="mt-3 rounded-lg border border-border bg-muted/40 p-3">
              <p className="text-[11px] text-muted-foreground">SAM 2 runs in segment-everything mode by default.</p>
              <p className="text-[10px] text-muted-foreground mt-1">Optional point/box prompts can be sent from backend options.</p>
            </div>
          )}
        </div>

        {/* Output */}
        <div className="rounded-lg bg-muted/50 border border-border p-4 min-h-[200px]">
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-3 font-mono">Results</p>
          {error && (
            <div className="flex items-start gap-2 text-destructive text-xs">
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <p>{error}</p>
            </div>
          )}
          {result && !error && (
            <ResultDisplay
              result={result}
              taskType={taskType}
              threshold={threshold}
              inferencePreview={inferencePreview}
            />
          )}
          {!result && !error && !loading && (
            <p className="text-xs text-muted-foreground/60 text-center mt-8">
              Upload an input and click "Run Model" to see results
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function ResultDisplay({
  result,
  taskType,
  threshold,
  inferencePreview,
}: {
  result: any;
  taskType: string;
  threshold: number;
  inferencePreview: string | null;
}) {
  if (taskType === "object-detection" && Array.isArray(result)) {
    const filtered = result.filter((item: any) => (item.score ?? 0) >= threshold);
    if (filtered.length === 0) {
      return <p className="text-xs text-muted-foreground">No objects above threshold.</p>;
    }
    return (
      <div className="space-y-2">
        {inferencePreview && <ImageOverlayResult src={inferencePreview} taskType={taskType} data={filtered} />}
        <p className="text-xs text-foreground font-medium mb-2">{filtered.length} object(s) detected</p>
        {filtered.map((item: any, i: number) => (
          <div key={i} className="rounded bg-card border border-border p-2 text-xs">
            <span className="text-primary font-medium">{item.label}</span>
            <span className="text-muted-foreground ml-2">{(item.score * 100).toFixed(1)}%</span>
            {item.box && (
              <span className="text-muted-foreground/60 ml-2 font-mono text-[10px]">
                [{item.box.xmin}, {item.box.ymin}, {item.box.xmax}, {item.box.ymax}]
              </span>
            )}
          </div>
        ))}
      </div>
    );
  }

  if (taskType === "image-segmentation" && Array.isArray(result)) {
    const filtered = result.filter((item: any) => (item.score ?? 0) >= threshold);
    if (filtered.length === 0) {
      return <p className="text-xs text-muted-foreground">No segments above threshold.</p>;
    }
    return (
      <div className="space-y-2">
        {inferencePreview && <ImageOverlayResult src={inferencePreview} taskType={taskType} data={filtered} />}
        <p className="text-xs text-foreground font-medium mb-2">{filtered.length} segment(s)</p>
        {filtered.map((item: any, i: number) => (
          <div key={i} className="rounded bg-card border border-border p-2 text-xs flex items-center gap-2">
            <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: `hsl(${(i * 40) % 360} 70% 50%)` }} />
            <span className="text-foreground">{item.label}</span>
            <span className="text-muted-foreground font-mono">{(item.score * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    );
  }

  if (taskType === "pose-estimation" && Array.isArray(result)) {
    const filtered = result.filter((item: any) => (item.score ?? 0) >= threshold);
    if (filtered.length === 0) {
      return <p className="text-xs text-muted-foreground">No pose detections above threshold.</p>;
    }
    return (
      <div className="space-y-2">
        {inferencePreview && <ImageOverlayResult src={inferencePreview} taskType={taskType} data={filtered} />}
        <p className="text-xs text-foreground font-medium mb-2">{filtered.length} person pose(s) detected</p>
        {filtered.slice(0, 6).map((item: any, i: number) => (
          <div key={i} className="rounded bg-card border border-border p-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-primary font-medium">{item.label || "person"}</span>
              <span className="text-muted-foreground font-mono">{(item.score * 100).toFixed(1)}%</span>
            </div>
            <p className="text-muted-foreground mt-1">
              {Array.isArray(item.keypoints) ? item.keypoints.length : 0} keypoints
            </p>
          </div>
        ))}
      </div>
    );
  }

  if (taskType === "sam2-segmentation" && Array.isArray(result?.instances)) {
    const filtered = result.instances.filter((item: any) => (item.score ?? 0) >= threshold);
    return (
      <div className="space-y-2">
        {inferencePreview && <ImageOverlayResult src={inferencePreview} taskType="image-segmentation" data={filtered} />}
        <p className="text-xs text-foreground font-medium mb-2">{filtered.length} SAM2 mask(s)</p>
        {Array.isArray(result.concepts) && (
          <p className="text-[11px] text-muted-foreground">Prompts: {result.concepts.join(", ")}</p>
        )}
      </div>
    );
  }
  if (taskType === "depth-estimation" && result?.depth_image) {
    return <img src={`data:image/png;base64,${result.depth_image}`} alt="Depth map" className="rounded max-h-48" />;
  }

  if ((taskType === "velocity-estimation" || taskType === "sam2-segmentation") && result?.annotated_video) {
    return (
      <div className="space-y-3">
        <VideoResult
          base64={result.annotated_video}
          contentType={result.content_type || "video/mp4"}
        />
        {result.metrics && (
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(result.metrics).map(([key, value]) => (
              <div key={key} className="rounded bg-card border border-border p-2">
                <p className="text-[10px] text-muted-foreground font-mono">{key}</p>
                <p className="text-xs text-foreground">{String(value)}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Generic JSON fallback
  return (
    <pre className="text-[10px] font-mono text-muted-foreground whitespace-pre-wrap overflow-auto max-h-64">
      {JSON.stringify(result, null, 2)}
    </pre>
  );
}

function VideoResult({ base64, contentType }: { base64: string; contentType: string }) {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  useEffect(() => {
    try {
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const blob = new Blob([bytes], { type: contentType });
      const url = URL.createObjectURL(blob);
      setVideoUrl(url);
      return () => URL.revokeObjectURL(url);
    } catch {
      setVideoUrl(null);
    }
  }, [base64, contentType]);

  if (!videoUrl) {
    return <p className="text-xs text-muted-foreground">Could not decode annotated video output.</p>;
  }

  return <video controls className="rounded w-full max-h-56 bg-black" src={videoUrl} />;
}

async function extractFirstFrameDataUrl(file: File): Promise<string> {
  const url = URL.createObjectURL(file);
  try {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;
    video.src = url;

    await new Promise<void>((resolve, reject) => {
      video.onloadeddata = () => resolve();
      video.onerror = () => reject(new Error("Could not decode video for frame extraction."));
    });

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Could not create canvas context.");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.92);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function ImageOverlayResult({
  src,
  taskType,
  data,
}: {
  src: string;
  taskType: string;
  data: any[];
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0, img.width, img.height);

      ctx.lineWidth = Math.max(2, Math.round(img.width / 420));
      ctx.font = `${Math.max(12, Math.round(img.width / 50))}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;

      data.forEach((item, idx) => {
        const hue = (idx * 47) % 360;
        const color = `hsl(${hue} 95% 55%)`;
        const box = item.box ?? (
          Array.isArray(item.bbox) && item.bbox.length === 4
            ? { xmin: item.bbox[0], ymin: item.bbox[1], xmax: item.bbox[2], ymax: item.bbox[3] }
            : null
        );

        if (taskType === "image-segmentation" && Array.isArray(item.mask_polygon) && item.mask_polygon.length > 2) {
          const poly = item.mask_polygon as number[][];
          ctx.beginPath();
          ctx.moveTo(Number(poly[0][0]), Number(poly[0][1]));
          for (let p = 1; p < poly.length; p++) {
            ctx.lineTo(Number(poly[p][0]), Number(poly[p][1]));
          }
          ctx.closePath();
          ctx.fillStyle = `hsl(${hue} 95% 55% / 0.25)`;
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.stroke();
        }

        if (box) {
          const x = Number(box.xmin);
          const y = Number(box.ymin);
          const w = Number(box.xmax) - x;
          const h = Number(box.ymax) - y;
          if (w > 0 && h > 0) {
            ctx.strokeStyle = color;
            ctx.strokeRect(x, y, w, h);
            const label = `${item.label || item.class_name || "obj"} ${(Number(item.score ?? item.confidence ?? 0) * 100).toFixed(1)}%`;
            const textW = ctx.measureText(label).width + 8;
            const textH = 18;
            ctx.fillStyle = color;
            ctx.fillRect(x, Math.max(0, y - textH), textW, textH);
            ctx.fillStyle = "#111";
            ctx.fillText(label, x + 4, Math.max(12, y - 4));
          }
        }

        if (taskType === "pose-estimation" && Array.isArray(item.keypoints)) {
          ctx.fillStyle = color;
          for (const kp of item.keypoints) {
            const [kx, ky, kv] = kp;
            if (Number(kv ?? 1) <= 0) continue;
            ctx.beginPath();
            ctx.arc(Number(kx), Number(ky), Math.max(2, Math.round(img.width / 220)), 0, Math.PI * 2);
            ctx.fill();
          }
        }
      });
    };
    img.src = src;
  }, [src, taskType, data]);

  return <canvas ref={canvasRef} className="rounded w-full border border-border bg-black/20" />;
}




