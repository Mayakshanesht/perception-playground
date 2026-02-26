import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Upload, Play, Loader2, ImageIcon, AlertCircle } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";

interface PlaygroundProps {
  title: string;
  description: string;
  taskType: string;
  acceptVideo?: boolean;
  acceptImage?: boolean;
}

export default function Playground({ title, description, taskType, acceptVideo = false, acceptImage = true }: PlaygroundProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = [
    ...(acceptImage ? ["image/jpeg", "image/png", "image/webp"] : []),
    ...(acceptVideo ? ["video/mp4", "video/webm"] : []),
  ].join(",");

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    if (f.type.startsWith("image/")) {
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
      const reader = new FileReader();
      const base64 = await new Promise<string>((resolve) => {
        reader.onload = () => {
          const result = reader.result as string;
          resolve(result.split(",")[1]);
        };
        reader.readAsDataURL(file);
      });

      const { data, error: fnError } = await supabase.functions.invoke("hf-inference", {
        body: { image: base64, task: taskType, mimeType: file.type },
      });

      if (fnError) throw fnError;
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
                <p className="text-xs">Click to upload {acceptVideo ? "image or video" : "an image"}</p>
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
            <ResultDisplay result={result} taskType={taskType} previewSrc={preview} />
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

function ResultDisplay({ result, taskType, previewSrc }: { result: any; taskType: string; previewSrc: string | null }) {
  if (taskType === "image-classification" && Array.isArray(result)) {
    return (
      <div className="space-y-2">
        {result.map((item: any, i: number) => (
          <div key={i} className="flex items-center gap-2">
            <div className="flex-1">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-foreground font-medium">{item.label}</span>
                <span className="text-muted-foreground font-mono">{(item.score * 100).toFixed(1)}%</span>
              </div>
              <div className="h-1.5 rounded-full bg-border overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${item.score * 100}%` }}
                  className="h-full rounded-full bg-primary"
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (taskType === "object-detection" && Array.isArray(result)) {
    return (
      <div className="space-y-2">
        <p className="text-xs text-foreground font-medium mb-2">{result.length} object(s) detected</p>
        {result.map((item: any, i: number) => (
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
    return (
      <div className="space-y-2">
        <p className="text-xs text-foreground font-medium mb-2">{result.length} segment(s)</p>
        {result.map((item: any, i: number) => (
          <div key={i} className="rounded bg-card border border-border p-2 text-xs flex items-center gap-2">
            <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: `hsl(${(i * 40) % 360} 70% 50%)` }} />
            <span className="text-foreground">{item.label}</span>
            <span className="text-muted-foreground font-mono">{(item.score * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    );
  }

  if (taskType === "depth-estimation" && result?.depth_image) {
    return <img src={`data:image/png;base64,${result.depth_image}`} alt="Depth map" className="rounded max-h-48" />;
  }

  // Generic JSON fallback
  return (
    <pre className="text-[10px] font-mono text-muted-foreground whitespace-pre-wrap overflow-auto max-h-64">
      {JSON.stringify(result, null, 2)}
    </pre>
  );
}
