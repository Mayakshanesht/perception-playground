import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft, ChevronDown, ArrowRight, Lightbulb, FlaskConical, BookOpen } from "lucide-react";
import { Link } from "react-router-dom";

interface Studio {
  id: string;
  title: string;
  description: string;
  pipeline: { step: string; module: string; path: string }[];
  concepts: string[];
  explanation: string;
  exercise: string;
}

const studios: Studio[] = [
  {
    id: "lane-detection",
    title: "Lane Detection System",
    description: "Design a complete lane detection pipeline for autonomous driving, from camera capture to lane boundary estimation.",
    pipeline: [
      { step: "Camera Capture", module: "Camera", path: "/module/camera" },
      { step: "Semantic Segmentation", module: "Semantic", path: "/module/semantic" },
      { step: "Geometric Fitting", module: "Geometric", path: "/module/geometric" },
      { step: "Lane Estimation", module: "Motion", path: "/module/motion" },
    ],
    concepts: ["Camera calibration", "Semantic segmentation", "Perspective transform (IPM)", "Polynomial curve fitting", "Temporal smoothing"],
    explanation: "A lane detection system starts with a calibrated camera to remove distortion. Semantic segmentation identifies lane marking pixels. An Inverse Perspective Mapping (IPM) transforms the front-view image into a bird's-eye view where lane lines appear parallel. Polynomial fitting (2nd or 3rd order) extracts lane boundaries. Temporal smoothing across frames ensures stable lane estimates even when markings are occluded.",
    exercise: "Given a front-view camera image with detected lane pixels, describe the steps you would take to estimate the lane curvature in world coordinates. What information do you need from camera calibration?",
  },
  {
    id: "ad-perception",
    title: "Autonomous Driving Perception",
    description: "Build a multi-task perception stack that detects, classifies, and tracks objects in driving scenarios.",
    pipeline: [
      { step: "Camera + Calibration", module: "Camera", path: "/module/camera" },
      { step: "Object Detection", module: "Semantic", path: "/module/semantic" },
      { step: "Multi-Object Tracking", module: "Motion", path: "/module/motion" },
      { step: "Velocity Estimation", module: "Motion", path: "/module/motion" },
    ],
    concepts: ["Multi-camera calibration", "2D/3D object detection", "DeepSORT tracking", "Kalman filter prediction", "Ego-motion compensation"],
    explanation: "Autonomous driving perception combines detection (YOLO/Faster R-CNN) with multi-object tracking (SORT/DeepSORT). Detection runs per-frame, producing bounding boxes. The tracker associates detections across frames using the Hungarian algorithm with IoU and appearance features. A Kalman filter predicts object motion between frames. Ego-motion from the vehicle's own movement must be compensated to estimate true object velocities.",
    exercise: "An object is detected at position (200, 300) in frame t and (220, 295) in frame t+1 (30 FPS). The camera has focal length f=1000px and the object is at depth Z=20m. Estimate the object's approximate real-world lateral velocity.",
  },
  {
    id: "3d-reconstruction",
    title: "3D Scene Reconstruction",
    description: "Reconstruct a complete 3D model of a scene from a collection of unstructured photographs.",
    pipeline: [
      { step: "Feature Extraction", module: "Reconstruction", path: "/module/reconstruction" },
      { step: "Feature Matching", module: "Reconstruction", path: "/module/reconstruction" },
      { step: "Structure from Motion", module: "Reconstruction", path: "/module/reconstruction" },
      { step: "Multi-View Stereo", module: "Reconstruction", path: "/module/reconstruction" },
      { step: "Neural Rendering", module: "Reconstruction", path: "/module/reconstruction" },
    ],
    concepts: ["SIFT/SuperPoint features", "Epipolar geometry", "Bundle adjustment", "Dense depth maps", "NeRF/3DGS"],
    explanation: "Starting from unstructured photos: (1) Extract SIFT/SuperPoint features, (2) Match features across image pairs, (3) Incrementally build the 3D structure via SfM — estimate camera poses and triangulate sparse 3D points, (4) Refine with bundle adjustment, (5) Use MVS for dense reconstruction OR train NeRF/3DGS for photorealistic novel view synthesis. The choice between MVS and neural rendering depends on whether you need an explicit mesh or photorealistic rendering.",
    exercise: "You have 50 photos of a building. After SfM, you get 10,000 sparse 3D points. Describe the trade-offs between using MVS (COLMAP dense) vs NeRF for the next step. Consider: output format, computation time, rendering quality, and editability.",
  },
  {
    id: "robot-reasoning",
    title: "Robot Scene Reasoning",
    description: "Enable a robot to understand its environment using detection, embeddings, and multimodal language models.",
    pipeline: [
      { step: "Camera Input", module: "Camera", path: "/module/camera" },
      { step: "Object Detection", module: "Semantic", path: "/module/semantic" },
      { step: "Visual Embeddings", module: "Scene Reasoning", path: "/module/scene-reasoning" },
      { step: "Multimodal LLM", module: "Scene Reasoning", path: "/module/scene-reasoning" },
    ],
    concepts: ["Object detection", "CLIP embeddings", "Vision-language models", "Spatial reasoning", "Task planning"],
    explanation: "Robot scene reasoning combines perception with language understanding. The camera captures the scene, detection identifies objects and their locations, CLIP embeddings encode visual concepts in a language-aligned space, and a multimodal LLM (GPT-4V/Gemini) performs high-level reasoning: 'The red cup is on the table near the edge — I should move it to prevent it from falling.' This enables robots to follow natural language instructions and reason about spatial relationships.",
    exercise: "A robot sees a kitchen scene with a cup near the edge of a table, a closed cabinet, and a person pointing at the cabinet. Using CLIP similarity, how would you determine what the person wants the robot to do? Design the reasoning pipeline.",
  },
];

const moduleColors: Record<string, string> = {
  Camera: "var(--module-camera)",
  Semantic: "var(--module-semantic)",
  Geometric: "var(--module-geometric)",
  Motion: "var(--module-motion)",
  Reconstruction: "var(--module-reconstruction)",
  "Scene Reasoning": "var(--module-reasoning)",
};

export default function PerceptionStudios() {
  const [expandedStudio, setExpandedStudio] = useState<string | null>(studios[0].id);

  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      <div className="mb-8">
        <div className="flex items-center gap-3 mb-3">
          <div className="h-10 w-10 rounded-xl bg-accent/10 flex items-center justify-center">
            <FlaskConical className="h-5 w-5 text-accent" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-foreground tracking-tight">Perception Studios</h1>
            <p className="text-sm text-muted-foreground">Capstone labs: design complete perception pipelines</p>
          </div>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed max-w-3xl">
          Perception Studios are system-level exercises where you combine concepts from multiple modules to design end-to-end perception pipelines. 
          Each studio presents a real-world problem, the required pipeline, key concepts, and a hands-on exercise.
        </p>
      </div>

      <div className="space-y-4">
        {studios.map((studio, studioIdx) => {
          const isExpanded = expandedStudio === studio.id;
          return (
            <div key={studio.id} className="rounded-xl border border-border bg-card/80 overflow-hidden">
              <button
                onClick={() => setExpandedStudio(isExpanded ? null : studio.id)}
                className="w-full flex items-center gap-4 p-5 hover:bg-muted/30 transition-colors text-left"
              >
                <div className="h-10 w-10 rounded-xl bg-accent/10 flex items-center justify-center shrink-0">
                  <span className="text-lg font-bold text-accent">{studioIdx + 1}</span>
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-foreground text-sm">{studio.title}</h3>
                  <p className="text-xs text-muted-foreground mt-0.5">{studio.description}</p>
                </div>
                <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${isExpanded ? "rotate-180" : ""}`} />
              </button>

              <AnimatePresence initial={false}>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    className="overflow-hidden"
                  >
                    <div className="px-5 pb-5 space-y-5">
                      {/* Pipeline diagram */}
                      <div>
                        <h4 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3">Pipeline</h4>
                        <div className="flex flex-wrap items-center gap-2">
                          {studio.pipeline.map((step, i) => (
                            <div key={step.step} className="flex items-center gap-2">
                              <Link
                                to={step.path}
                                className="px-3 py-2 rounded-lg border border-border bg-muted/30 text-xs font-medium text-foreground hover:border-primary/40 transition-all"
                                style={{ borderColor: `hsl(${moduleColors[step.module] || "var(--border)"} / 0.3)` }}
                              >
                                <span style={{ color: `hsl(${moduleColors[step.module] || "var(--foreground)"})` }}>{step.step}</span>
                              </Link>
                              {i < studio.pipeline.length - 1 && (
                                <ArrowRight className="h-3 w-3 text-muted-foreground/40 shrink-0" />
                              )}
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Concepts required */}
                      <div>
                        <h4 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-2">Concepts Required</h4>
                        <div className="flex flex-wrap gap-1.5">
                          {studio.concepts.map((c) => (
                            <span key={c} className="text-[10px] px-2 py-1 rounded-full bg-primary/10 text-primary font-medium">{c}</span>
                          ))}
                        </div>
                      </div>

                      {/* Explanation */}
                      <div className="rounded-lg bg-muted/30 border border-border p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Lightbulb className="h-3.5 w-3.5 text-accent" />
                          <h4 className="text-xs font-semibold text-foreground uppercase tracking-wider">System Design</h4>
                        </div>
                        <p className="text-sm text-muted-foreground leading-relaxed">{studio.explanation}</p>
                      </div>

                      {/* Exercise */}
                      <div className="rounded-lg bg-primary/5 border border-primary/20 p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <BookOpen className="h-3.5 w-3.5 text-primary" />
                          <h4 className="text-xs font-semibold text-foreground uppercase tracking-wider">Exercise</h4>
                        </div>
                        <p className="text-sm text-foreground leading-relaxed">{studio.exercise}</p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>
    </div>
  );
}
