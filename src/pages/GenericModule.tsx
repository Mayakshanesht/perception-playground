import { motion } from "framer-motion";
import { ArrowLeft, BookOpen } from "lucide-react";
import { Link, useParams } from "react-router-dom";

const moduleInfo: Record<string, { title: string; description: string; topics: string[] }> = {
  classification: {
    title: "Image Classification",
    description: "Assign a single label to an entire image. Foundation of deep learning in vision.",
    topics: ["AlexNet (2012)", "VGGNet (2014)", "GoogLeNet / Inception", "ResNet & Skip Connections", "Vision Transformers (ViT)"],
  },
  segmentation: {
    title: "Segmentation",
    description: "Pixel-level understanding: semantic, instance, and panoptic segmentation.",
    topics: ["FCN", "U-Net", "DeepLab v3+", "Mask R-CNN (Instance)", "Panoptic Segmentation"],
  },
  depth: {
    title: "Depth Estimation",
    description: "Predict depth from monocular or stereo images for 3D scene understanding.",
    topics: ["Stereo Matching", "Disparity Maps", "Monocular Depth (MiDaS)", "Self-supervised Depth", "LiDAR Fusion"],
  },
  sfm: {
    title: "Structure from Motion",
    description: "Reconstruct 3D structure and camera poses from 2D image sequences.",
    topics: ["Feature Detection (SIFT, ORB)", "Epipolar Geometry", "Essential & Fundamental Matrix", "Bundle Adjustment", "COLMAP Pipeline"],
  },
  nerf: {
    title: "Neural Rendering",
    description: "Novel view synthesis using neural representations of 3D scenes.",
    topics: ["NeRF: Volumetric Rendering", "Positional Encoding", "Instant NGP", "3D Gaussian Splatting", "Differentiable Rendering"],
  },
  pose: {
    title: "Pose Estimation",
    description: "Detect and localize human body keypoints in 2D and 3D.",
    topics: ["OpenPose", "HRNet", "MediaPipe", "3D Pose Lifting", "Multi-person Pose"],
  },
  tracking: {
    title: "Multi-Object Tracking",
    description: "Track multiple objects across video frames with consistent identities.",
    topics: ["SORT Algorithm", "DeepSORT", "Hungarian Algorithm", "Kalman Filter", "ReID Features"],
  },
  action: {
    title: "Video Action Recognition",
    description: "Classify human actions in video using temporal modeling.",
    topics: ["Two-Stream Networks", "3D CNNs (C3D, I3D)", "SlowFast Networks", "Video Transformers (TimeSformer)", "Temporal Shift Module"],
  },
};

export default function GenericModule() {
  const { moduleId } = useParams<{ moduleId: string }>();
  const info = moduleInfo[moduleId || ""];

  if (!info) {
    return (
      <div className="p-8">
        <p className="text-muted-foreground">Module not found.</p>
        <Link to="/" className="text-primary text-sm mt-2 inline-block">← Back to Dashboard</Link>
      </div>
    );
  }

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      <h1 className="text-2xl font-bold text-foreground tracking-tight mb-2">{info.title}</h1>
      <p className="text-sm text-muted-foreground mb-8 max-w-2xl">{info.description}</p>

      <div className="flex items-center gap-2 mb-4">
        <BookOpen className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">Key Topics</h2>
      </div>

      <div className="space-y-3">
        {info.topics.map((topic, i) => (
          <motion.div
            key={topic}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.06 }}
            className="rounded-lg border border-border bg-card p-4 flex items-center gap-4"
          >
            <div className="h-8 w-8 rounded-md bg-primary/10 flex items-center justify-center text-primary font-mono text-xs font-bold">
              {String(i + 1).padStart(2, "0")}
            </div>
            <span className="text-sm text-foreground">{topic}</span>
          </motion.div>
        ))}
      </div>

      <div className="mt-8 rounded-xl border border-dashed border-border p-8 text-center">
        <p className="text-sm text-muted-foreground">
          Interactive architecture visualizer and sandbox coming soon for this module.
        </p>
      </div>
    </div>
  );
}
