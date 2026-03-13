import { QuizQuestion } from "@/components/ConceptQuiz";
import { FailureMode } from "@/components/FailureModes";

// Quiz questions for each module
export const moduleQuizzes: Record<string, QuizQuestion[]> = {
  semantic: [
    {
      question: "What is the key difference between semantic and instance segmentation?",
      options: [
        "Semantic uses CNNs while instance uses transformers",
        "Semantic labels every pixel by class; instance additionally separates individual objects",
        "Instance segmentation is faster than semantic segmentation",
        "Semantic segmentation only works on indoor scenes",
      ],
      correctIndex: 1,
      explanation: "Semantic segmentation assigns a class label to every pixel (all cars share one label), while instance segmentation separates individual object instances (car-1 vs car-2).",
    },
    {
      question: "What problem does Focal Loss (RetinaNet) address?",
      options: [
        "Slow training convergence",
        "Class imbalance between easy and hard examples in one-stage detectors",
        "Bounding box regression inaccuracy",
        "Feature pyramid resolution loss",
      ],
      correctIndex: 1,
      explanation: "One-stage detectors face extreme class imbalance (~100k background vs few object locations). Focal Loss down-weights easy negatives, focusing training on hard examples.",
    },
    {
      question: "How does DETR eliminate the need for NMS and anchor boxes?",
      options: [
        "It uses a very high confidence threshold",
        "It processes one object at a time",
        "It uses set-based Hungarian matching to produce unique predictions",
        "It uses a voting mechanism between detections",
      ],
      correctIndex: 2,
      explanation: "DETR treats detection as a set prediction problem. Hungarian matching ensures each prediction is unique, eliminating duplicate detections that NMS would normally filter.",
    },
    {
      question: "Why does Faster R-CNN outperform R-CNN despite using the same classification head?",
      options: [
        "It uses a more powerful ResNet backbone",
        "It runs the CNN once on the whole image and extracts RoIs from the shared feature map",
        "It predicts bounding boxes in a single forward pass without proposals",
        "It replaces cross-entropy loss with focal loss",
      ],
      correctIndex: 1,
      explanation: "R-CNN ran AlexNet separately on each of ~2000 warped proposals — extremely slow. Faster R-CNN computes the feature map once, then uses RoI Pooling to extract fixed-size regions from it. The RPN also replaces Selective Search with learned proposals.",
    },
    {
      question: "Why does Vision Transformer (ViT) need very large datasets to match CNN performance?",
      options: [
        "ViT uses more parameters than ResNet",
        "ViT cannot process images larger than 224×224",
        "CNN has built-in inductive biases (translation equivariance, locality) that ViT must learn from data",
        "ViT requires higher resolution patches to capture fine details",
      ],
      correctIndex: 2,
      explanation: "CNNs have strong inductive biases baked in: local receptive fields and weight sharing encode the assumption that nearby pixels are related and features are translation-invariant. ViT has no such priors — it treats every patch equally via global attention and must learn spatial structure from scratch, requiring massive datasets.",
    },
    {
      question: "In U-Net, what information do skip connections preserve?",
      options: [
        "Class probability distributions from deep layers",
        "High-resolution spatial detail from the encoder that is lost during pooling",
        "Gradient flow to prevent vanishing gradients",
        "Channel statistics for batch normalization",
      ],
      correctIndex: 1,
      explanation: "Pooling layers in the encoder compress spatial dimensions to capture global context, but lose fine-grained location detail. U-Net's skip connections concatenate the full-resolution encoder feature maps directly to the corresponding decoder level, giving the decoder both semantic understanding AND precise localization.",
    },
  ],
  geometric: [
    {
      question: "Why does monocular depth estimation suffer from scale ambiguity?",
      options: [
        "Cameras cannot estimate depth at all",
        "Multiple 3D scenes can project to the same 2D image",
        "Neural networks cannot learn depth representations",
        "Depth sensors are always required",
      ],
      correctIndex: 1,
      explanation: "A single 2D image is consistent with infinitely many 3D interpretations — a small nearby object produces the same projection as a large distant one. This is the fundamental ambiguity of monocular depth.",
    },
    {
      question: "What does the stereo depth formula Z = fB/d tell us?",
      options: [
        "Depth increases with larger disparity",
        "Depth is inversely proportional to disparity",
        "Focal length doesn't affect depth estimation",
        "Baseline doesn't matter for depth",
      ],
      correctIndex: 1,
      explanation: "Z = fB/d means depth is inversely proportional to disparity. Closer objects have larger disparity (appear to shift more between left and right views), giving smaller Z.",
    },
  ],
  motion: [
    {
      question: "What is the aperture problem in optical flow?",
      options: [
        "Cameras have limited field of view",
        "One equation with two unknowns (u, v) makes flow under-determined",
        "Motion blur prevents accurate estimation",
        "Objects move too fast for the camera",
      ],
      correctIndex: 1,
      explanation: "The optical flow equation Ix·u + Iy·v + It = 0 gives one constraint for two unknowns. Through a small aperture, you can only measure the flow component perpendicular to the edge, not the full 2D motion.",
    },
    {
      question: "What is the key innovation of RAFT for optical flow?",
      options: [
        "Using larger convolution kernels",
        "Building a 4D all-pairs correlation volume with iterative GRU-based refinement",
        "Using optical flow as input to another network",
        "Training on real-world video only",
      ],
      correctIndex: 1,
      explanation: "RAFT computes correlations between ALL pairs of pixels (4D volume), then iteratively refines flow estimates using a GRU. This enables strong generalization and state-of-the-art accuracy.",
    },
  ],
  reconstruction: [
    {
      question: "What does bundle adjustment optimize?",
      options: [
        "Only the 3D point positions",
        "Only the camera poses",
        "Both 3D points and camera parameters simultaneously by minimizing reprojection error",
        "The image pixel values",
      ],
      correctIndex: 2,
      explanation: "Bundle adjustment jointly optimizes all camera parameters and 3D point positions to minimize the sum of squared reprojection errors — the gold standard for refinement in SfM.",
    },
    {
      question: "How does 3D Gaussian Splatting achieve real-time rendering compared to NeRF?",
      options: [
        "It uses a smaller neural network",
        "It represents scenes as explicit 3D Gaussians rendered via rasterization instead of ray marching",
        "It renders at lower resolution",
        "It pre-computes all possible views",
      ],
      correctIndex: 1,
      explanation: "3DGS uses explicit 3D Gaussians (not neural networks) that are projected and alpha-composited via fast rasterization. This avoids the expensive per-ray sampling of NeRF, enabling >100 FPS.",
    },
  ],
  camera: [
    {
      question: "What does the intrinsic matrix K encode?",
      options: [
        "The camera's position and orientation in the world",
        "The lens distortion coefficients",
        "The camera's internal parameters: focal lengths and principal point",
        "The image resolution",
      ],
      correctIndex: 2,
      explanation: "K contains focal lengths (fx, fy) and the principal point (cx, cy) — internal camera properties independent of the camera's position in the world.",
    },
  ],
  "scene-reasoning": [
    {
      question: "What is the key insight behind CLIP's contrastive learning?",
      options: [
        "Training on a single large dataset with fixed labels",
        "Learning aligned vision-language representations by pulling matching image-text pairs together",
        "Using only visual features for classification",
        "Fine-tuning on each downstream task separately",
      ],
      correctIndex: 1,
      explanation: "CLIP learns a shared embedding space where matching image-text pairs are close and non-matching ones are far apart. This enables zero-shot classification by comparing image embeddings with text descriptions of classes.",
    },
  ],
};

export const moduleFailureModes: Record<string, FailureMode[]> = {
  semantic: [
    {
      title: "Object Detection Failures",
      causes: [
        "Small objects below the network's effective receptive field",
        "Heavy occlusion where less than 30% of the object is visible",
        "Motion blur at high speeds making features unrecognizable",
        "Domain shift between training data and deployment environment",
      ],
    },
    {
      title: "Segmentation Failures",
      causes: [
        "Thin structures (poles, wires) lost during downsampling",
        "Ambiguous boundaries between similar-looking objects",
        "Novel object classes not present in training data",
        "Reflective or transparent surfaces confusing appearance models",
      ],
    },
  ],
  geometric: [
    {
      title: "Depth Estimation Failures",
      causes: [
        "Glass and reflective surfaces create phantom depths",
        "Textureless regions (white walls) provide no depth cues",
        "Low lighting reduces visible texture gradients",
        "Scale ambiguity in monocular methods — can't recover absolute distance",
        "Repeated textures cause matching ambiguity in stereo",
      ],
    },
    {
      title: "Pose Estimation Failures",
      causes: [
        "Self-occlusion when limbs are hidden behind the body",
        "Crowded scenes where people overlap extensively",
        "Unusual poses not well-represented in training data",
        "Low resolution making joint locations ambiguous",
      ],
    },
  ],
  motion: [
    {
      title: "Optical Flow Failures",
      causes: [
        "Large displacements exceeding the search range",
        "Occlusion boundaries where pixels appear/disappear",
        "Brightness changes violating the constancy assumption",
        "Textureless regions where flow is under-determined",
      ],
    },
    {
      title: "Tracking Failures",
      causes: [
        "Identity switches during prolonged occlusions",
        "Crowded scenes with many similar-looking objects",
        "Abrupt appearance changes (lighting, perspective)",
        "Objects entering/leaving the field of view",
      ],
    },
  ],
  reconstruction: [
    {
      title: "SfM Failures",
      causes: [
        "Insufficient texture for feature matching",
        "Repetitive patterns causing incorrect matches",
        "Moving objects violating the static scene assumption",
        "Wide baselines with too few corresponding features",
      ],
    },
    {
      title: "Neural Rendering Failures",
      causes: [
        "Sparse input views causing artifacts in unobserved regions",
        "Reflective/specular surfaces with view-dependent appearance",
        "Dynamic scenes not handled by static NeRF/3DGS",
        "Long training times for NeRF (hours per scene)",
      ],
    },
  ],
  camera: [
    {
      title: "Calibration Failures",
      causes: [
        "Insufficient calibration images or poor coverage of orientations",
        "Blurry images causing inaccurate corner detection",
        "Non-planar calibration targets with manufacturing defects",
        "Temperature changes causing lens expansion after calibration",
      ],
    },
  ],
  "scene-reasoning": [
    {
      title: "Visual Reasoning Failures",
      causes: [
        "Hallucination — generating plausible but incorrect descriptions",
        "Counting errors for more than ~5 objects",
        "Spatial reasoning mistakes (left/right, behind/in-front)",
        "Inability to read small or stylized text in images",
        "Sensitivity to prompt phrasing affecting answer quality",
      ],
    },
  ],
};
