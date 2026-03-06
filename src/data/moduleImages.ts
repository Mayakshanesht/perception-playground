export interface ModuleImageData {
  src: string;
  alt: string;
  caption: string;
}

export const moduleImages: Record<string, ModuleImageData[]> = {
  camera: [
    {
      src: "/images/camera-model-coords.png",
      alt: "Camera coordinate systems showing world, camera, and image coordinate frames",
      caption: "Camera coordinate systems: A 3D world point is transformed from world coordinates (Ow) to camera coordinates (Oc) via extrinsic parameters [R|t], then projected onto the image plane (Oi → Op) via the intrinsic matrix K. The focal length determines the scaling between 3D and 2D.",
    },
    {
      src: "/images/calibration.jpg",
      alt: "Camera calibration using checkerboard patterns showing various distortion types",
      caption: "Camera calibration with checkerboard patterns. Top: detected corners on a calibration board. Bottom-left: fisheye lens with extreme barrel distortion. Bottom-center: radial distortion visualized on a grid. Bottom-right: the same pattern projected onto different surfaces demonstrates the need for intrinsic/extrinsic calibration.",
    },
  ],
  semantic: [
    {
      src: "/images/segmentation-types.png",
      alt: "Comparison of semantic segmentation, classification, object detection, and instance segmentation",
      caption: "Types of visual recognition tasks: Semantic segmentation labels every pixel by class (no object separation). Classification + Localization identifies a single object. Object detection localizes multiple objects with bounding boxes. Instance segmentation combines detection with per-instance pixel masks.",
    },
    {
      src: "/images/cityscapes-segmentation.webp",
      alt: "Cityscapes dataset semantic segmentation examples",
      caption: "Semantic segmentation on the Cityscapes dataset. Each color represents a different class (road, sidewalk, car, person, building, vegetation). This pixel-level understanding is critical for autonomous driving perception.",
    },
  ],
  geometric: [
    {
      src: "/images/deep-learning-disparity.png",
      alt: "Deep learning stereo matching pipeline for disparity estimation",
      caption: "Deep learning pipeline for stereo disparity estimation: Input stereo image pairs are processed through shared-weight feature extractors, then a sparse cost volume is constructed. Similarity evaluation and loss computation produce dense disparity maps for depth estimation.",
    },
    {
      src: "/images/pose-tracking.png",
      alt: "Human pose estimation and tracking across video sequences",
      caption: "Pose estimation and tracking pipeline: Image sequences are processed to extract joint-trajectories in high-dimensional space. Temporal alignment matrices match movements to known motion capture sequences (walking, running, boxing) to infer full 3D skeleton poses.",
    },
  ],
  motion: [
    {
      src: "/images/optical-flow.png",
      alt: "Optical flow visualization on basketball scene",
      caption: "Optical flow estimation between consecutive video frames. Left: original frames showing a basketball player. Right: color-coded flow field where hue encodes direction and saturation encodes magnitude. The moving person creates a distinct flow pattern against the static background.",
    },
  ],
  reconstruction: [
    {
      src: "/images/nerf-pipeline.png",
      alt: "NeRF architecture and Instant-NGP hash encoding",
      caption: "Top: NeRF pipeline — 5D input (position + direction) is mapped through an MLP to produce color and density. Volume rendering integrates samples along each ray. Bottom: Instant-NGP uses multi-resolution hash encoding for dramatically faster training while maintaining quality.",
    },
  ],
  "scene-reasoning": [
    {
      src: "/images/vlm-architecture.png",
      alt: "Vision Language Model architecture with visual and textual encoders",
      caption: "Vision-Language Model architecture: An input image passes through a Visual Encoder while text goes through a Textual Encoder. Both representations are combined in a Fusion Block, processed by feedforward layers, and a softmax produces the output — enabling tasks like image captioning and visual question answering.",
    },
  ],
};
