export interface TutorialCard {
  id: string;
  title: string;
  description: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  duration: string;
  tags: string[];
  imageUrl: string;
  colabUrl: string;
  topics: string[];
  learningObjectives: string[];
}

export const tutorialsContent: TutorialCard[] = [
  {
    id: "object-detection-kitti",
    title: "Object Detection on KITTI Dataset",
    description: "Learn object detection using YOLO models on the KITTI autonomous driving dataset.",
    difficulty: "Beginner",
    duration: "45 min",
    tags: ["object-detection", "yolo", "kitti", "autonomous-driving"],
    imageUrl: "/images/tutorial-od.jpg",
    colabUrl: "https://colab.research.google.com/drive/1GbNu2a0rY-LphWMHCI-ksUsJ6OgRYttn?usp=sharing",
    topics: ["YOLO Architecture", "Bounding Box Regression", "KITTI Dataset", "Evaluation Metrics"],
    learningObjectives: [
      "Understand YOLO object detection architecture",
      "Learn to work with KITTI dataset format",
      "Implement training and evaluation pipelines",
      "Master mAP and IoU evaluation metrics",
    ],
  },
  {
    id: "instance-segmentation-kitti",
    title: "Instance Segmentation on KITTI Dataset",
    description: "Master instance segmentation using advanced models on KITTI dataset for autonomous driving.",
    difficulty: "Intermediate",
    duration: "60 min",
    tags: ["instance-segmentation", "kitti", "mask-rcnn", "autonomous-driving"],
    imageUrl: "/images/tutorial-seg.jpg",
    colabUrl: "https://colab.research.google.com/drive/12ksImTpZFdweIl0OXfJGifJi9qKBgCoi?usp=sharing",
    topics: ["Mask R-CNN", "Instance Segmentation", "KITTI Dataset", "Mask Evaluation"],
    learningObjectives: [
      "Learn instance segmentation architectures",
      "Understand mask generation and refinement",
      "Work with KITTI segmentation annotations",
      "Implement mask IoU and AP evaluation",
    ],
  },
  {
    id: "depth-estimation",
    title: "Monocular Depth Estimation",
    description: "Learn depth estimation from single images using modern neural networks.",
    difficulty: "Intermediate",
    duration: "50 min",
    tags: ["depth-estimation", "monocular", "neural-networks", "3d-reconstruction"],
    imageUrl: "/images/tutorial-depth.jpg",
    colabUrl: "https://colab.research.google.com/drive/1JqoOYe3zZgauMOL5hxvpw5yHDzJBOvoU?usp=sharing",
    topics: ["Depth Prediction", "Encoder-Decoder Architecture", "Self-Supervised Learning"],
    learningObjectives: [
      "Understand monocular depth estimation",
      "Learn encoder-decoder architectures",
      "Implement depth loss functions",
      "Evaluate depth prediction accuracy",
    ],
  },
  {
    id: "velocity-estimation",
    title: "Velocity Estimation and Tracking",
    description: "Learn to estimate object velocities and track them across video frames.",
    difficulty: "Advanced",
    duration: "55 min",
    tags: ["velocity-estimation", "tracking", "optical-flow", "kalman-filter"],
    imageUrl: "/images/tutorial-velocity.jpg",
    colabUrl: "https://colab.research.google.com/drive/1syVJhN5p2ipB878Ou7z1hn_xaftQyr-V?usp=sharing",
    topics: ["Object Tracking", "Velocity Estimation", "Optical Flow", "Multi-Object Tracking"],
    learningObjectives: [
      "Learn object tracking algorithms",
      "Understand velocity estimation techniques",
      "Implement tracking-by-detection",
      "Master evaluation metrics for tracking",
    ],
  },
  {
    id: "scene-reasoning-florence",
    title: "Scene Reasoning with Florence-2",
    description: "Advanced multimodal scene understanding using Florence-2 large model for visual reasoning.",
    difficulty: "Advanced",
    duration: "65 min",
    tags: ["scene-reasoning", "florence-2", "multimodal", "visual-grounding"],
    imageUrl: "/images/tutorial-florence.jpg",
    colabUrl: "https://colab.research.google.com/drive/1K9VvjUNserYPSJWvArJMUZ_UkPj3KZ0F?usp=sharing",
    topics: ["Florence-2 Architecture", "Visual Question Answering", "Object Detection", "Region Segmentation"],
    learningObjectives: [
      "Understand Florence-2 multimodal architecture",
      "Learn visual-text embedding techniques",
      "Implement object detection and segmentation",
      "Master scene reasoning and captioning",
    ],
  },
];
