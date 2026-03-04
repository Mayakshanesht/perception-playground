import { ModuleContent } from "./moduleContent";

// Re-export existing content and add new consolidated module data
// Camera Image Generation module content
export const cameraModule: ModuleContent = {
  id: "camera",
  title: "Camera & Image Formation",
  subtitle: "Understand how cameras capture the 3D world as 2D images — the foundation of every computer vision pipeline.",
  color: "220 70% 55%",
  theory: [
    {
      title: "Intuition",
      content:
        "A camera is a device that projects the 3D world onto a 2D image plane. Understanding this projection is essential because every downstream CV task — detection, segmentation, depth — operates on images produced by this process. The pinhole camera model is the simplest and most widely used abstraction: light from a scene point passes through a single point (the optical center) and hits the image plane.",
    },
    {
      title: "Pinhole Camera Model",
      content:
        "The pinhole model describes image formation as a perspective projection. A 3D point P = (X, Y, Z) in the camera coordinate system projects to a 2D point p = (u, v) on the image plane. The intrinsic matrix K encodes the camera's internal parameters: focal lengths (fx, fy) and principal point (cx, cy). The extrinsic parameters [R|t] describe the camera's pose in the world.",
      equations: [
        {
          label: "Perspective Projection",
          tex: "\\lambda \\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix} = K [R | t] \\begin{bmatrix} X \\\\ Y \\\\ Z \\\\ 1 \\end{bmatrix}",
        },
        {
          label: "Intrinsic Matrix",
          tex: "K = \\begin{bmatrix} f_x & 0 & c_x \\\\ 0 & f_y & c_y \\\\ 0 & 0 & 1 \\end{bmatrix}",
        },
        {
          label: "Homogeneous Coordinates",
          tex: "\\tilde{\\mathbf{p}} = \\lambda \\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix}, \\quad \\lambda > 0",
        },
      ],
    },
    {
      title: "Lens Distortion",
      content:
        "Real cameras use lenses that introduce geometric distortions. Radial distortion causes straight lines to appear curved (barrel or pincushion distortion). Tangential distortion occurs when the lens is not perfectly aligned with the sensor. Camera calibration estimates both intrinsic parameters and distortion coefficients using known calibration patterns (e.g., checkerboards).",
      equations: [
        {
          label: "Radial Distortion",
          tex: "\\begin{aligned} x_{\\text{dist}} &= x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) \\\\ y_{\\text{dist}} &= y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) \\end{aligned}",
        },
        {
          label: "Radial Distance",
          tex: "r^2 = x^2 + y^2",
        },
      ],
    },
    {
      title: "Camera Calibration (Zhang's Method)",
      content:
        "Zhang's method estimates intrinsic and extrinsic parameters from multiple images of a planar calibration pattern (checkerboard). Each view of the pattern provides a homography H relating world plane points to image points. From at least 3 views, the intrinsics K can be recovered by solving constraints on the image of the absolute conic (IAC). Radial distortion is then estimated via nonlinear optimization.",
      equations: [
        {
          label: "Homography Relation",
          tex: "\\tilde{\\mathbf{p}} = H \\tilde{\\mathbf{P}}, \\quad H = K [r_1 \\; r_2 \\; t]",
        },
        {
          label: "IAC Constraint",
          tex: "h_i^T \\omega h_j = 0, \\quad \\omega = K^{-T} K^{-1}",
        },
      ],
    },
    {
      title: "Image Formation & Sensor",
      content:
        "The digital image sensor (CCD or CMOS) converts photons into electrical signals. Each pixel integrates light over its area during the exposure time. The sensor introduces noise (shot noise, read noise, dark current). The Bayer filter pattern covers each pixel with a red, green, or blue filter; demosaicing reconstructs full-color images. Understanding these physical processes helps explain why CV models sometimes fail on noisy, over/under-exposed, or motion-blurred images.",
      equations: [
        {
          label: "Image Irradiance",
          tex: "E = \\frac{\\pi}{4} L \\left(\\frac{d}{f}\\right)^2 \\cos^4 \\theta",
        },
      ],
    },
    {
      title: "Real-World Applications",
      content:
        "Camera calibration is critical in autonomous driving (multi-camera rigs), augmented reality (camera-world alignment), robotics (hand-eye calibration), and 3D reconstruction (Structure from Motion requires known intrinsics). Stereo camera calibration additionally requires estimating the relative pose between two cameras for depth computation.",
    },
  ],
  algorithms: [
    {
      name: "Zhang's Calibration Pipeline",
      steps: [
        { step: "Capture Calibration Images", detail: "Take 10-20 images of a checkerboard at different orientations and distances" },
        { step: "Detect Corner Points", detail: "Use Harris or Shi-Tomasi corner detector to find checkerboard intersections" },
        { step: "Estimate Homographies", detail: "Compute homography H for each view relating pattern to image coordinates" },
        { step: "Solve for Intrinsics", detail: "Extract K from IAC constraints using at least 3 homographies" },
        { step: "Estimate Extrinsics", detail: "Decompose each H = K[r₁ r₂ t] to get per-view rotation and translation" },
        { step: "Refine with LM", detail: "Levenberg-Marquardt nonlinear optimization minimizing reprojection error" },
        { step: "Estimate Distortion", detail: "Fit radial distortion coefficients k₁, k₂, k₃ in final refinement" },
      ],
    },
  ],
  papers: [
    { year: 1999, title: "Zhang's Calibration", authors: "Zhang", venue: "ICCV", summary: "Flexible camera calibration from planar patterns. Foundation of OpenCV calibration." },
    { year: 2000, title: "Camera Calibration Toolbox", authors: "Bouguet", venue: "Caltech", summary: "Widely used MATLAB toolbox implementing Zhang's method with GUI." },
    { year: 2006, title: "OpenCV Calibration", authors: "Bradski & Kaehler", venue: "O'Reilly", summary: "Open-source implementation of camera calibration used worldwide." },
    { year: 2017, title: "Kalibr", authors: "Furgale et al.", venue: "IJRR", summary: "Multi-camera and camera-IMU calibration toolbox for robotics applications." },
  ],
};

// Scene Reasoning module content
export const sceneReasoningModule: ModuleContent = {
  id: "scene-reasoning",
  title: "Scene Reasoning with Multimodal LLMs",
  subtitle: "Go beyond recognition — use large vision-language models to understand, reason about, and describe complex visual scenes.",
  color: "290 70% 55%",
  theory: [
    {
      title: "Intuition",
      content:
        "Traditional CV solves narrow tasks (detect objects, segment regions). Scene reasoning asks higher-level questions: 'What is happening in this scene?', 'Why is the person running?', 'What will happen next?' Multimodal LLMs bridge vision and language, enabling open-ended visual reasoning by combining powerful language models with visual encoders.",
    },
    {
      title: "Vision-Language Pre-training",
      content:
        "Models like CLIP learn aligned vision-language representations by training on millions of image-text pairs. The key insight: use contrastive learning to pull matching image-text embeddings together and push non-matching ones apart. This produces a shared embedding space where images and text can be directly compared. CLIP's zero-shot capabilities emerge from learning to align visual concepts with natural language descriptions.",
      equations: [
        {
          label: "CLIP Contrastive Loss",
          tex: "\\mathcal{L} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{\\exp(\\text{sim}(v_i, t_i)/\\tau)}{\\sum_{j=1}^{N} \\exp(\\text{sim}(v_i, t_j)/\\tau)}",
        },
        {
          label: "Cosine Similarity",
          tex: "\\text{sim}(\\mathbf{v}, \\mathbf{t}) = \\frac{\\mathbf{v} \\cdot \\mathbf{t}}{\\|\\mathbf{v}\\| \\|\\mathbf{t}\\|}",
        },
      ],
    },
    {
      title: "Visual Grounding & Referring Expressions",
      content:
        "Visual grounding localizes objects or regions described by natural language (e.g., 'the red car behind the tree'). This requires understanding spatial relationships, attributes, and context. Models like MDETR extend DETR with text conditioning, while Florence-2 uses a unified sequence-to-sequence approach for multiple vision tasks including grounding, captioning, and detection.",
      equations: [
        {
          label: "Grounding Objective",
          tex: "\\hat{b} = \\arg\\max_{b \\in \\mathcal{B}} P(b | I, \\text{query})",
        },
      ],
    },
    {
      title: "Large Vision-Language Models (LVLMs)",
      content:
        "LVLMs like GPT-4V, Gemini, and LLaVA combine frozen or fine-tuned vision encoders with large language models. The vision encoder produces visual tokens that are projected into the LLM's embedding space. The LLM then processes these visual tokens alongside text tokens, enabling complex visual reasoning, multi-turn dialogue about images, and instruction following. These models can perform OCR, spatial reasoning, counting, and even basic physics understanding.",
      equations: [
        {
          label: "Visual Token Projection",
          tex: "\\mathbf{z}_v = W_p \\cdot \\text{ViT}(I) + \\mathbf{b}_p",
        },
        {
          label: "Multimodal Input",
          tex: "\\mathbf{h} = \\text{LLM}([\\mathbf{z}_v; \\mathbf{z}_t])",
        },
      ],
    },
    {
      title: "Florence-2: Unified Vision Foundation Model",
      content:
        "Florence-2 is a vision foundation model that handles multiple tasks (captioning, detection, segmentation, grounding, OCR) with a single sequence-to-sequence architecture. Instead of task-specific heads, it formats all outputs as text sequences including coordinates. This unification enables training on diverse datasets and zero-shot transfer to new tasks.",
      equations: [
        {
          label: "Sequence-to-Sequence Formulation",
          tex: "y = \\text{Decoder}(\\text{Encoder}(I, \\text{task\\_prompt}))",
        },
      ],
    },
    {
      title: "Real-World Applications",
      content:
        "Scene reasoning powers visual assistants for the blind (describing environments), autonomous driving (understanding traffic situations), content moderation (detecting harmful content with context), medical imaging (report generation from X-rays), and robotics (task planning from visual observations). The ability to combine perception with language understanding opens up applications that were previously impossible with fixed-vocabulary classification.",
    },
  ],
  algorithms: [
    {
      name: "Florence-2 Inference Pipeline",
      steps: [
        { step: "Image Encoding", detail: "DaViT vision encoder produces multi-scale image features" },
        { step: "Task Prompt Formatting", detail: "Convert task (e.g., '<OD>') into text tokens as decoder prompt" },
        { step: "Cross-Attention Fusion", detail: "Decoder attends to visual features conditioned on task prompt" },
        { step: "Autoregressive Decoding", detail: "Generate output tokens (text + coordinates) sequentially" },
        { step: "Post-processing", detail: "Parse output tokens into structured results (boxes, masks, captions)" },
      ],
    },
  ],
  papers: [
    { year: 2021, title: "CLIP", authors: "Radford et al.", venue: "ICML", summary: "Contrastive Language-Image Pretraining for zero-shot visual recognition." },
    { year: 2022, title: "Flamingo", authors: "Alayrac et al.", venue: "NeurIPS", summary: "Few-shot visual language model with interleaved image-text inputs." },
    { year: 2023, title: "LLaVA", authors: "Liu et al.", venue: "NeurIPS", summary: "Visual instruction tuning connecting CLIP to LLaMA for multimodal reasoning." },
    { year: 2023, title: "Florence-2", authors: "Xiao et al.", venue: "CVPR", summary: "Unified vision foundation model for multiple tasks via sequence-to-sequence." },
    { year: 2023, title: "GPT-4V", authors: "OpenAI", venue: "arXiv", summary: "Multimodal GPT-4 with vision capabilities for complex visual reasoning." },
    { year: 2024, title: "Gemini", authors: "Google DeepMind", venue: "arXiv", summary: "Natively multimodal model processing text, images, audio, and video." },
  ],
};
