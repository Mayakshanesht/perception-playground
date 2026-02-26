export interface TheorySection {
  title: string;
  content: string; // markdown-ish text
  equations?: { label: string; tex: string }[];
}

export interface PaperEntry {
  year: number;
  title: string;
  authors: string;
  venue: string;
  summary: string;
}

export interface AlgorithmStep {
  step: string;
  detail: string;
}

export interface PlaygroundConfig {
  title: string;
  description: string;
  taskType: string;
  acceptVideo?: boolean;
  acceptImage?: boolean;
  modelName?: string;
  learningFocus?: string;
}

export interface ModuleContent {
  id: string;
  title: string;
  subtitle: string;
  color: string;
  theory: TheorySection[];
  algorithms: { name: string; steps: AlgorithmStep[] }[];
  papers: PaperEntry[];
  playground?: PlaygroundConfig;
  playgrounds?: PlaygroundConfig[];
}

export const moduleContents: Record<string, ModuleContent> = {
  classification: {
    id: "classification",
    title: "Image Classification",
    subtitle: "Assign a single label to an entire image — the foundational task that launched the deep learning revolution in computer vision.",
    color: "187 85% 53%",
    theory: [
      {
        title: "What is Image Classification?",
        content: "Image classification is the task of assigning a categorical label to an entire input image from a fixed set of categories. Given an image x ∈ ℝ^(H×W×C), the goal is to learn a function f(x) → y where y ∈ {1, 2, ..., K} represents one of K predefined classes. This is a supervised learning problem where we train on a dataset D = {(x_i, y_i)} of labeled image-label pairs.",
        equations: [
          { label: "Softmax Classifier", tex: "P(y=k \\mid \\mathbf{x}) = \\frac{e^{\\mathbf{w}_k^T \\mathbf{x} + b_k}}{\\sum_{j=1}^{K} e^{\\mathbf{w}_j^T \\mathbf{x} + b_j}}" },
          { label: "Cross-Entropy Loss", tex: "\\mathcal{L} = -\\sum_{i=1}^{N} \\sum_{k=1}^{K} y_{ik} \\log \\hat{y}_{ik}" },
        ],
      },
      {
        title: "Convolutional Neural Networks (CNNs)",
        content: "CNNs exploit spatial locality and translation equivariance through learned convolutional filters. A convolutional layer applies a set of learnable kernels W ∈ ℝ^(k×k×C_in×C_out) that slide across the input feature map, computing dot products at each spatial location. Key innovations include weight sharing (same filter applied everywhere), local receptive fields, and hierarchical feature extraction from edges → textures → parts → objects.",
        equations: [
          { label: "Convolution Operation", tex: "(f * g)(i, j) = \\sum_{m} \\sum_{n} f(m, n) \\cdot g(i-m, j-n)" },
          { label: "Output Size", tex: "O = \\frac{W - K + 2P}{S} + 1" },
          { label: "ReLU Activation", tex: "\\text{ReLU}(x) = \\max(0, x)" },
        ],
      },
      {
        title: "Batch Normalization",
        content: "Batch normalization normalizes the pre-activations within each mini-batch, reducing internal covariate shift. This allows using higher learning rates and acts as a regularizer. For each channel, BN computes the mean and variance across the batch and spatial dimensions, then applies a learned affine transform.",
        equations: [
          { label: "Batch Norm Transform", tex: "\\hat{x}_i = \\frac{x_i - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}}, \\quad y_i = \\gamma \\hat{x}_i + \\beta" },
        ],
      },
      {
        title: "Residual Learning (ResNet)",
        content: "ResNet introduced skip (shortcut) connections that allow gradients to flow directly through identity mappings. Instead of learning H(x) directly, residual blocks learn F(x) = H(x) - x, making it easier to optimize very deep networks (50, 101, 152+ layers). The identity shortcut means the network only needs to learn the residual, and if the optimal mapping is close to identity, it's easier to push F(x) → 0 than to fit an identity mapping with stacked nonlinear layers.",
        equations: [
          { label: "Residual Block", tex: "\\mathbf{y} = \\mathcal{F}(\\mathbf{x}, \\{W_i\\}) + \\mathbf{x}" },
          { label: "Bottleneck Block", tex: "\\mathbf{y} = W_3 \\cdot \\sigma(W_2 \\cdot \\sigma(W_1 \\cdot \\mathbf{x})) + W_s \\cdot \\mathbf{x}" },
        ],
      },
      {
        title: "Vision Transformers (ViT)",
        content: "Vision Transformers split the image into fixed-size patches (e.g., 16×16), linearly embed each patch, add positional embeddings, and feed the resulting sequence into a standard Transformer encoder. The [CLS] token's output is used for classification. ViT demonstrates that with sufficient data (JFT-300M, ImageNet-21k), pure attention can match or exceed CNN performance without any convolution.",
        equations: [
          { label: "Patch Embedding", tex: "\\mathbf{z}_0 = [\\mathbf{x}_{\\text{class}}; \\mathbf{x}_p^1 E; \\mathbf{x}_p^2 E; \\ldots; \\mathbf{x}_p^N E] + \\mathbf{E}_{\\text{pos}}" },
          { label: "Self-Attention", tex: "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V" },
          { label: "Multi-Head Attention", tex: "\\text{MHA}(Q,K,V) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h)W^O" },
        ],
      },
    ],
    algorithms: [
      {
        name: "CNN Training Pipeline",
        steps: [
          { step: "Data Augmentation", detail: "Random crop, horizontal flip, color jitter, mixup/cutmix" },
          { step: "Forward Pass", detail: "Input → Conv blocks → Pooling → FC → Softmax probabilities" },
          { step: "Loss Computation", detail: "Cross-entropy between predicted and ground-truth distributions" },
          { step: "Backpropagation", detail: "Compute gradients via chain rule through all layers" },
          { step: "Optimizer Step", detail: "SGD with momentum, Adam, or AdamW weight update" },
          { step: "Learning Rate Schedule", detail: "Cosine annealing, warm-up, or step decay" },
        ],
      },
    ],
    papers: [
      { year: 2012, title: "AlexNet", authors: "Krizhevsky, Sutskever, Hinton", venue: "NeurIPS", summary: "Won ImageNet 2012 with deep CNN, launching the deep learning era. Introduced ReLU, dropout, and GPU training." },
      { year: 2014, title: "VGGNet", authors: "Simonyan & Zisserman", venue: "ICLR", summary: "Showed depth matters: 16-19 layers of 3×3 convolutions. Simple, uniform architecture. VGG-16 had 138M parameters." },
      { year: 2014, title: "GoogLeNet / Inception", authors: "Szegedy et al.", venue: "CVPR", summary: "Inception module with parallel 1×1, 3×3, 5×5 convolutions. Only 6.8M params via factorization." },
      { year: 2015, title: "ResNet", authors: "He et al.", venue: "CVPR", summary: "Skip connections enabling 152-layer networks. Won ImageNet with 3.57% top-5 error. Revolutionized deep learning." },
      { year: 2016, title: "DenseNet", authors: "Huang et al.", venue: "CVPR", summary: "Dense connections: every layer connects to every other. Feature reuse and strong gradient flow." },
      { year: 2017, title: "SENet", authors: "Hu et al.", venue: "CVPR", summary: "Squeeze-and-Excitation blocks: channel attention mechanism recalibrating feature responses." },
      { year: 2019, title: "EfficientNet", authors: "Tan & Le", venue: "ICML", summary: "Compound scaling of depth, width, and resolution. State-of-the-art efficiency on ImageNet." },
      { year: 2021, title: "ViT", authors: "Dosovitskiy et al.", venue: "ICLR", summary: "Pure transformer for images. Patch embeddings + self-attention achieves excellent results with large-scale pre-training." },
      { year: 2021, title: "Swin Transformer", authors: "Liu et al.", venue: "ICCV", summary: "Hierarchical vision transformer with shifted windows. Linear complexity and strong across vision tasks." },
    ],
    playground: {
      title: "Image Classification Playground",
      description: "Upload an image to classify it using Google's ViT (Vision Transformer) model pre-trained on ImageNet-1K. The model outputs top-5 predicted labels with confidence scores.",
      taskType: "image-classification",
      acceptImage: true,
      modelName: "google/vit-base-patch16-224",
      learningFocus: "Observe confidence calibration across similar classes and test robustness to viewpoint/background changes.",
    },
  },

  detection: {
    id: "detection",
    title: "Object Detection",
    subtitle: "Localize and classify multiple objects in an image — the bridge between image understanding and scene comprehension.",
    color: "160 84% 39%",
    theory: [
      {
        title: "What is Object Detection?",
        content: "Object detection combines classification and localization: for each object in an image, predict its class label AND bounding box coordinates. The output is a set of tuples {(c_i, b_i, s_i)} where c is the class, b = (x, y, w, h) is the bounding box, and s is the confidence score. This is fundamentally harder than classification because the number of objects varies per image, and the model must handle multiple scales, aspect ratios, and occlusions.",
        equations: [
          { label: "Bounding Box Parameterization", tex: "b = (b_x, b_y, b_w, b_h) \\in \\mathbb{R}^4" },
          { label: "Intersection over Union (IoU)", tex: "\\text{IoU}(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}" },
        ],
      },
      {
        title: "Two-Stage Detectors: R-CNN Family",
        content: "Two-stage detectors decouple object detection into (1) generating class-agnostic region proposals and (2) classifying and refining each proposal. R-CNN used Selective Search to generate ~2000 proposals, warped each to 227×227, and ran through AlexNet. Fast R-CNN shared computation by running the CNN once on the entire image and using RoI pooling. Faster R-CNN replaced Selective Search with a learned Region Proposal Network (RPN) that shares convolutional features with the detection head.",
        equations: [
          { label: "RPN Objectness Loss", tex: "L_{cls}(p_i, p_i^*) = -p_i^* \\log p_i - (1 - p_i^*) \\log(1 - p_i)" },
          { label: "Smooth L1 Loss", tex: "L_{reg}(t, t^*) = \\sum_{i \\in \\{x,y,w,h\\}} \\text{smooth}_{L_1}(t_i - t_i^*)" },
          { label: "Faster R-CNN Multi-task Loss", tex: "L = \\frac{1}{N_{cls}} \\sum_i L_{cls}(p_i, p_i^*) + \\lambda \\frac{1}{N_{reg}} \\sum_i p_i^* L_{reg}(t_i, t_i^*)" },
        ],
      },
      {
        title: "One-Stage Detectors: YOLO & SSD",
        content: "One-stage detectors skip the proposal generation step and directly predict bounding boxes and class probabilities from the feature map in a single forward pass. YOLO divides the image into an S×S grid; each cell predicts B bounding boxes with confidence and C class probabilities. SSD uses multi-scale feature maps from different layers for detecting objects at various sizes. RetinaNet addressed the class imbalance issue in one-stage detectors with Focal Loss.",
        equations: [
          { label: "YOLO Grid Prediction", tex: "\\hat{y} = (b_x, b_y, b_w, b_h, p_{obj}, c_1, c_2, \\ldots, c_K)" },
          { label: "Focal Loss (RetinaNet)", tex: "\\text{FL}(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)" },
          { label: "YOLO Box Regression", tex: "b_x = \\sigma(t_x) + c_x, \\quad b_w = p_w e^{t_w}" },
        ],
      },
      {
        title: "Non-Maximum Suppression (NMS)",
        content: "After prediction, many overlapping boxes may fire for the same object. NMS is a post-processing step that: (1) sorts all detections by confidence score, (2) selects the highest-scoring box, (3) removes all boxes with IoU > threshold with the selected box, (4) repeats until no boxes remain. Soft-NMS replaces the binary removal with a continuous decay of scores. DETR (Detection Transformer) eliminates NMS entirely via set-based Hungarian matching.",
        equations: [
          { label: "NMS Score Decay (Soft-NMS)", tex: "s_i = s_i \\cdot e^{-\\frac{\\text{IoU}(M, b_i)^2}{\\sigma}}" },
        ],
      },
      {
        title: "Feature Pyramid Networks (FPN)",
        content: "FPN addresses the multi-scale detection problem by building a top-down feature pyramid with lateral connections from the bottom-up backbone. Each pyramid level handles objects at a specific scale range. This allows the detector to leverage both semantically strong deep features and spatially precise shallow features. FPN is used in both Faster R-CNN (with RPN) and one-stage detectors like RetinaNet.",
        equations: [
          { label: "Lateral Connection", tex: "P_l = \\text{Conv}_{1 \\times 1}(C_l) + \\text{Upsample}(P_{l+1})" },
        ],
      },
    ],
    algorithms: [
      {
        name: "Faster R-CNN Pipeline",
        steps: [
          { step: "Backbone Feature Extraction", detail: "ResNet/VGG processes the entire image to produce convolutional feature maps" },
          { step: "FPN Multi-scale Features", detail: "Build feature pyramid P2-P5 with top-down pathway and lateral connections" },
          { step: "Region Proposal Network", detail: "3×3 conv sliding window predicts objectness + box regression for 9 anchors per location" },
          { step: "Proposal Filtering", detail: "Top-N proposals by score, NMS to reduce to ~300 proposals" },
          { step: "RoI Align", detail: "Bilinear interpolation to extract 7×7 feature maps from each proposal" },
          { step: "Classification & Regression Head", detail: "FC layers predict class scores and bounding box refinement" },
          { step: "Post-processing NMS", detail: "Per-class NMS to produce final detections" },
        ],
      },
      {
        name: "YOLOv3 Pipeline",
        steps: [
          { step: "Input Preprocessing", detail: "Resize image to 416×416, normalize pixel values" },
          { step: "Darknet-53 Backbone", detail: "53-layer backbone with residual connections extracts features at 3 scales" },
          { step: "Multi-scale Detection", detail: "Predictions at 13×13, 26×26, and 52×52 grids using 3 anchor boxes each" },
          { step: "Box Decoding", detail: "Apply sigmoid for center, exponential for size relative to anchor priors" },
          { step: "Confidence Thresholding", detail: "Filter predictions below objectness threshold (e.g., 0.5)" },
          { step: "NMS per Class", detail: "Remove duplicate detections with IoU > 0.45" },
        ],
      },
    ],
    papers: [
      { year: 2014, title: "R-CNN", authors: "Girshick et al.", venue: "CVPR", summary: "Introduced region-based CNN for object detection using selective search proposals." },
      { year: 2015, title: "Fast R-CNN", authors: "Girshick", venue: "ICCV", summary: "Shared convolutional features across proposals with RoI pooling for efficiency." },
      { year: 2015, title: "Faster R-CNN", authors: "Ren et al.", venue: "NeurIPS", summary: "Replaced selective search with a learned Region Proposal Network (RPN)." },
      { year: 2016, title: "YOLOv1", authors: "Redmon et al.", venue: "CVPR", summary: "First real-time single-stage detector treating detection as regression." },
      { year: 2016, title: "SSD", authors: "Liu et al.", venue: "ECCV", summary: "Multi-scale feature maps for single-shot detection at different resolutions." },
      { year: 2017, title: "FPN", authors: "Lin et al.", venue: "CVPR", summary: "Feature Pyramid Networks for multi-scale feature fusion in detection." },
      { year: 2017, title: "RetinaNet", authors: "Lin et al.", venue: "ICCV", summary: "Focal loss to address class imbalance in one-stage detectors." },
      { year: 2018, title: "YOLOv3", authors: "Redmon & Farhadi", venue: "arXiv", summary: "Multi-scale predictions with Darknet-53 backbone." },
      { year: 2020, title: "DETR", authors: "Carion et al.", venue: "ECCV", summary: "End-to-end detection with transformers, eliminating NMS and anchors." },
    ],
    playground: {
      title: "Object Detection Playground",
      description: "Upload an image to detect objects using Facebook's DETR (Detection Transformer) model with ResNet-50 backbone, trained on COCO.",
      taskType: "object-detection",
      acceptImage: true,
      modelName: "facebook/detr-resnet-50",
      learningFocus: "Compare detections across cluttered scenes and inspect how confidence and bounding boxes change with scale.",
    },
  },

  segmentation: {
    id: "segmentation",
    title: "Image Segmentation",
    subtitle: "Pixel-level scene understanding — assigning a label to every pixel in the image for precise spatial comprehension.",
    color: "265 70% 60%",
    theory: [
      {
        title: "Types of Segmentation",
        content: "Semantic segmentation assigns a class label to every pixel (all cars share one label). Instance segmentation separates individual object instances (car-1 vs car-2). Panoptic segmentation unifies both: stuff (sky, road) + things (individual cars, people). This hierarchy represents increasingly complete scene understanding.",
      },
      {
        title: "Fully Convolutional Networks (FCN)",
        content: "FCN replaced the fully connected layers of classification networks (VGG, AlexNet) with 1×1 convolutions, producing spatial predictions. Upsampling via transposed convolutions (learnable) or bilinear interpolation recovers spatial resolution. Skip connections from earlier layers provide fine-grained spatial details that are lost in deeper layers.",
        equations: [
          { label: "Transposed Convolution", tex: "y(i) = \\sum_k x\\left(\\frac{i + p - k}{s}\\right) \\cdot w(k)" },
          { label: "Pixel-wise Cross Entropy", tex: "\\mathcal{L} = -\\frac{1}{HW} \\sum_{i,j} \\sum_{c=1}^{C} y_{ijc} \\log \\hat{y}_{ijc}" },
        ],
      },
      {
        title: "U-Net Architecture",
        content: "U-Net features a symmetric encoder-decoder with skip connections that concatenate encoder features to decoder features at each level. The encoder captures context (what), the decoder enables precise localization (where). Originally designed for biomedical image segmentation with very few training images, U-Net's architecture became the gold standard for dense prediction tasks including medical imaging, satellite imagery, and autonomous driving.",
        equations: [
          { label: "Skip Connection", tex: "D_l = \\text{Conv}(\\text{Concat}(\\text{Up}(D_{l+1}), E_l))" },
          { label: "Dice Loss", tex: "\\mathcal{L}_{Dice} = 1 - \\frac{2|P \\cap G|}{|P| + |G|} = 1 - \\frac{2\\sum_i p_i g_i}{\\sum_i p_i + \\sum_i g_i}" },
        ],
      },
      {
        title: "Atrous/Dilated Convolutions (DeepLab)",
        content: "DeepLab introduced atrous (dilated) convolutions to increase the receptive field without reducing spatial resolution. Atrous Spatial Pyramid Pooling (ASPP) captures multi-scale context by applying parallel atrous convolutions with different dilation rates. DeepLab v3+ adds a decoder module for refined boundaries and uses depthwise separable convolutions for efficiency.",
        equations: [
          { label: "Dilated Convolution", tex: "y(i) = \\sum_{k=1}^{K} x(i + r \\cdot k) \\cdot w(k), \\quad r = \\text{dilation rate}" },
          { label: "Effective Receptive Field", tex: "RF_{\\text{eff}} = K + (K-1)(r-1)" },
        ],
      },
      {
        title: "Instance Segmentation: Mask R-CNN",
        content: "Mask R-CNN extends Faster R-CNN by adding a parallel branch that outputs a binary mask for each detected object. Key innovation: RoI Align (bilinear interpolation instead of quantized RoI Pooling) preserves spatial alignment. The mask head applies small FCN to each RoI independently, decoupling mask prediction from classification.",
        equations: [
          { label: "Mask R-CNN Loss", tex: "L = L_{cls} + L_{box} + L_{mask}" },
          { label: "Binary Mask Loss (per-pixel)", tex: "L_{mask} = -\\frac{1}{m^2} \\sum_{1 \\le i,j \\le m} [y_{ij} \\log \\hat{y}_{ij}^k + (1-y_{ij}) \\log(1-\\hat{y}_{ij}^k)]" },
        ],
      },
    ],
    algorithms: [
      {
        name: "U-Net Segmentation Pipeline",
        steps: [
          { step: "Encoder (Contracting Path)", detail: "Repeated 3×3 conv + ReLU + 2×2 max pool, doubling channels at each level" },
          { step: "Bottleneck", detail: "Deepest layer with maximum channels, captures global context" },
          { step: "Decoder (Expanding Path)", detail: "2×2 up-conv, concatenate with skip connection, 3×3 conv + ReLU" },
          { step: "Output Layer", detail: "1×1 convolution maps features to class probabilities per pixel" },
          { step: "Loss & Optimization", detail: "Dice loss + cross-entropy, Adam optimizer with learning rate warmup" },
        ],
      },
    ],
    papers: [
      { year: 2015, title: "FCN", authors: "Long, Shelhamer, Darrell", venue: "CVPR", summary: "First end-to-end trained model for pixel-wise prediction using fully convolutional layers." },
      { year: 2015, title: "U-Net", authors: "Ronneberger et al.", venue: "MICCAI", summary: "Encoder-decoder with skip connections for biomedical segmentation. Works with few training images." },
      { year: 2017, title: "Mask R-CNN", authors: "He et al.", venue: "ICCV", summary: "Added mask branch to Faster R-CNN for instance segmentation. Introduced RoI Align." },
      { year: 2017, title: "PSPNet", authors: "Zhao et al.", venue: "CVPR", summary: "Pyramid Pooling Module captures multi-scale context for semantic segmentation." },
      { year: 2018, title: "DeepLab v3+", authors: "Chen et al.", venue: "ECCV", summary: "Atrous convolutions + encoder-decoder for state-of-the-art semantic segmentation." },
      { year: 2019, title: "Panoptic FPN", authors: "Kirillov et al.", venue: "CVPR", summary: "Unified panoptic segmentation merging stuff and things predictions." },
      { year: 2021, title: "SegFormer", authors: "Xie et al.", venue: "NeurIPS", summary: "Transformer-based segmentation with hierarchical features and lightweight decoder." },
    ],
    playgrounds: [
      {
        title: "Panoptic Segmentation Playground",
        description: "Upload an image to run panoptic segmentation using Facebook's DETR model. Outputs both stuff (sky, road) and thing (car, person) segments.",
        taskType: "image-segmentation",
        acceptImage: true,
        modelName: "facebook/detr-resnet-50-panoptic",
        learningFocus: "Contrast panoptic labels with object detection results and inspect how small objects are grouped.",
      },
      {
        title: "SAM Playground (Mask Proposals)",
        description: "Upload an image to generate prompt-free mask proposals using Segment Anything.",
        taskType: "sam-segmentation",
        acceptImage: true,
        modelName: "facebook/sam-vit-base",
        learningFocus: "Compare how a foundation segmentation model proposes object masks without explicit class labels.",
      },
    ],
  },

  depth: {
    id: "depth",
    title: "Depth Estimation",
    subtitle: "Predict per-pixel distance from the camera — recovering 3D structure from 2D images for spatial understanding.",
    color: "32 95% 55%",
    theory: [
      {
        title: "Monocular Depth Estimation",
        content: "Monocular depth estimation predicts a dense depth map from a single RGB image. This is inherently an ill-posed problem since infinitely many 3D scenes can project to the same 2D image. Deep networks learn depth cues from monocular priors: texture gradients, occlusion, relative size, perspective, and atmospheric haze. The model learns a mapping f: ℝ^(H×W×3) → ℝ^(H×W) where each output pixel represents estimated depth or disparity.",
        equations: [
          { label: "Scale-Invariant Loss (Eigen et al.)", tex: "\\mathcal{L}_{SI} = \\frac{1}{n}\\sum_i d_i^2 - \\frac{\\lambda}{n^2}\\left(\\sum_i d_i\\right)^2, \\quad d_i = \\log \\hat{y}_i - \\log y_i" },
          { label: "Berhu Loss", tex: "B(x) = \\begin{cases} |x| & \\text{if } |x| \\le c \\\\ \\frac{x^2 + c^2}{2c} & \\text{if } |x| > c \\end{cases}" },
        ],
      },
      {
        title: "Stereo Vision & Disparity",
        content: "Stereo vision uses two cameras with known baseline to estimate depth via triangulation. For each pixel in the left image, find the corresponding pixel in the right image along the epipolar line. The horizontal displacement (disparity d) is inversely proportional to depth. Stereo matching algorithms include block matching, semi-global matching (SGM), and learned approaches like GC-Net and PSMNet.",
        equations: [
          { label: "Stereo Depth Formula", tex: "Z = \\frac{f \\cdot B}{d}" },
          { label: "Epipolar Constraint", tex: "\\mathbf{p}_R^T \\mathbf{F} \\mathbf{p}_L = 0" },
          { label: "Disparity Range", tex: "d = x_L - x_R, \\quad d \\in [d_{\\min}, d_{\\max}]" },
        ],
      },
      {
        title: "Self-Supervised Depth Learning",
        content: "Monodepth and Monodepth2 learn depth from monocular video without ground-truth depth labels. The key insight: predict depth + ego-motion, then use them to synthesize novel views. The photometric loss between the synthesized and actual views provides the training signal. This makes depth estimation possible with just unlabeled video data — critical for autonomous driving where LiDAR ground truth is expensive.",
        equations: [
          { label: "Photometric Reprojection Loss", tex: "\\mathcal{L}_p = \\alpha \\frac{1 - \\text{SSIM}(I_a, I_b)}{2} + (1-\\alpha) \\|I_a - I_b\\|_1" },
          { label: "View Synthesis", tex: "p_s \\sim K T_{t \\to s} D_t(p_t) K^{-1} p_t" },
        ],
      },
      {
        title: "Depth with Transformers (DPT, MiDaS)",
        content: "Dense Prediction Transformers (DPT) use ViT as a backbone for depth estimation, leveraging global attention to capture long-range dependencies that CNNs miss. MiDaS achieves zero-shot monocular depth by training on 12 diverse datasets simultaneously with a scale-and-shift invariant loss. This produces relative depth maps that generalize across domains (indoor, outdoor, art).",
        equations: [
          { label: "Scale-Shift Invariant Loss", tex: "\\mathcal{L} = \\frac{1}{2n} \\sum_i \\left(\\frac{\\hat{d}_i - \\text{median}(\\hat{d})}{\\text{MAD}(\\hat{d})} - \\frac{d_i - \\text{median}(d)}{\\text{MAD}(d)}\\right)^2" },
        ],
      },
    ],
    algorithms: [
      {
        name: "Monocular Depth Pipeline",
        steps: [
          { step: "Image Preprocessing", detail: "Resize, normalize, apply standard ImageNet transforms" },
          { step: "Encoder (Backbone)", detail: "ResNet, EfficientNet, or ViT extracts multi-scale features" },
          { step: "Decoder", detail: "Progressive upsampling with skip connections to produce dense depth" },
          { step: "Output Head", detail: "Sigmoid or ReLU activation, scale to depth range [d_min, d_max]" },
          { step: "Post-processing", detail: "Bilateral filtering, edge-aware refinement, median scaling" },
        ],
      },
    ],
    papers: [
      { year: 2014, title: "Eigen et al.", authors: "Eigen, Puhrsch, Fergus", venue: "NeurIPS", summary: "First deep monocular depth estimation with multi-scale CNN. Scale-invariant error metric." },
      { year: 2016, title: "Monodepth", authors: "Godard et al.", venue: "CVPR", summary: "Self-supervised depth from stereo pairs using left-right consistency." },
      { year: 2019, title: "Monodepth2", authors: "Godard et al.", venue: "ICCV", summary: "Improved self-supervised depth from monocular video with minimum reprojection loss." },
      { year: 2020, title: "MiDaS", authors: "Ranftl et al.", venue: "TPAMI", summary: "Robust monocular depth via multi-dataset training with mixing strategy." },
      { year: 2021, title: "DPT", authors: "Ranftl et al.", venue: "ICCV", summary: "Dense Prediction Transformer using ViT backbone for depth estimation." },
      { year: 2023, title: "Depth Anything", authors: "Yang et al.", venue: "CVPR", summary: "Foundation model for monocular depth using 62M unlabeled images." },
    ],
    playground: {
      title: "Depth Estimation Playground",
      description: "Upload an image to estimate depth using Intel's DPT-Large model. The output is a relative depth map visualized as a heatmap.",
      taskType: "depth-estimation",
      acceptImage: true,
      modelName: "Intel/dpt-large",
      learningFocus: "Test indoor vs outdoor scenes and inspect where monocular depth is uncertain (transparent/reflective regions).",
    },
  },

  sfm: {
    id: "sfm",
    title: "Structure from Motion",
    subtitle: "Reconstruct 3D scene structure and camera poses from 2D image sequences — the geometry engine behind 3D mapping.",
    color: "340 75% 55%",
    theory: [
      {
        title: "Feature Detection & Matching",
        content: "SfM begins with finding corresponding points across images. Feature detectors (SIFT, SURF, ORB, SuperPoint) identify distinctive keypoints invariant to scale, rotation, and illumination changes. Feature descriptors encode local image appearance around each keypoint as a high-dimensional vector. Matching finds pairs of descriptors with smallest distance, filtered by Lowe's ratio test (reject matches where the best match is too similar to the second-best).",
        equations: [
          { label: "SIFT Scale Space", tex: "L(x, y, \\sigma) = G(x, y, \\sigma) * I(x, y)" },
          { label: "Difference of Gaussians", tex: "D(x, y, \\sigma) = L(x, y, k\\sigma) - L(x, y, \\sigma)" },
          { label: "Lowe's Ratio Test", tex: "\\frac{\\|d_1 - d_2\\|}{\\|d_1 - d_3\\|} < \\tau \\quad (\\tau \\approx 0.75)" },
        ],
      },
      {
        title: "Epipolar Geometry",
        content: "The epipolar constraint relates corresponding points between two views. Given a point x in image 1, its match x' in image 2 must lie on the epipolar line l' = Fx. The fundamental matrix F encodes this relationship for uncalibrated cameras; the essential matrix E = K'^T F K does so for calibrated cameras. F has 7 DOF (3×3 matrix with rank-2 and det=0 constraints).",
        equations: [
          { label: "Epipolar Constraint", tex: "\\mathbf{x'}^T F \\mathbf{x} = 0" },
          { label: "Essential Matrix", tex: "E = [\\mathbf{t}]_\\times R = K'^T F K" },
          { label: "Decomposition", tex: "E = U \\text{diag}(1,1,0) V^T \\Rightarrow R, \\mathbf{t}" },
        ],
      },
      {
        title: "Triangulation",
        content: "Given known camera poses and corresponding 2D points, triangulation recovers the 3D point. In practice, rays from two cameras don't intersect exactly due to noise, so we solve a least-squares problem. The Direct Linear Transform (DLT) method sets up a system AX = 0 and solves via SVD. Mid-point and optimal triangulation methods also exist.",
        equations: [
          { label: "Projection Equation", tex: "\\lambda \\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix} = P \\begin{bmatrix} X \\\\ Y \\\\ Z \\\\ 1 \\end{bmatrix} = K[R|t] \\mathbf{X}" },
          { label: "DLT System", tex: "A \\mathbf{X} = 0, \\quad \\mathbf{X}^* = \\arg\\min_{\\|\\mathbf{X}\\|=1} \\|A\\mathbf{X}\\|^2" },
        ],
      },
      {
        title: "Bundle Adjustment",
        content: "Bundle adjustment is the gold standard for refining 3D structure and camera parameters simultaneously. It minimizes the reprojection error — the sum of squared distances between observed 2D points and their predicted projections. This is a large-scale nonlinear least-squares problem solved with Levenberg-Marquardt. The Jacobian has a sparse block structure that can be exploited for efficiency (Schur complement).",
        equations: [
          { label: "Reprojection Error", tex: "\\min_{R_j, t_j, X_i} \\sum_{i,j} \\| \\mathbf{x}_{ij} - \\pi(R_j, t_j, X_i) \\|^2" },
          { label: "Levenberg-Marquardt Update", tex: "(J^T J + \\lambda I) \\delta = -J^T \\mathbf{r}" },
        ],
      },
    ],
    algorithms: [
      {
        name: "Incremental SfM (COLMAP)",
        steps: [
          { step: "Feature Extraction", detail: "Detect SIFT keypoints and compute 128-d descriptors for all images" },
          { step: "Feature Matching", detail: "Exhaustive or vocabulary tree matching with ratio test filtering" },
          { step: "Geometric Verification", detail: "RANSAC with fundamental/essential matrix to filter outlier matches" },
          { step: "Initialize Two-View Reconstruction", detail: "Select best image pair, estimate E, triangulate initial points" },
          { step: "Image Registration (PnP)", detail: "Add new images by solving Perspective-n-Point with known 3D-2D correspondences" },
          { step: "Triangulation", detail: "Add new 3D points from newly registered image pairs" },
          { step: "Bundle Adjustment", detail: "Jointly optimize all cameras and 3D points (after every N images)" },
        ],
      },
    ],
    papers: [
      { year: 2004, title: "SIFT", authors: "Lowe", venue: "IJCV", summary: "Scale-Invariant Feature Transform for robust keypoint detection and description." },
      { year: 2006, title: "Photo Tourism", authors: "Snavely et al.", venue: "SIGGRAPH", summary: "SfM from unstructured photo collections. Pioneered large-scale 3D reconstruction from internet photos." },
      { year: 2011, title: "ORB", authors: "Rublee et al.", venue: "ICCV", summary: "Oriented FAST and Rotated BRIEF: fast binary feature descriptor." },
      { year: 2016, title: "COLMAP", authors: "Schönberger & Frahm", venue: "CVPR", summary: "State-of-the-art SfM pipeline with robust incremental reconstruction." },
      { year: 2020, title: "SuperPoint + SuperGlue", authors: "Sarlin et al.", venue: "CVPR", summary: "Learned keypoint detection and matching with graph neural networks." },
    ],
  },

  nerf: {
    id: "nerf",
    title: "Neural Rendering",
    subtitle: "Synthesize photorealistic novel views using neural scene representations — where geometry meets deep learning.",
    color: "200 80% 55%",
    theory: [
      {
        title: "Neural Radiance Fields (NeRF)",
        content: "NeRF represents a 3D scene as a continuous volumetric function parameterized by a neural network: F: (x, y, z, θ, φ) → (c, σ), mapping 3D position and viewing direction to color and volume density. Novel views are rendered by casting rays through each pixel and integrating color and density along the ray using volume rendering. The network is trained by minimizing the photometric loss between rendered and ground-truth images.",
        equations: [
          { label: "NeRF MLP", tex: "F_\\Theta: (\\mathbf{x}, \\mathbf{d}) \\to (\\mathbf{c}, \\sigma)" },
          { label: "Volume Rendering Integral", tex: "C(\\mathbf{r}) = \\int_{t_n}^{t_f} T(t) \\cdot \\sigma(\\mathbf{r}(t)) \\cdot \\mathbf{c}(\\mathbf{r}(t), \\mathbf{d}) \\, dt" },
          { label: "Transmittance", tex: "T(t) = \\exp\\left(-\\int_{t_n}^{t} \\sigma(\\mathbf{r}(s)) \\, ds\\right)" },
          { label: "Discrete Approximation", tex: "\\hat{C}(\\mathbf{r}) = \\sum_{i=1}^{N} T_i (1 - e^{-\\sigma_i \\delta_i}) \\mathbf{c}_i" },
        ],
      },
      {
        title: "Positional Encoding",
        content: "MLPs have spectral bias — they tend to learn low-frequency functions. Positional encoding maps low-dimensional coordinates to high-dimensional space using sinusoidal functions, enabling the network to learn high-frequency details (sharp edges, fine textures). This was the key insight that made NeRF work well.",
        equations: [
          { label: "Positional Encoding", tex: "\\gamma(p) = \\left(\\sin(2^0\\pi p), \\cos(2^0\\pi p), \\ldots, \\sin(2^{L-1}\\pi p), \\cos(2^{L-1}\\pi p)\\right)" },
        ],
      },
      {
        title: "3D Gaussian Splatting",
        content: "3DGS represents scenes as millions of 3D Gaussians, each with position (mean μ), covariance Σ, color (spherical harmonics), and opacity α. Unlike NeRF, rendering is done via rasterization: each Gaussian is projected (splatted) onto the image plane and alpha-composited front-to-back. This enables real-time (>100 FPS) rendering while maintaining quality comparable to NeRF. Training optimizes Gaussian parameters via gradient descent with adaptive density control (splitting/cloning/pruning).",
        equations: [
          { label: "3D Gaussian", tex: "G(\\mathbf{x}) = e^{-\\frac{1}{2}(\\mathbf{x}-\\boldsymbol{\\mu})^T \\Sigma^{-1} (\\mathbf{x}-\\boldsymbol{\\mu})}" },
          { label: "Covariance Decomposition", tex: "\\Sigma = R S S^T R^T" },
          { label: "2D Projection", tex: "\\Sigma' = J W \\Sigma W^T J^T" },
          { label: "Alpha Compositing", tex: "C = \\sum_{i=1}^{N} c_i \\alpha_i \\prod_{j=1}^{i-1}(1 - \\alpha_j)" },
        ],
      },
    ],
    algorithms: [
      {
        name: "NeRF Training Pipeline",
        steps: [
          { step: "Input Preparation", detail: "Calibrated images + camera poses (from COLMAP SfM)" },
          { step: "Ray Generation", detail: "For each pixel, compute ray origin and direction in world coordinates" },
          { step: "Stratified Sampling", detail: "Sample N_c points along each ray (coarse), then importance-sample N_f more (fine)" },
          { step: "Network Query", detail: "Evaluate MLP at each sample point: (γ(x), γ(d)) → (c, σ)" },
          { step: "Volume Rendering", detail: "Integrate colors and densities along ray to produce pixel color" },
          { step: "Loss & Backprop", detail: "MSE between rendered and ground-truth pixel colors" },
        ],
      },
    ],
    papers: [
      { year: 2020, title: "NeRF", authors: "Mildenhall et al.", venue: "ECCV", summary: "Neural radiance fields for novel view synthesis. Continuous volumetric scene representation with MLPs." },
      { year: 2022, title: "Instant NGP", authors: "Müller et al.", venue: "SIGGRAPH", summary: "Multi-resolution hash encoding for real-time NeRF training (seconds instead of hours)." },
      { year: 2022, title: "Mip-NeRF 360", authors: "Barron et al.", venue: "CVPR", summary: "Anti-aliased NeRF for unbounded 360° scenes with integrated positional encoding." },
      { year: 2023, title: "3D Gaussian Splatting", authors: "Kerbl et al.", venue: "SIGGRAPH", summary: "Real-time radiance field rendering with 3D Gaussians and differentiable rasterization." },
      { year: 2024, title: "4D Gaussian Splatting", authors: "Wu et al.", venue: "CVPR", summary: "Extension to dynamic scenes with temporal Gaussian deformation." },
    ],
  },

  pose: {
    id: "pose",
    title: "Pose Estimation",
    subtitle: "Detect and localize human body keypoints in 2D and 3D — fundamental for understanding human actions and interactions.",
    color: "50 90% 50%",
    theory: [
      {
        title: "2D Pose Estimation",
        content: "2D pose estimation predicts the locations of anatomical keypoints (joints) in pixel coordinates: nose, eyes, shoulders, elbows, wrists, hips, knees, ankles. Top-down methods first detect people with a bounding box detector, then estimate pose within each box. Bottom-up methods detect all keypoints first, then group them into individuals. The output is a set of K keypoints, each with (x, y, confidence).",
        equations: [
          { label: "Heatmap Prediction", tex: "\\hat{H}_k(x,y) = \\exp\\left(-\\frac{(x - x_k)^2 + (y - y_k)^2}{2\\sigma^2}\\right)" },
          { label: "MSE Heatmap Loss", tex: "\\mathcal{L} = \\frac{1}{K} \\sum_{k=1}^{K} \\|H_k - \\hat{H}_k\\|^2" },
        ],
      },
      {
        title: "High-Resolution Networks (HRNet)",
        content: "HRNet maintains high-resolution representations throughout the network by connecting multi-resolution subnetworks in parallel. Unlike encoder-decoder architectures that lose resolution and recover it, HRNet never drops below a certain resolution. Multi-scale fusion via repeated exchange of information across resolutions produces spatially precise and semantically rich features.",
        equations: [
          { label: "Multi-Resolution Fusion", tex: "\\mathbf{x}_r^s = \\sum_{i=1}^{S} a_{ri}(\\mathbf{x}_i^{s-1})" },
        ],
      },
      {
        title: "3D Pose Estimation (Lifting)",
        content: "3D pose estimation recovers the (x, y, z) coordinates of joints in 3D space. Two approaches: (1) Direct regression from images to 3D coordinates, (2) 2D-to-3D lifting — first predict 2D pose, then lift to 3D using a learned mapping. The lifting approach is popular because 2D pose estimation is more mature and can leverage existing detectors. Challenges include depth ambiguity and the need for 3D ground truth (from MoCap systems).",
        equations: [
          { label: "3D Lifting", tex: "\\mathbf{X}_{3D} = f_{\\theta}(\\mathbf{x}_{2D}), \\quad f: \\mathbb{R}^{K \\times 2} \\to \\mathbb{R}^{K \\times 3}" },
          { label: "MPJPE Metric", tex: "\\text{MPJPE} = \\frac{1}{K} \\sum_{k=1}^{K} \\|\\hat{\\mathbf{X}}_k - \\mathbf{X}_k\\|_2" },
        ],
      },
      {
        title: "Part Affinity Fields (OpenPose)",
        content: "OpenPose is a bottom-up multi-person pose estimation method. It simultaneously predicts (1) confidence maps for keypoint locations and (2) Part Affinity Fields (PAFs) — 2D vector fields that encode the association between keypoints (e.g., which elbow connects to which wrist). A greedy bipartite matching algorithm then assembles detected keypoints into individual skeletons.",
        equations: [
          { label: "Part Affinity Field", tex: "\\mathbf{L}_c^*(\\mathbf{p}) = \\begin{cases} \\mathbf{v} & \\text{if } \\mathbf{p} \\text{ on limb } c \\\\ \\mathbf{0} & \\text{otherwise} \\end{cases}" },
          { label: "Limb Score", tex: "E = \\int_0^1 \\mathbf{L}_c(\\mathbf{p}(u)) \\cdot \\frac{\\mathbf{d}_{j_2} - \\mathbf{d}_{j_1}}{\\|\\mathbf{d}_{j_2} - \\mathbf{d}_{j_1}\\|_2} du" },
        ],
      },
    ],
    algorithms: [
      {
        name: "Top-Down Pose Estimation",
        steps: [
          { step: "Person Detection", detail: "Run object detector (Faster R-CNN) to get person bounding boxes" },
          { step: "Crop & Resize", detail: "Extract and resize each person crop to fixed size (256×192)" },
          { step: "Backbone Extraction", detail: "HRNet or ResNet extracts multi-scale features" },
          { step: "Heatmap Regression", detail: "Predict K heatmaps (one per keypoint), each representing likelihood" },
          { step: "Keypoint Decoding", detail: "Argmax + sub-pixel refinement to get keypoint coordinates" },
        ],
      },
    ],
    papers: [
      { year: 2016, title: "Convolutional Pose Machines", authors: "Wei et al.", venue: "CVPR", summary: "Sequential convolutional architecture with intermediate supervision for pose estimation." },
      { year: 2017, title: "OpenPose", authors: "Cao et al.", venue: "CVPR", summary: "Bottom-up multi-person pose with Part Affinity Fields. Real-time on GPU." },
      { year: 2018, title: "Simple Baselines", authors: "Xiao et al.", venue: "ECCV", summary: "ResNet + deconvolution layers achieve competitive pose estimation with minimal complexity." },
      { year: 2019, title: "HRNet", authors: "Sun et al.", venue: "CVPR", summary: "High-Resolution Network maintains high-res representations throughout for precise keypoints." },
      { year: 2020, title: "MediaPipe Pose", authors: "Google", venue: "arXiv", summary: "Real-time on-device pose estimation using BlazePose architecture." },
      { year: 2022, title: "ViTPose", authors: "Xu et al.", venue: "NeurIPS", summary: "Plain Vision Transformer for pose estimation. Simple and scalable." },
    ],
    playground: {
      title: "Pose Estimation Playground",
      description: "Upload an image to estimate human keypoints using a ViTPose model.",
      taskType: "pose-estimation",
      acceptImage: true,
      modelName: "usyd-community/vitpose-base-simple",
      learningFocus: "Try crowd scenes, occlusions, and unusual poses to study keypoint confidence and failure cases.",
    },
  },

  tracking: {
    id: "tracking",
    title: "Multi-Object Tracking",
    subtitle: "Follow multiple objects across video frames while maintaining consistent identities — the temporal dimension of perception.",
    color: "280 70% 55%",
    theory: [
      {
        title: "Tracking by Detection",
        content: "Modern MOT follows the 'tracking by detection' paradigm: (1) detect objects independently in each frame, (2) associate detections across frames to form trajectories (tracks). The association problem is typically formulated as bipartite matching between existing tracks and new detections, optimizing an affinity/cost matrix that combines spatial proximity, appearance similarity, and motion prediction.",
      },
      {
        title: "Kalman Filter for Motion Prediction",
        content: "The Kalman filter is a recursive Bayesian estimator for linear dynamic systems with Gaussian noise. In MOT, it models the state of each tracked object as [x, y, a, h, ẋ, ẏ, ȧ, ḣ] (center position, aspect ratio, height, and their velocities). The predict step estimates the next state; the update step corrects this with the new detection. This allows handling short occlusions by predicting where the object should be.",
        equations: [
          { label: "State Prediction", tex: "\\hat{\\mathbf{x}}_{k|k-1} = F \\hat{\\mathbf{x}}_{k-1|k-1}" },
          { label: "Covariance Prediction", tex: "P_{k|k-1} = F P_{k-1|k-1} F^T + Q" },
          { label: "Kalman Gain", tex: "K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}" },
          { label: "State Update", tex: "\\hat{\\mathbf{x}}_{k|k} = \\hat{\\mathbf{x}}_{k|k-1} + K_k(\\mathbf{z}_k - H\\hat{\\mathbf{x}}_{k|k-1})" },
        ],
      },
      {
        title: "Hungarian Algorithm for Assignment",
        content: "The Hungarian algorithm solves the linear assignment problem in O(n³): given a cost matrix C where C[i,j] is the cost of assigning track i to detection j, find the assignment that minimizes total cost. In SORT, the cost is the IoU distance (1 - IoU) between predicted and detected bounding boxes. Unmatched tracks and detections are handled by creating new tracks or marking tracks for deletion.",
        equations: [
          { label: "Assignment Problem", tex: "\\min \\sum_{i,j} C_{ij} x_{ij}, \\quad \\text{s.t. } \\sum_j x_{ij} = 1, \\sum_i x_{ij} = 1" },
          { label: "IoU Cost", tex: "C_{ij} = 1 - \\text{IoU}(\\hat{b}_i, d_j)" },
        ],
      },
      {
        title: "Appearance Features (DeepSORT)",
        content: "DeepSORT extends SORT by adding a CNN-based appearance descriptor (Re-ID network) that encodes each detection into a 128-d feature vector. The association cost combines both motion (Mahalanobis distance from Kalman filter) and appearance (cosine distance between feature vectors). A gallery of recent appearance features is maintained for each track, enabling re-identification after long occlusions.",
        equations: [
          { label: "Combined Cost", tex: "c_{ij} = \\lambda \\cdot d^{(1)}_{ij} + (1-\\lambda) \\cdot d^{(2)}_{ij}" },
          { label: "Cosine Distance", tex: "d^{(2)}(i, j) = \\min_{r_k \\in \\mathcal{R}_i} (1 - r_k^T \\mathbf{f}_j)" },
          { label: "Mahalanobis Distance", tex: "d^{(1)}(i, j) = (d_j - \\hat{y}_i)^T S_i^{-1} (d_j - \\hat{y}_i)" },
        ],
      },
    ],
    algorithms: [
      {
        name: "DeepSORT Pipeline",
        steps: [
          { step: "Detection", detail: "Run object detector (YOLO/Faster R-CNN) to get bounding boxes per frame" },
          { step: "Feature Extraction", detail: "Extract appearance features for each detection using Re-ID CNN" },
          { step: "Kalman Predict", detail: "Predict next state for all existing tracks" },
          { step: "Cost Matrix Construction", detail: "Compute combined Mahalanobis + cosine distance between tracks and detections" },
          { step: "Cascade Matching", detail: "Match with priority to recently seen tracks (cascade over age)" },
          { step: "IoU Matching", detail: "Match remaining unmatched detections using IoU only" },
          { step: "Track Management", detail: "Create new tracks, delete old unmatched tracks after T_lost frames" },
        ],
      },
    ],
    papers: [
      { year: 2016, title: "SORT", authors: "Bewley et al.", venue: "ICIP", summary: "Simple Online Realtime Tracking using Kalman filter + Hungarian algorithm. Fast baseline." },
      { year: 2017, title: "DeepSORT", authors: "Wojke et al.", venue: "ICIP", summary: "Adds deep appearance features to SORT for robust re-identification after occlusion." },
      { year: 2020, title: "FairMOT", authors: "Zhang et al.", venue: "IJCV", summary: "Joint detection and Re-ID in one network with anchor-free detection." },
      { year: 2022, title: "ByteTrack", authors: "Zhang et al.", venue: "ECCV", summary: "Associates every detection box including low-confidence ones for improved tracking." },
      { year: 2023, title: "BoTrack", authors: "Aharon et al.", venue: "arXiv", summary: "BoT-SORT with camera motion compensation and improved association." },
    ],
    playgrounds: [
      {
        title: "Velocity Estimation Playground",
        description: "Upload a short video. The backend runs detection + DeepSORT tracking and estimates per-track velocity in pixels/second with annotated output.",
        taskType: "velocity-estimation",
        acceptVideo: true,
        acceptImage: false,
        modelName: "DETR + DeepSORT",
        learningFocus: "Inspect how FPS, object scale, and camera motion affect apparent velocity estimates.",
      },
      {
        title: "Integrated Perception Pipeline Playground",
        description: "End-to-end chain: object detection, DeepSORT tracking, instance-style segmentation, depth estimation, and velocity estimation with annotated video output.",
        taskType: "perception-pipeline",
        acceptVideo: true,
        acceptImage: false,
        modelName: "DETR + DeepSORT + Segmentation + DPT",
        learningFocus: "Study how errors propagate across tasks in a full perception stack and compare trade-offs between speed and richness.",
      },
    ],
  },

  action: {
    id: "action",
    title: "Video Action Recognition",
    subtitle: "Classify human actions in video sequences — understanding what is happening over time requires temporal reasoning.",
    color: "15 85% 55%",
    theory: [
      {
        title: "Temporal Modeling Challenges",
        content: "Unlike image classification, action recognition requires understanding temporal dynamics: how visual patterns change over time. A 'jumping' action looks like 'standing' in any single frame. Key challenges include: varying action duration, camera motion vs object motion, temporal boundaries (when does an action start/end?), and the combinatorial explosion of spatial-temporal patterns.",
      },
      {
        title: "Two-Stream Networks",
        content: "Two-stream networks process spatial (RGB frames) and temporal (optical flow) information separately, then fuse predictions. The spatial stream captures appearance ('what'), while the temporal stream captures motion ('how'). The optical flow stream explicitly provides motion information that would be difficult for a single-frame CNN to infer. Late fusion (averaging predictions) or early fusion (concatenating features) combine the streams.",
        equations: [
          { label: "Two-Stream Fusion", tex: "P(y|V) = \\frac{1}{2}P_{spatial}(y|I_t) + \\frac{1}{2}P_{temporal}(y|F_t)" },
        ],
      },
      {
        title: "3D Convolutions (C3D, I3D)",
        content: "3D CNNs extend 2D convolutions to the temporal dimension, learning spatiotemporal features directly from raw video clips. C3D uses 3×3×3 kernels throughout. I3D 'inflates' 2D ImageNet-pretrained weights into 3D by repeating and rescaling 2D filters along the temporal dimension. SlowFast networks use two pathways: a Slow pathway (low frame rate, spatial detail) and a Fast pathway (high frame rate, temporal detail).",
        equations: [
          { label: "3D Convolution", tex: "(f * g)(x, y, t) = \\sum_{i,j,k} f(i,j,k) \\cdot g(x-i, y-j, t-k)" },
          { label: "I3D Inflation", tex: "W_{3D}(i,j,t) = \\frac{1}{T} W_{2D}(i,j) \\quad \\forall t \\in [1, T]" },
        ],
      },
      {
        title: "Video Transformers",
        content: "Video transformers extend ViT to video by tokenizing spatiotemporal patches. TimeSformer explores different attention patterns: space-only, time-only, joint space-time, and divided space-time (most efficient). ViViT (Video Vision Transformer) uses tubelet embeddings spanning multiple frames. These models achieve state-of-the-art results but require significant compute for joint spatiotemporal attention over potentially hundreds of tokens.",
        equations: [
          { label: "Spatiotemporal Patch", tex: "\\mathbf{x}_p \\in \\mathbb{R}^{t \\times h \\times w \\times C}" },
          { label: "Divided Attention", tex: "\\text{Attn}_{ST} = \\text{Attn}_{time}(\\text{Attn}_{space}(\\mathbf{z}))" },
        ],
      },
    ],
    algorithms: [
      {
        name: "SlowFast Network Pipeline",
        steps: [
          { step: "Input Sampling", detail: "Sample T frames: Slow path gets T/α frames (e.g. 4), Fast gets T frames (e.g. 32)" },
          { step: "Slow Pathway", detail: "High channel capacity, low temporal rate → captures spatial semantics" },
          { step: "Fast Pathway", detail: "Low channel capacity (β=1/8), high frame rate → captures motion" },
          { step: "Lateral Connections", detail: "Fuse Fast→Slow features via temporal strided convolutions" },
          { step: "Global Average Pooling", detail: "Pool spatiotemporal features to fixed-length vector" },
          { step: "Classification Head", detail: "FC layer with softmax over action classes" },
        ],
      },
    ],
    papers: [
      { year: 2014, title: "Two-Stream Networks", authors: "Simonyan & Zisserman", venue: "NeurIPS", summary: "Dual-stream architecture for spatial and temporal information in action recognition." },
      { year: 2015, title: "C3D", authors: "Tran et al.", venue: "ICCV", summary: "Learning spatiotemporal features with 3D convolutions for video understanding." },
      { year: 2017, title: "I3D", authors: "Carreira & Zisserman", venue: "CVPR", summary: "Inflated 3D ConvNets: 2D ConvNet weights inflated to 3D for video recognition." },
      { year: 2019, title: "SlowFast", authors: "Feichtenhofer et al.", venue: "ICCV", summary: "Dual-pathway network: slow for spatial semantics, fast for temporal dynamics." },
      { year: 2021, title: "TimeSformer", authors: "Bertasius et al.", venue: "ICML", summary: "Divided space-time attention in transformers for efficient video understanding." },
      { year: 2021, title: "Video Swin Transformer", authors: "Liu et al.", venue: "CVPR", summary: "3D shifted windows for efficient video recognition with hierarchical features." },
    ],
    playground: {
      title: "Video Action Recognition Playground",
      description: "Upload a short MP4/WebM clip to classify the dominant action with a VideoMAE model.",
      taskType: "video-action-recognition",
      acceptVideo: true,
      acceptImage: false,
      modelName: "MCG-NJU/videomae-base-finetuned-kinetics",
      learningFocus: "Test short clips with distinct motions and compare predictions when trimming the same video to different durations.",
    },
  },

  opticalflow: {
    id: "opticalflow",
    title: "Optical Flow & Motion Estimation",
    subtitle: "Estimate per-pixel motion between consecutive frames — the fundamental representation of visual motion in video.",
    color: "170 80% 45%",
    theory: [
      {
        title: "What is Optical Flow?",
        content: "Optical flow is the apparent motion of image pixels between two consecutive frames. For each pixel (x, y) in frame t, optical flow predicts the displacement vector (u, v) such that the pixel moves to (x+u, y+v) in frame t+1. The optical flow field F: ℝ² → ℝ² is a dense 2D vector field capturing the projection of 3D motion onto the image plane. It encodes both object motion and camera ego-motion.",
        equations: [
          { label: "Optical Flow Field", tex: "\\mathbf{F}(x,y) = (u(x,y), v(x,y)) \\in \\mathbb{R}^2" },
          { label: "Motion Constraint", tex: "I(x + u, y + v, t+1) = I(x, y, t)" },
        ],
      },
      {
        title: "Brightness Constancy Assumption",
        content: "The fundamental assumption: pixel intensities are preserved between frames (no lighting changes, no occlusion). Taking a first-order Taylor expansion of this constraint yields the optical flow constraint equation. This single equation has two unknowns (u, v), making it an under-determined system — the aperture problem. Additional constraints are needed: Lucas-Kanade assumes locally constant flow (spatial smoothness within a window), while Horn-Schunck adds a global smoothness regularization term.",
        equations: [
          { label: "Brightness Constancy", tex: "I(x + u, y + v, t + \\Delta t) = I(x, y, t)" },
          { label: "Optical Flow Equation", tex: "I_x u + I_y v + I_t = 0 \\quad \\text{or} \\quad \\nabla I \\cdot \\mathbf{v} + I_t = 0" },
          { label: "Aperture Problem", tex: "\\text{One equation, two unknowns: } (u, v)" },
        ],
      },
      {
        title: "Lucas-Kanade Method",
        content: "Lucas-Kanade assumes the flow is constant within a small window (e.g., 5×5) around each pixel. This gives a system of n equations (one per pixel in the window) with 2 unknowns, solved by least squares. The method works well for small motions and textured regions but fails for large displacements and at motion boundaries. It is typically applied in a coarse-to-fine pyramid to handle larger motions.",
        equations: [
          { label: "Least Squares System", tex: "\\begin{bmatrix} \\sum I_x^2 & \\sum I_x I_y \\\\ \\sum I_x I_y & \\sum I_y^2 \\end{bmatrix} \\begin{bmatrix} u \\\\ v \\end{bmatrix} = -\\begin{bmatrix} \\sum I_x I_t \\\\ \\sum I_y I_t \\end{bmatrix}" },
          { label: "Structure Tensor", tex: "A^T A = \\begin{bmatrix} \\sum I_x^2 & \\sum I_x I_y \\\\ \\sum I_x I_y & \\sum I_y^2 \\end{bmatrix}" },
          { label: "Solvability Condition", tex: "\\lambda_1, \\lambda_2 \\gg 0 \\quad (\\text{both eigenvalues large} \\Rightarrow \\text{corner/texture})" },
        ],
      },
      {
        title: "Horn-Schunck Global Method",
        content: "Horn-Schunck formulates optical flow as a global energy minimization: minimize the data term (brightness constancy violation) plus a smoothness term (penalizing flow gradients). This produces dense flow fields but over-smooths motion boundaries. The energy is minimized iteratively using the Euler-Lagrange equations. Modern variational methods (Brox et al., TV-L1) use robust penalty functions and coarse-to-fine warping.",
        equations: [
          { label: "Horn-Schunck Energy", tex: "E(u,v) = \\iint \\left[(I_x u + I_y v + I_t)^2 + \\alpha^2(|\\nabla u|^2 + |\\nabla v|^2)\\right] dx\\,dy" },
          { label: "Euler-Lagrange Equations", tex: "I_x(I_x u + I_y v + I_t) - \\alpha^2 \\nabla^2 u = 0" },
        ],
      },
      {
        title: "Deep Optical Flow: FlowNet to RAFT",
        content: "FlowNet was the first end-to-end CNN for optical flow, with two variants: FlowNetS (simple encoder-decoder) and FlowNetC (with correlation layer). PWC-Net introduced warping, cost volume, and coarse-to-fine refinement. RAFT (Recurrent All-Pairs Field Transform) revolutionized the field by building a full 4D correlation volume between all pairs of pixels and iteratively refining flow predictions using a GRU-based update operator. RAFT achieves state-of-the-art accuracy with strong generalization.",
        equations: [
          { label: "Correlation Volume (RAFT)", tex: "C_{ijkl} = \\sum_h g_\\theta^1(I_1)_{ijh} \\cdot g_\\theta^2(I_2)_{klh}" },
          { label: "GRU Update (RAFT)", tex: "\\mathbf{h}_{k+1} = \\text{GRU}(\\mathbf{h}_k, \\mathbf{x}_k), \\quad \\Delta \\mathbf{f} = \\text{Conv}(\\mathbf{h}_{k+1})" },
          { label: "Iterative Refinement", tex: "\\mathbf{f}_{k+1} = \\mathbf{f}_k + \\Delta \\mathbf{f}_k" },
          { label: "Endpoint Error", tex: "\\text{EPE} = \\frac{1}{N}\\sum_i \\|\\mathbf{f}_i^{pred} - \\mathbf{f}_i^{gt}\\|_2" },
        ],
      },
    ],
    algorithms: [
      {
        name: "RAFT Optical Flow Pipeline",
        steps: [
          { step: "Feature Extraction", detail: "Shared-weight CNN extracts feature maps g(I₁) and g(I₂) at 1/8 resolution" },
          { step: "Correlation Volume", detail: "Compute all-pairs dot product between feature maps → 4D volume H×W×H×W" },
          { step: "Correlation Pyramid", detail: "Pool last two dimensions at multiple scales for multi-scale matching" },
          { step: "Initialize Flow", detail: "Set initial flow to zero: f₀ = 0" },
          { step: "Iterative Updates (×12)", detail: "Look up correlation features, concatenate with context, run GRU to predict Δf" },
          { step: "Upsample", detail: "Convex upsampling from 1/8 to full resolution using learned weights" },
        ],
      },
      {
        name: "Lucas-Kanade Pyramid Pipeline",
        steps: [
          { step: "Build Image Pyramid", detail: "Downsample both frames by factor 2 at each level (L levels)" },
          { step: "Coarsest Level", detail: "Estimate flow at lowest resolution using LK with window W" },
          { step: "Propagate Upward", detail: "Upsample flow estimate and use as initialization for next level" },
          { step: "Refine at Each Level", detail: "Warp image by current flow, compute residual flow with LK" },
          { step: "Output", detail: "Final flow at full resolution captures both large and small motions" },
        ],
      },
    ],
    papers: [
      { year: 1981, title: "Horn-Schunck", authors: "Horn & Schunck", venue: "AI", summary: "Pioneering global variational approach to optical flow with smoothness constraint." },
      { year: 1981, title: "Lucas-Kanade", authors: "Lucas & Kanade", venue: "IJCAI", summary: "Local window-based least-squares optical flow. Foundation for sparse tracking." },
      { year: 2004, title: "Brox et al.", authors: "Brox, Bruhn, Papenberg, Weickert", venue: "ECCV", summary: "High accuracy variational optical flow with robust penalty functions and nested warping." },
      { year: 2015, title: "FlowNet", authors: "Dosovitskiy et al.", venue: "ICCV", summary: "First CNN for optical flow estimation. Introduced Flying Chairs synthetic dataset." },
      { year: 2018, title: "PWC-Net", authors: "Sun et al.", venue: "CVPR", summary: "Pyramid, Warping, Cost volume: compact and efficient deep optical flow." },
      { year: 2020, title: "RAFT", authors: "Teed & Deng", venue: "ECCV", summary: "Recurrent All-Pairs Field Transform. State-of-the-art optical flow with iterative refinement." },
      { year: 2022, title: "FlowFormer", authors: "Huang et al.", venue: "ECCV", summary: "Transformer-based cost volume processing for accurate optical flow estimation." },
      { year: 2023, title: "UniMatch", authors: "Xu et al.", venue: "TPAMI", summary: "Unified model for flow, stereo, and depth using task-agnostic matching." },
    ],
  },
};
