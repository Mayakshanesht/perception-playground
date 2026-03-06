import { ConceptLabExercise } from "@/components/ConceptLab";

export const moduleLabs: Record<string, ConceptLabExercise[]> = {
  camera: [
    {
      id: "camera-projection",
      title: "Camera Projection Lab",
      description: "Compute image projection of a 3D point using the pinhole model.",
      difficulty: "beginner",
      task: "A 3D point P = (2, 3, 10) is observed by a camera with focal lengths fx = fy = 500 pixels and principal point (cx, cy) = (320, 240). Compute the 2D image coordinates (u, v) using the pinhole model.",
      equation: {
        label: "Pinhole Projection",
        tex: "u = f_x \\frac{X}{Z} + c_x, \\quad v = f_y \\frac{Y}{Z} + c_y",
      },
      variables: [
        { symbol: "X, Y, Z", meaning: "3D point coordinates in camera frame" },
        { symbol: "fx, fy", meaning: "focal lengths in pixels" },
        { symbol: "cx, cy", meaning: "principal point (image center)" },
        { symbol: "u, v", meaning: "resulting 2D pixel coordinates" },
      ],
      hints: [
        "Divide X and Y by Z to project onto the normalized image plane",
        "Then scale by focal length and shift by principal point",
      ],
      solution: "u = 500 × (2/10) + 320 = 500 × 0.2 + 320 = 100 + 320 = 420\nv = 500 × (3/10) + 240 = 500 × 0.3 + 240 = 150 + 240 = 390\n\nThe 3D point (2, 3, 10) projects to pixel (420, 390).",
    },
    {
      id: "lens-distortion",
      title: "Lens Distortion Lab",
      description: "Simulate radial distortion on a grid and understand barrel vs pincushion distortion.",
      difficulty: "intermediate",
      task: "A point at normalized coordinates (x, y) = (0.5, 0.3) has radial distance r = √(0.5² + 0.3²). Given k₁ = -0.2 (barrel distortion), compute the distorted coordinates. Then predict what happens if k₁ = +0.2 (pincushion).",
      equation: {
        label: "Radial Distortion",
        tex: "x_{dist} = x(1 + k_1 r^2), \\quad y_{dist} = y(1 + k_1 r^2)",
      },
      variables: [
        { symbol: "x, y", meaning: "undistorted normalized image coordinates" },
        { symbol: "k₁", meaning: "first radial distortion coefficient" },
        { symbol: "r", meaning: "radial distance from optical center" },
      ],
      hints: [
        "First compute r² = 0.5² + 0.3² = 0.25 + 0.09 = 0.34",
        "Then compute the scale factor: 1 + k₁ × r²",
        "Barrel distortion (k₁ < 0) pushes points inward; pincushion (k₁ > 0) pushes outward",
      ],
      solution: "r² = 0.34\n\nBarrel (k₁ = -0.2):\nScale = 1 + (-0.2)(0.34) = 1 - 0.068 = 0.932\nx_dist = 0.5 × 0.932 = 0.466\ny_dist = 0.3 × 0.932 = 0.280\nPoints move inward → barrel distortion.\n\nPincushion (k₁ = +0.2):\nScale = 1 + 0.2(0.34) = 1.068\nx_dist = 0.534, y_dist = 0.320\nPoints move outward → pincushion distortion.",
    },
    {
      id: "stereo-depth",
      title: "Stereo Depth Lab",
      description: "Compute depth from stereo disparity.",
      difficulty: "beginner",
      task: "A stereo camera system has baseline B = 0.12 m and focal length f = 720 pixels. An object appears at pixel x_L = 400 in the left image and x_R = 350 in the right image. Compute the disparity and the depth Z.",
      equation: {
        label: "Stereo Depth",
        tex: "Z = \\frac{f \\cdot B}{d}, \\quad d = x_L - x_R",
      },
      variables: [
        { symbol: "Z", meaning: "depth (distance to object)" },
        { symbol: "f", meaning: "focal length in pixels" },
        { symbol: "B", meaning: "baseline (distance between cameras)" },
        { symbol: "d", meaning: "disparity (horizontal pixel difference)" },
      ],
      hints: [
        "Disparity d = x_L - x_R",
        "Then plug into Z = fB/d",
      ],
      solution: "d = 400 - 350 = 50 pixels\nZ = (720 × 0.12) / 50 = 86.4 / 50 = 1.728 m\n\nThe object is approximately 1.73 meters away from the camera.",
    },
  ],
  semantic: [
    {
      id: "iou-computation",
      title: "IoU Computation Lab",
      description: "Calculate Intersection over Union for object detection evaluation.",
      difficulty: "beginner",
      task: "A predicted bounding box has corners (100, 100) to (200, 200). The ground truth box is (120, 110) to (220, 210). Compute the IoU.",
      equation: {
        label: "Intersection over Union",
        tex: "\\text{IoU} = \\frac{|A \\cap B|}{|A \\cup B|} = \\frac{\\text{Intersection Area}}{\\text{Area}_A + \\text{Area}_B - \\text{Intersection Area}}",
      },
      variables: [
        { symbol: "A", meaning: "predicted bounding box area" },
        { symbol: "B", meaning: "ground truth bounding box area" },
        { symbol: "A ∩ B", meaning: "overlapping area between prediction and ground truth" },
      ],
      hints: [
        "Find the intersection rectangle by taking max of left/top edges and min of right/bottom edges",
        "Intersection: x_left = max(100,120)=120, y_top = max(100,110)=110, x_right = min(200,220)=200, y_bottom = min(200,210)=200",
        "Intersection area = (200-120) × (200-110) = 80 × 90 = 7200",
      ],
      solution: "Pred area = 100 × 100 = 10,000\nGT area = 100 × 100 = 10,000\n\nIntersection corners: (120,110) to (200,200)\nIntersection area = 80 × 90 = 7,200\n\nUnion = 10,000 + 10,000 - 7,200 = 12,800\n\nIoU = 7,200 / 12,800 = 0.5625 ≈ 56.25%\n\nThis would pass a 50% IoU threshold but fail at 75%.",
    },
    {
      id: "cnn-receptive-field",
      title: "CNN Receptive Field Lab",
      description: "Compute the receptive field of a CNN layer stack.",
      difficulty: "intermediate",
      task: "A CNN has three 3×3 convolutional layers (stride 1, no dilation), each followed by ReLU. What is the effective receptive field at the output? What about with a 2×2 max pool between layer 2 and 3?",
      equation: {
        label: "Receptive Field Growth",
        tex: "RF_l = RF_{l-1} + (K_l - 1) \\times \\prod_{i=1}^{l-1} S_i",
      },
      variables: [
        { symbol: "RF_l", meaning: "receptive field at layer l" },
        { symbol: "K_l", meaning: "kernel size at layer l" },
        { symbol: "S_i", meaning: "stride at layer i" },
      ],
      hints: [
        "For stride-1 convolutions, each 3×3 layer adds 2 to the receptive field",
        "A 2×2 max pool effectively doubles the stride for subsequent layers",
      ],
      solution: "Without pooling:\nLayer 1: RF = 3\nLayer 2: RF = 3 + (3-1)×1 = 5\nLayer 3: RF = 5 + (3-1)×1 = 7\n\nWith 2×2 pool between layer 2 and 3:\nLayer 1: RF = 3 (stride 1)\nLayer 2: RF = 5 (stride 1)\nPool: stride becomes 2\nLayer 3: RF = 5 + (3-1)×2 = 9\n\nThe pooling layer expands the receptive field more aggressively.",
    },
  ],
  geometric: [
    {
      id: "epipolar-constraint",
      title: "Epipolar Geometry Lab",
      description: "Verify the epipolar constraint between two views.",
      difficulty: "advanced",
      task: "Given a fundamental matrix F and corresponding points x = (150, 200, 1)ᵀ and x' = (180, 220, 1)ᵀ, verify whether x'ᵀFx ≈ 0. If F = [[0, 0, -0.002], [0, 0, 0.003], [0.001, -0.002, 0.1]], compute x'ᵀFx.",
      equation: {
        label: "Epipolar Constraint",
        tex: "\\mathbf{x'}^T F \\mathbf{x} = 0",
      },
      variables: [
        { symbol: "x, x'", meaning: "corresponding points in homogeneous coordinates" },
        { symbol: "F", meaning: "fundamental matrix (rank 2, encodes epipolar geometry)" },
      ],
      hints: [
        "First compute Fx by matrix-vector multiplication",
        "Then take the dot product of x' with the result",
        "A perfect correspondence gives exactly 0; noise gives a small residual",
      ],
      solution: "Fx = [0×150 + 0×200 + (-0.002)×1, 0×150 + 0×200 + 0.003×1, 0.001×150 + (-0.002)×200 + 0.1×1]\n= [-0.002, 0.003, 0.15 - 0.4 + 0.1]\n= [-0.002, 0.003, -0.15]\n\nx'ᵀFx = 180×(-0.002) + 220×0.003 + 1×(-0.15)\n= -0.36 + 0.66 - 0.15\n= 0.15\n\nThe result is not zero, meaning these points are NOT a perfect correspondence (noise or incorrect match). A good match would yield a value close to 0.",
    },
  ],
  motion: [
    {
      id: "optical-flow-brightness",
      title: "Optical Flow Brightness Constancy Lab",
      description: "Derive and understand the optical flow constraint equation.",
      difficulty: "intermediate",
      task: "The brightness constancy assumption states I(x, y, t) = I(x+u, y+v, t+1). Using a first-order Taylor expansion, derive the optical flow constraint equation. Then explain why this gives one equation for two unknowns (the aperture problem).",
      equation: {
        label: "Optical Flow Constraint",
        tex: "I_x u + I_y v + I_t = 0",
      },
      variables: [
        { symbol: "Ix, Iy", meaning: "spatial image gradients (∂I/∂x, ∂I/∂y)" },
        { symbol: "It", meaning: "temporal gradient (∂I/∂t)" },
        { symbol: "u, v", meaning: "horizontal and vertical flow components" },
      ],
      hints: [
        "Apply Taylor expansion: I(x+u, y+v, t+1) ≈ I(x,y,t) + Ix·u + Iy·v + It",
        "Since I(x,y,t) = I(x+u,y+v,t+1), the expansion minus original gives the constraint",
      ],
      solution: "Taylor expansion of I(x+u, y+v, t+1):\n≈ I(x,y,t) + (∂I/∂x)u + (∂I/∂y)v + (∂I/∂t)\n\nSetting equal to I(x,y,t) and canceling:\nIx·u + Iy·v + It = 0\n\nThis is ONE equation with TWO unknowns (u, v). Through a small aperture (local window), you can only measure the flow component normal to the image gradient direction:\nv_n = -It / √(Ix² + Iy²)\n\nThis is the aperture problem. Lucas-Kanade resolves it by assuming constant flow in a local window, giving an overdetermined system.",
    },
  ],
  reconstruction: [
    {
      id: "nerf-ray-sampling",
      title: "NeRF Ray Sampling Lab",
      description: "Simulate volumetric rendering along a single ray.",
      difficulty: "advanced",
      task: "A ray passes through 4 sample points with densities σ = [0.1, 0.5, 2.0, 0.3] and colors c = [red, green, blue, yellow]. Distances between samples are δ = [0.5, 0.5, 0.5]. Compute the transmittance T and accumulated color using discrete volume rendering.",
      equation: {
        label: "Discrete Volume Rendering",
        tex: "\\hat{C}(\\mathbf{r}) = \\sum_{i=1}^{N} T_i \\alpha_i \\mathbf{c}_i, \\quad T_i = \\prod_{j=1}^{i-1}(1-\\alpha_j), \\quad \\alpha_i = 1 - e^{-\\sigma_i \\delta_i}",
      },
      variables: [
        { symbol: "σᵢ", meaning: "volume density at sample i (opacity per unit length)" },
        { symbol: "δᵢ", meaning: "distance between consecutive samples" },
        { symbol: "αᵢ", meaning: "opacity at sample i (how much it blocks light)" },
        { symbol: "Tᵢ", meaning: "transmittance (how much light reaches sample i)" },
        { symbol: "cᵢ", meaning: "color at sample i" },
      ],
      hints: [
        "First compute α for each sample: α = 1 - exp(-σδ)",
        "T₁ = 1 (nothing in front), T₂ = (1-α₁), T₃ = (1-α₁)(1-α₂), etc.",
        "Weight for each sample = T × α. Final color = weighted sum.",
      ],
      solution: "α₁ = 1 - e^(-0.1×0.5) = 1 - 0.951 = 0.049\nα₂ = 1 - e^(-0.5×0.5) = 1 - 0.779 = 0.221\nα₃ = 1 - e^(-2.0×0.5) = 1 - 0.368 = 0.632\nα₄ = 1 - e^(-0.3×0.5) = 1 - 0.861 = 0.139\n\nT₁ = 1.000\nT₂ = (1-0.049) = 0.951\nT₃ = 0.951 × (1-0.221) = 0.741\nT₄ = 0.741 × (1-0.632) = 0.273\n\nWeights: w₁=0.049, w₂=0.210, w₃=0.468, w₄=0.038\n\nBlue (sample 3) dominates because it has the highest density. The ray mostly 'sees' blue with some green from sample 2.",
    },
  ],
  "scene-reasoning": [
    {
      id: "clip-similarity",
      title: "CLIP Zero-Shot Classification Lab",
      description: "Understand how CLIP performs zero-shot classification via cosine similarity.",
      difficulty: "beginner",
      task: "An image of a cat has CLIP embedding v = [0.5, 0.7, 0.1]. Three text prompts have embeddings: 'a photo of a cat' t₁ = [0.6, 0.8, 0.0], 'a photo of a dog' t₂ = [0.3, 0.2, 0.9], 'a photo of a car' t₃ = [0.1, 0.0, 0.8]. Which text is the best match?",
      equation: {
        label: "Cosine Similarity",
        tex: "\\text{sim}(\\mathbf{v}, \\mathbf{t}) = \\frac{\\mathbf{v} \\cdot \\mathbf{t}}{\\|\\mathbf{v}\\| \\|\\mathbf{t}\\|}",
      },
      variables: [
        { symbol: "v", meaning: "image embedding from CLIP visual encoder" },
        { symbol: "t", meaning: "text embedding from CLIP text encoder" },
        { symbol: "sim", meaning: "cosine similarity (range -1 to 1, higher = more similar)" },
      ],
      hints: [
        "Compute dot product v·t for each text",
        "Compute magnitudes ||v|| and ||t||",
        "The highest cosine similarity wins the zero-shot classification",
      ],
      solution: "||v|| = √(0.25 + 0.49 + 0.01) = √0.75 = 0.866\n\nv·t₁ = 0.30 + 0.56 + 0 = 0.86, ||t₁|| = √(0.36+0.64+0) = 1.0\nsim₁ = 0.86 / (0.866 × 1.0) = 0.993\n\nv·t₂ = 0.15 + 0.14 + 0.09 = 0.38, ||t₂|| = √(0.09+0.04+0.81) = 0.970\nsim₂ = 0.38 / (0.866 × 0.970) = 0.452\n\nv·t₃ = 0.05 + 0 + 0.08 = 0.13, ||t₃|| = √(0.01+0+0.64) = 0.806\nsim₃ = 0.13 / (0.866 × 0.806) = 0.186\n\nBest match: 'a photo of a cat' with similarity 0.993 ✓",
    },
  ],
};
