import { ModuleContent } from "./moduleContent";

export const generativeVisionModule: ModuleContent = {
  id: "generative-vision",
  title: "Generative Models for Vision",
  subtitle: "From VAEs to Stable Diffusion, ControlNet adapters, and video generation — understand the math and architectures behind modern generative AI.",
  color: "340 75% 58%",
  theory: [
    {
      title: "Intuition",
      content:
        "Generative models learn to create new data that looks like the training distribution. Unlike discriminative models (which map input → label), generative models learn the data distribution p(x) itself. Three major paradigms have emerged: VAEs (learn a smooth latent space via variational inference), GANs (adversarial game between generator and discriminator), and Diffusion Models (learn to reverse a noise-adding process). Diffusion models now dominate image generation with models like Stable Diffusion, DALL-E 3, and Midjourney.",
    },
    {
      title: "VAE — Variational Autoencoders",
      content:
        "VAEs maximize the evidence lower bound (ELBO) on log p(x). The key insight: since marginalizing over all latent z is intractable, introduce an approximate posterior q_φ(z|x) and derive a tractable lower bound. The ELBO has two terms: (1) reconstruction loss — how well the decoder reconstructs x from z, and (2) KL regularizer — how close the encoder posterior is to the prior N(0,I). The reparameterization trick z = μ + σ⊙ε enables gradient flow through the sampling operation.",
      equations: [
        {
          label: "ELBO Derivation",
          tex: "\\log p_\\theta(x) \\geq \\mathbb{E}_{q_\\phi(z|x)}[\\log p_\\theta(x|z)] - \\text{KL}[q_\\phi(z|x) \\| p(z)] = \\text{ELBO}",
          variables: [
            { symbol: "q_φ(z|x)", meaning: "encoder (approximate posterior)" },
            { symbol: "p_θ(x|z)", meaning: "decoder (likelihood)" },
            { symbol: "p(z) = N(0,I)", meaning: "prior on latent space" },
          ],
        },
        {
          label: "Reparameterization Trick",
          tex: "z = \\mu_\\phi(x) + \\sigma_\\phi(x) \\odot \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, I)",
        },
        {
          label: "KL Divergence (Gaussian)",
          tex: "\\text{KL}[\\mathcal{N}(\\mu, \\sigma^2) \\| \\mathcal{N}(0, I)] = \\frac{1}{2} \\sum_i (\\mu_i^2 + \\sigma_i^2 - \\log \\sigma_i^2 - 1)",
        },
      ],
    },
    {
      title: "VQ-VAE — Discrete Latent Codes",
      content:
        "VQ-VAE replaces the continuous latent space with a discrete codebook of K embedding vectors. The encoder output z_e is mapped to the nearest codebook entry z_q = argmin_k ||z_e - e_k||. The straight-through estimator passes gradients through the quantization step as if z_q = z_e. VQ-VAE is the backbone of DALL-E 1, MaskGIT, and VQGAN (used as the tokenizer in Stable Diffusion). The codebook learns a compact discrete representation of visual patterns.",
      equations: [
        {
          label: "Vector Quantization",
          tex: "z_q = \\arg\\min_k \\|z_e - e_k\\|_2 \\quad \\text{(nearest codebook entry)}",
        },
        {
          label: "VQ-VAE Loss",
          tex: "\\mathcal{L} = \\mathcal{L}_{\\text{recon}} + \\beta\\|\\text{sg}(z_e) - e\\|^2 + \\|z_e - \\text{sg}(e)\\|^2",
          variables: [
            { symbol: "sg(·)", meaning: "stop-gradient operator" },
            { symbol: "β", meaning: "commitment loss weight" },
          ],
        },
      ],
    },
    {
      title: "GANs — Adversarial Training",
      content:
        "GANs are a two-player minimax game: the Generator G tries to produce realistic samples, the Discriminator D tries to distinguish real from fake. The optimal discriminator D*(x) = p_data(x)/(p_data(x) + p_G(x)), and substituting back gives the Jensen-Shannon Divergence. Problem: JSD is constant (log 2) when supports don't overlap, causing vanishing gradients for G. The non-saturating heuristic L_G = -E[log D(G(z))] partially addresses this.",
      equations: [
        {
          label: "GAN Minimax Objective",
          tex: "\\min_G \\max_D V(D, G) = \\mathbb{E}_{x \\sim p_{\\text{data}}}[\\log D(x)] + \\mathbb{E}_{z \\sim p_z}[\\log(1 - D(G(z)))]",
        },
        {
          label: "Optimal Discriminator",
          tex: "D^*(x) = \\frac{p_{\\text{data}}(x)}{p_{\\text{data}}(x) + p_G(x)}",
        },
      ],
    },
    {
      title: "WGAN — Wasserstein Distance",
      content:
        "WGAN replaces JSD with the Wasserstein (Earth Mover) distance, which provides continuous, meaningful gradients even when distributions don't overlap. The Kantorovich-Rubinstein duality reformulates the distance as a supremum over 1-Lipschitz functions. WGAN-GP enforces the Lipschitz constraint via gradient penalty: penalize the gradient norm on interpolated samples. This eliminates mode collapse and provides a loss metric that correlates with sample quality.",
      equations: [
        {
          label: "Wasserstein Distance (Dual)",
          tex: "W(p_r, p_g) = \\sup_{\\|f\\|_L \\leq 1} \\mathbb{E}_{x \\sim p_r}[f(x)] - \\mathbb{E}_{x \\sim p_g}[f(x)]",
        },
        {
          label: "Gradient Penalty (WGAN-GP)",
          tex: "\\mathcal{L}_{GP} = \\lambda \\cdot \\mathbb{E}_{\\hat{x}}\\left[(\\|\\nabla_{\\hat{x}} D(\\hat{x})\\|_2 - 1)^2\\right], \\quad \\lambda = 10",
          variables: [
            { symbol: "x̂", meaning: "interpolation: εx_real + (1-ε)x_fake" },
          ],
        },
      ],
    },
    {
      title: "Score Functions & Tweedie's Formula",
      content:
        "Score-based generative models learn the score function s_θ(x,t) ≈ ∇_x log p_t(x), the gradient of the log-density. This score points toward regions of high data density. The deep connection to diffusion: the noise prediction ε_θ and the score are related by s_θ = -ε_θ/σ_t. Tweedie's formula provides the optimal denoiser: E[x₀|x_t] = (x_t + σ_t² · ∇ log p(x_t))/α_t. This means predicting noise IS predicting the score IS denoising — they are mathematically equivalent views of the same objective. Score matching allows training without knowing the normalizing constant of p(x).",
      equations: [
        {
          label: "Score Matching Objective",
          tex: "\\mathcal{L}_{SM} = \\mathbb{E}_{t,x_0,\\epsilon}\\left[\\lambda(t) \\|s_\\theta(x_t, t) - \\nabla_{x_t} \\log q(x_t|x_0)\\|^2\\right]",
          variables: [
            { symbol: "s_θ(x,t)", meaning: "learned score function approximating ∇ log p_t(x)" },
            { symbol: "λ(t)", meaning: "weighting function (often σ_t² for uniform SNR weighting)" },
          ],
        },
        {
          label: "Tweedie's Formula (Optimal Denoiser)",
          tex: "\\mathbb{E}[x_0 | x_t] = \\frac{x_t + (1-\\bar{\\alpha}_t) \\nabla_{x_t} \\log p(x_t)}{\\sqrt{\\bar{\\alpha}_t}}",
        },
        {
          label: "Score-Noise Equivalence",
          tex: "s_\\theta(x_t, t) = -\\frac{\\epsilon_\\theta(x_t, t)}{\\sqrt{1 - \\bar{\\alpha}_t}}",
        },
      ],
    },
    {
      title: "DDPM — Denoising Diffusion Probabilistic Models",
      content:
        "Diffusion models learn to reverse a Markov chain that gradually adds Gaussian noise over T steps. The forward process q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI) with β increasing from 1e-4 to 0.02 over T=1000 steps. The key reparameterization: q(xₜ|x₀) = N(xₜ; √ᾱₜx₀, (1-ᾱₜ)I) gives a closed-form expression for any timestep directly. The simplified training objective just predicts the noise ε added at each step, which is much more stable than predicting the mean. The signal-to-noise ratio SNR(t) = ᾱ_t/(1-ᾱ_t) monotonically decreases, connecting the noise schedule to information content.",
      equations: [
        {
          label: "Forward Process (Closed-Form)",
          tex: "x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, I)",
          variables: [
            { symbol: "ᾱₜ = Πₛ(1-βₛ)", meaning: "cumulative noise schedule product" },
            { symbol: "SNR(t) = ᾱₜ/(1-ᾱₜ)", meaning: "signal-to-noise ratio → 0 as t→T" },
          ],
        },
        {
          label: "Simplified Training Loss",
          tex: "\\mathcal{L}_{\\text{simple}} = \\mathbb{E}_{t, x_0, \\epsilon}\\left[\\|\\epsilon - \\epsilon_\\theta(\\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon, t)\\|^2\\right]",
        },
        {
          label: "Variational Lower Bound (Full)",
          tex: "\\mathcal{L}_{\\text{VLB}} = \\sum_{t=1}^{T} \\text{KL}[q(x_{t-1}|x_t,x_0) \\| p_\\theta(x_{t-1}|x_t)]",
        },
      ],
    },
    {
      title: "Noise Schedules — Linear, Cosine & Scaled-Linear",
      content:
        "The noise schedule β_t controls how quickly signal is destroyed. Linear schedule (DDPM): β ∈ [1e-4, 0.02] — simple but wastes compute near t=T where the image is already pure noise. Cosine schedule (Nichol & Dhariwal): ᾱ_t = cos²((t/T+s)/(1+s) · π/2) — much smoother SNR decay, preserving more signal at intermediate steps. Scaled-linear (SDXL): β_t = (√β_min + t·(√β_max−√β_min))² — compromise for higher resolutions. The v-prediction parameterization predicts v = ᾱ_t·ε − √(1-ᾱ_t)·x₀, providing better signal at low SNR and enabling better fine-tuning from pretrained models.",
      equations: [
        {
          label: "Cosine Schedule",
          tex: "\\bar{\\alpha}_t = \\frac{f(t)^2}{f(0)^2}, \\quad f(t) = \\cos\\left(\\frac{t/T + s}{1 + s} \\cdot \\frac{\\pi}{2}\\right)",
          variables: [
            { symbol: "s = 0.008", meaning: "offset to prevent β_0 from being too small" },
          ],
        },
        {
          label: "v-Prediction",
          tex: "v_t = \\sqrt{\\bar{\\alpha}_t} \\cdot \\epsilon - \\sqrt{1 - \\bar{\\alpha}_t} \\cdot x_0",
          variables: [
            { symbol: "v_t", meaning: "velocity target — interpolates between ε and x₀ predictions" },
          ],
        },
      ],
    },
    {
      title: "DDIM & Flow Matching",
      content:
        "DDIM (Denoising Diffusion Implicit Models) provides deterministic sampling by setting the noise parameter σₜ = 0, enabling the same xT to always produce the same image. This allows 50-step sampling instead of 1000 steps. Flow Matching (Lipman et al.) offers a simpler alternative: learn a velocity field v_θ(x,t) that moves samples along an ODE from noise to data. The conditional flow uses constant velocity v = x₁ - x₀, giving a simpler loss than DDPM. Rectified Flow straightens the ODE trajectories via iterative distillation, enabling 1-4 step generation. Classifier-Free Guidance (CFG) combines conditional and unconditional scores to amplify prompt alignment.",
      equations: [
        {
          label: "DDIM Update",
          tex: "x_{t-1} = \\sqrt{\\bar{\\alpha}_{t-1}} \\cdot x_0^{\\text{pred}} + \\sqrt{1 - \\bar{\\alpha}_{t-1} - \\sigma_t^2} \\cdot \\epsilon_\\theta + \\sigma_t \\cdot \\epsilon",
        },
        {
          label: "Classifier-Free Guidance",
          tex: "\\tilde{\\epsilon}_\\theta(x_t, c) = \\epsilon_\\theta(x_t, \\emptyset) + w \\cdot (\\epsilon_\\theta(x_t, c) - \\epsilon_\\theta(x_t, \\emptyset))",
          variables: [
            { symbol: "w = 7.5", meaning: "typical guidance scale in Stable Diffusion" },
            { symbol: "∅", meaning: "unconditional (null) prompt embedding" },
          ],
        },
        {
          label: "Flow Matching Loss",
          tex: "\\mathcal{L}_{FM} = \\mathbb{E}_{t,x_0,x_1}\\|v_\\theta(x_t, t) - (x_1 - x_0)\\|^2, \\quad x_t = (1-t)x_0 + t \\cdot x_1",
        },
        {
          label: "Rectified Flow (Distillation)",
          tex: "x_t^{\\text{new}} = (1-t) x_0^{\\text{gen}} + t \\cdot x_1, \\quad \\text{iterate to straighten ODE paths}",
        },
      ],
    },
    {
      title: "Latent Diffusion & Stable Diffusion",
      content:
        "Latent Diffusion Models (LDM) perform diffusion in a compressed latent space instead of pixel space. An image x ∈ ℝ^{512×512×3} is encoded to z ∈ ℝ^{64×64×4} via a pre-trained VQ-VAE encoder (f=8 compression), reducing compute by 64×. The U-Net denoiser uses: (1) ResBlocks with time embedding via AdaGN, (2) self-attention over spatial features, (3) cross-attention with Q from features and K,V from CLIP text tokens to inject text conditioning. SDXL uses a larger U-Net, dual text encoders (CLIP ViT-L + OpenCLIP ViT-bigG), and a refiner model for detail enhancement.",
      equations: [
        {
          label: "Latent Diffusion",
          tex: "x \\xrightarrow{\\mathcal{E}} z \\in \\mathbb{R}^{h/f \\times w/f \\times c}, \\quad \\text{diffuse in } z, \\quad \\mathcal{D}(z_0^{\\text{pred}}) = \\hat{x}",
          variables: [
            { symbol: "f = 8", meaning: "spatial compression factor" },
            { symbol: "c = 4", meaning: "latent channels" },
          ],
        },
        {
          label: "Cross-Attention Conditioning",
          tex: "\\text{Attn}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d}}\\right)V, \\quad Q = W_Q \\cdot z, \\quad K = W_K \\cdot c, \\quad V = W_V \\cdot c",
          variables: [
            { symbol: "z", meaning: "spatial features from U-Net" },
            { symbol: "c", meaning: "CLIP text embeddings ∈ ℝ^{77×768}" },
          ],
        },
      ],
    },
    {
      title: "ControlNet — Structural Conditioning",
      content:
        "ControlNet adds spatial conditioning (edges, depth, pose, segmentation maps) to pre-trained diffusion models while preserving the original model's capabilities. The key architectural innovation is zero-convolution initialization: a trainable copy of the U-Net encoder is connected to the locked original via zero-initialized convolution layers. At training start, the output is exactly the original model (zero × anything = 0). During training, the zero-conv weights gradually learn to inject conditioning signals. This prevents catastrophic forgetting and enables stable fine-tuning. Multi-ControlNet stacks multiple conditions with weighted contributions.",
      equations: [
        {
          label: "ControlNet Output",
          tex: "y = \\mathcal{F}_{\\text{locked}}(x) + \\text{zero\\_conv}(\\mathcal{F}_{\\text{trainable}}(x + \\text{zero\\_conv}(c)))",
          variables: [
            { symbol: "F_locked", meaning: "frozen pre-trained U-Net decoder blocks" },
            { symbol: "F_trainable", meaning: "trainable copy of U-Net encoder blocks" },
            { symbol: "c", meaning: "conditioning signal (Canny edges, depth map, pose skeleton)" },
            { symbol: "zero_conv", meaning: "1×1 conv initialized to zero weights and zero bias" },
          ],
        },
        {
          label: "Multi-ControlNet",
          tex: "y = \\mathcal{F}_{\\text{locked}}(x) + \\sum_{i=1}^{N} w_i \\cdot \\text{zero\\_conv}_i(\\mathcal{F}_{\\text{train},i}(x, c_i))",
          variables: [
            { symbol: "w_i", meaning: "per-condition weight (conditioning scale)" },
            { symbol: "c_i", meaning: "i-th conditioning signal" },
          ],
        },
      ],
    },
    {
      title: "IP-Adapter & T2I-Adapter",
      content:
        "IP-Adapter (Image Prompt Adapter) enables image-based conditioning by adding a separate cross-attention mechanism for image features alongside the existing text cross-attention. The image features from a CLIP image encoder are projected and used as K,V in dedicated cross-attention layers, while text K,V go through the original layers. T2I-Adapter is a lightweight alternative: simple feature addition at specific U-Net resolutions without cross-attention modification. Both allow combining text and image prompts for style transfer, character consistency, and composition control.",
      equations: [
        {
          label: "IP-Adapter Dual Cross-Attention",
          tex: "z' = \\text{Attn}(Q, K_{\\text{text}}, V_{\\text{text}}) + \\lambda \\cdot \\text{Attn}(Q, K_{\\text{img}}, V_{\\text{img}})",
          variables: [
            { symbol: "K_text, V_text", meaning: "from CLIP text encoder (original)" },
            { symbol: "K_img, V_img", meaning: "from CLIP image encoder (new adapter)" },
            { symbol: "λ", meaning: "image prompt strength (typically 0.5–1.0)" },
          ],
        },
      ],
    },
    {
      title: "Video Diffusion Models",
      content:
        "Video generation extends image diffusion to the temporal domain. Three dominant architectures: (1) Stable Video Diffusion (SVD) inflates the 2D U-Net to 3D by inserting temporal attention layers after each spatial attention block — spatial convolutions process each frame, temporal convolutions model inter-frame coherence. (2) Sora uses a Diffusion Transformer (DiT) that processes spacetime patches — the video is divided into 3D patches (τ×P×P), flattened, and processed by a full transformer with adaptive layer norm conditioned on timestep and text. (3) AnimateDiff adds plug-and-play temporal attention modules to frozen SD checkpoints, training only the motion module on video data. Key challenge: temporal consistency vs diversity.",
      equations: [
        {
          label: "Temporal Attention (SVD)",
          tex: "z_f' = z_f + \\text{Attn}(Q_f, [K_1,...,K_F], [V_1,...,V_F])",
          variables: [
            { symbol: "z_f", meaning: "features at frame f" },
            { symbol: "F", meaning: "total number of frames (14-25 typically)" },
            { symbol: "Q_f, K_f, V_f", meaning: "projected from frame features" },
          ],
        },
        {
          label: "Spacetime Patching (Sora/DiT)",
          tex: "p_{t,h,w} = \\text{video}[t{:}t{+}\\tau, h{:}h{+}P, w{:}w{+}P] \\in \\mathbb{R}^{\\tau \\times P^2 \\times C}",
          variables: [
            { symbol: "τ", meaning: "temporal patch size (number of frames per patch)" },
            { symbol: "P", meaning: "spatial patch size (e.g., 2×2 pixels)" },
          ],
        },
        {
          label: "DiT Block (Adaptive LayerNorm)",
          tex: "z' = z + \\text{Attn}(\\text{adaLN}(z; \\gamma(t,c), \\beta(t,c)))",
          variables: [
            { symbol: "γ, β", meaning: "scale and shift from timestep t and conditioning c" },
          ],
        },
      ],
    },
    {
      title: "Consistency Models & Distillation",
      content:
        "Consistency models (Song et al.) learn a function that maps any point on the diffusion ODE trajectory directly to x₀ in a single step. The consistency function f(x_t, t) must satisfy self-consistency: f(x_t, t) = f(x_t', t') for any two points on the same ODE trajectory. Training via consistency distillation from a pre-trained diffusion model, or direct consistency training without a teacher. Progressive distillation (Salimans & Ho) halves the number of steps iteratively: 1000 → 500 → 250 → ... → 1 step. LCM (Latent Consistency Models) apply consistency distillation in latent space for 1-4 step Stable Diffusion.",
      equations: [
        {
          label: "Consistency Condition",
          tex: "f(x_t, t) = f(x_{t'}, t') \\quad \\forall\\, t, t' \\text{ on same ODE trajectory}",
        },
        {
          label: "Consistency Distillation Loss",
          tex: "\\mathcal{L}_{CD} = \\mathbb{E}\\left[d\\left(f_\\theta(x_{t_{n+1}}, t_{n+1}),\\, f_{\\theta^-}(\\hat{x}_{t_n}, t_n)\\right)\\right]",
          variables: [
            { symbol: "θ⁻", meaning: "EMA of θ (target network)" },
            { symbol: "x̂_{t_n}", meaning: "one-step ODE estimate from x_{t_{n+1}}" },
            { symbol: "d(·,·)", meaning: "distance metric (LPIPS or L2)" },
          ],
        },
      ],
    },
    {
      title: "Evaluation Metrics — FID, IS & CLIPScore",
      content:
        "FID (Fréchet Inception Distance) measures the distance between real and generated image distributions in InceptionV3 feature space as the Fréchet distance between two fitted Gaussians. Lower FID = more similar. Need ≥10K samples for stability. Inception Score (IS) measures both sharpness (p(y|x) should be peaked) and diversity (p(y) should be uniform). CLIPScore measures text-image alignment via cosine similarity in CLIP embedding space. LPIPS measures perceptual similarity using deep features, matching human judgment better than MSE/SSIM. FVD (Fréchet Video Distance) extends FID to video using I3D features.",
      equations: [
        {
          label: "Fréchet Inception Distance",
          tex: "\\text{FID} = \\|\\mu_r - \\mu_g\\|^2 + \\text{Tr}(\\Sigma_r + \\Sigma_g - 2(\\Sigma_r \\Sigma_g)^{1/2})",
        },
        {
          label: "Inception Score",
          tex: "\\text{IS} = \\exp\\left(\\mathbb{E}_{x \\sim p_g}[\\text{KL}(p(y|x) \\| p(y))]\\right)",
        },
        {
          label: "CLIPScore",
          tex: "\\text{CLIPScore}(c, v) = w \\cdot \\max(\\cos(E_{\\text{text}}(c), E_{\\text{vis}}(v)), 0)",
          variables: [
            { symbol: "w = 2.5", meaning: "calibration constant" },
          ],
        },
        {
          label: "Fréchet Video Distance",
          tex: "\\text{FVD} = \\|\\mu_r - \\mu_g\\|^2 + \\text{Tr}(\\Sigma_r + \\Sigma_g - 2(\\Sigma_r \\Sigma_g)^{1/2}) \\quad \\text{(I3D features)}",
        },
      ],
    },
    {
      title: "Real-World Applications",
      content:
        "Generative vision models power text-to-image synthesis (Stable Diffusion, DALL-E 3, Midjourney), image editing (inpainting, outpainting, style transfer), video generation (Sora, Runway Gen-3, Stable Video Diffusion), 3D asset creation (Zero-1-to-3, DreamFusion), data augmentation for training, drug molecule design, medical image synthesis, and creative tools. ControlNet enables precise structural control. LoRA enables personal style adaptation with just 5 photos (DreamBooth). IP-Adapter allows image-prompt conditioning. AnimateDiff makes any SD model generate video. Understanding these models is essential for evaluating quality, detecting deepfakes, and building responsible AI systems.",
    },
  ],
  algorithms: [
    {
      name: "Stable Diffusion Inference Pipeline",
      steps: [
        { step: "Tokenize Prompt", detail: "CLIP tokenizer converts text to 77 token sequence; pad/truncate as needed" },
        { step: "Text Encoding", detail: "CLIP ViT-L/14 text encoder produces c ∈ ℝ^{77×768}; also encode empty string for CFG" },
        { step: "Sample Noise", detail: "Sample xT ~ N(0,I) ∈ ℝ^{4×64×64} in latent space; seed controls reproducibility" },
        { step: "DDIM Denoising Loop", detail: "For each timestep: predict noise ε_θ, apply CFG with w=7.5, compute x₀_pred, step to xₜ₋Δ" },
        { step: "VAE Decode", detail: "Decode final latent z₀ ∈ ℝ^{4×64×64} to image ∈ ℝ^{3×512×512} via pre-trained decoder" },
      ],
    },
    {
      name: "ControlNet Training",
      steps: [
        { step: "Prepare Conditioning", detail: "Extract condition maps (Canny/depth/pose) from training images" },
        { step: "Clone Encoder", detail: "Copy U-Net encoder blocks; init zero-conv layers to zero" },
        { step: "Freeze Original", detail: "Lock all weights of the original SD U-Net" },
        { step: "Train Adapter", detail: "Train the cloned encoder + zero-convs on (image, condition, caption) triplets" },
        { step: "Inference", detail: "Add ControlNet output to locked U-Net features at each resolution" },
      ],
    },
    {
      name: "Video Diffusion Pipeline",
      steps: [
        { step: "Encode Conditioning", detail: "Encode text prompt (+ optional reference image) for conditioning" },
        { step: "Init Noise Volume", detail: "Sample z_T ∈ ℝ^{F×C×H×W} noise for F frames in latent space" },
        { step: "Spatial Denoising", detail: "2D convolution + spatial attention per frame (shared weights)" },
        { step: "Temporal Attention", detail: "Cross-frame attention for temporal coherence (trained on video data)" },
        { step: "Frame-wise Decode", detail: "VAE decode each latent frame to pixel space independently" },
      ],
    },
    {
      name: "GAN Training Loop",
      steps: [
        { step: "Sample Real & Fake", detail: "Get real batch from dataset; sample z ~ N(0,1) and generate fake batch G(z)" },
        { step: "Train Discriminator", detail: "Maximize log D(x_real) + log(1 - D(G(z))); update D parameters only" },
        { step: "Train Generator", detail: "Minimize -log D(G(z)) using non-saturating loss; update G parameters only" },
        { step: "Monitor FID", detail: "Periodically compute FID against real data to track training quality" },
      ],
    },
  ],
  papers: [
    { year: 2013, title: "VAE", authors: "Kingma & Welling", venue: "ICLR", summary: "Variational Autoencoders — learning latent representations via ELBO maximization." },
    { year: 2014, title: "GAN", authors: "Goodfellow et al.", venue: "NeurIPS", summary: "Generative Adversarial Networks — two-player minimax game for image generation." },
    { year: 2017, title: "WGAN-GP", authors: "Gulrajani et al.", venue: "NeurIPS", summary: "Wasserstein GAN with gradient penalty — stable training without mode collapse." },
    { year: 2020, title: "DDPM", authors: "Ho et al.", venue: "NeurIPS", summary: "Denoising Diffusion Probabilistic Models — noise prediction objective for high-quality generation." },
    { year: 2020, title: "Score-Based SDE", authors: "Song et al.", venue: "ICLR", summary: "Score-based generative modeling through stochastic differential equations." },
    { year: 2021, title: "DDIM", authors: "Song et al.", venue: "ICLR", summary: "Implicit diffusion models enabling deterministic, accelerated sampling." },
    { year: 2021, title: "Improved DDPM", authors: "Nichol & Dhariwal", venue: "ICML", summary: "Cosine noise schedule and learned variance for better sample quality." },
    { year: 2022, title: "Stable Diffusion", authors: "Rombach et al.", venue: "CVPR", summary: "Latent Diffusion Models — diffusion in compressed latent space with text conditioning." },
    { year: 2023, title: "ControlNet", authors: "Zhang et al.", venue: "ICCV", summary: "Adding spatial control (edges, depth, pose) to pre-trained diffusion models." },
    { year: 2023, title: "IP-Adapter", authors: "Ye et al.", venue: "arXiv", summary: "Image prompt adapter via decoupled cross-attention for image conditioning." },
    { year: 2023, title: "Consistency Models", authors: "Song et al.", venue: "ICML", summary: "Single-step generation via consistency function on ODE trajectories." },
    { year: 2023, title: "SDXL", authors: "Podell et al.", venue: "arXiv", summary: "Improved Stable Diffusion with larger U-Net, dual text encoders, and refiner." },
    { year: 2023, title: "Flow Matching", authors: "Lipman et al.", venue: "ICLR", summary: "Simpler alternative to diffusion via ODE velocity field regression." },
    { year: 2023, title: "AnimateDiff", authors: "Guo et al.", venue: "ICLR", summary: "Plug-and-play motion module for animating personalized text-to-image models." },
    { year: 2023, title: "SVD", authors: "Blattmann et al.", venue: "arXiv", summary: "Stable Video Diffusion: temporal attention insertion for image-to-video." },
    { year: 2024, title: "Sora", authors: "OpenAI", venue: "Technical Report", summary: "Spacetime patches + DiT for variable-duration, variable-resolution video generation." },
    { year: 2024, title: "Stable Diffusion 3", authors: "Stability AI", venue: "arXiv", summary: "Multimodal diffusion transformer with rectified flow matching." },
    { year: 2024, title: "LCM", authors: "Luo et al.", venue: "arXiv", summary: "Latent Consistency Models: 1-4 step generation via consistency distillation." },
  ],
};
