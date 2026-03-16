import { ModuleContent } from "./moduleContent";

export const generativeVisionModule: ModuleContent = {
  id: "generative-vision",
  title: "Generative Models for Vision",
  subtitle: "From VAEs to Stable Diffusion — understand the math and architectures behind image generation, GANs, diffusion models, and conditional synthesis.",
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
      title: "DDPM — Denoising Diffusion Probabilistic Models",
      content:
        "Diffusion models learn to reverse a Markov chain that gradually adds Gaussian noise over T steps. The forward process q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI) with β increasing from 1e-4 to 0.02 over T=1000 steps. The key reparameterization: q(xₜ|x₀) = N(xₜ; √ᾱₜx₀, (1-ᾱₜ)I) gives a closed-form expression for any timestep directly. The simplified training objective just predicts the noise ε added at each step, which is much more stable than predicting the mean.",
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
      ],
    },
    {
      title: "DDIM & Flow Matching",
      content:
        "DDIM (Denoising Diffusion Implicit Models) provides deterministic sampling by setting the noise parameter σₜ = 0, enabling the same xT to always produce the same image. This allows 50-step sampling instead of 1000 steps. Flow Matching (Lipman et al.) offers a simpler alternative: learn a velocity field v_θ(x,t) that moves samples along an ODE from noise to data. The conditional flow uses constant velocity v = x₁ - x₀, giving a simpler loss than DDPM. Classifier-Free Guidance (CFG) combines conditional and unconditional scores to amplify prompt alignment.",
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
          tex: "\\mathcal{L}_{FM} = \\mathbb{E}\\|v_\\theta(x_t, t) - (x_1 - x_0)\\|^2",
        },
      ],
    },
    {
      title: "Latent Diffusion & Stable Diffusion",
      content:
        "Latent Diffusion Models (LDM) perform diffusion in a compressed latent space instead of pixel space. An image x ∈ ℝ^{512×512×3} is encoded to z ∈ ℝ^{64×64×4} via a pre-trained VQ-VAE encoder (f=8 compression), reducing compute by 64×. The U-Net denoiser uses: (1) ResBlocks with time embedding via AdaGN, (2) self-attention over spatial features, (3) cross-attention with Q from features and K,V from CLIP text tokens to inject text conditioning. ControlNet adds structural conditioning (edges, depth, pose) by training a copy of the encoder with zero-convolution initialization.",
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
          label: "ControlNet",
          tex: "y_{\\text{ctrl}} = f_{\\text{locked}}(x) + \\text{zero\\_conv}(f_{\\text{trainable}}(x, c))",
        },
      ],
    },
    {
      title: "Evaluation Metrics — FID, IS & CLIPScore",
      content:
        "FID (Fréchet Inception Distance) measures the distance between real and generated image distributions in InceptionV3 feature space as the Fréchet distance between two fitted Gaussians. Lower FID = more similar. Need ≥10K samples for stability. Inception Score (IS) measures both sharpness (p(y|x) should be peaked) and diversity (p(y) should be uniform). CLIPScore measures text-image alignment via cosine similarity in CLIP embedding space. LPIPS measures perceptual similarity using deep features, matching human judgment better than MSE/SSIM.",
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
      ],
    },
    {
      title: "Real-World Applications",
      content:
        "Generative vision models power text-to-image synthesis (Stable Diffusion, DALL-E 3, Midjourney), image editing (inpainting, outpainting, style transfer), video generation (Sora, Runway), 3D asset creation, data augmentation for training, drug molecule design, medical image synthesis, and creative tools. LoRA enables personal style adaptation with just 5 photos (DreamBooth). IP-Adapter allows image-prompt conditioning. Understanding these models is essential for evaluating quality, detecting deepfakes, and building responsible AI systems.",
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
    { year: 2021, title: "DDIM", authors: "Song et al.", venue: "ICLR", summary: "Implicit diffusion models enabling deterministic, accelerated sampling." },
    { year: 2022, title: "Stable Diffusion", authors: "Rombach et al.", venue: "CVPR", summary: "Latent Diffusion Models — diffusion in compressed latent space with text conditioning." },
    { year: 2023, title: "ControlNet", authors: "Zhang et al.", venue: "ICCV", summary: "Adding spatial control (edges, depth, pose) to pre-trained diffusion models." },
    { year: 2023, title: "SDXL", authors: "Podell et al.", venue: "arXiv", summary: "Improved Stable Diffusion with larger U-Net, dual text encoders, and refiner." },
    { year: 2023, title: "Flow Matching", authors: "Lipman et al.", venue: "ICLR", summary: "Simpler alternative to diffusion via ODE velocity field regression." },
    { year: 2024, title: "Stable Diffusion 3", authors: "Stability AI", venue: "arXiv", summary: "Multimodal diffusion transformer with rectified flow matching." },
  ],
};
