import ModulePage from "@/components/ModulePage";
import { generativeVisionModule } from "@/data/generativeVisionModuleData";
import { MathEquation } from "@/components/MathBlock";
import AITutor from "@/components/AITutor";
import {
  VAEElboViz, GANMinimaxViz, DiffusionTimeline, StableDiffusionPipelineViz,
  NoiseScheduleViz, ControlNetViz, VideoGenPipelineViz, ScoreFunctionViz,
} from "@/components/GenerativeCanvasAnimations";
import { ArrowLeft, GraduationCap, Sparkles, Layers, Zap, Paintbrush, BarChart3, Wand2, Film, SlidersHorizontal, Target } from "lucide-react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

const color = generativeVisionModule.color;

const theoryByTitle: Record<string, typeof generativeVisionModule.theory[0]> = {};
generativeVisionModule.theory.forEach(s => { theoryByTitle[s.title] = s; });

function TheoryInline({ title }: { title: string }) {
  const section = theoryByTitle[title];
  if (!section) return null;
  return (
    <div className="concept-card">
      <div className="flex items-center flex-wrap gap-y-1 mb-3">
        <h3 className="font-semibold text-foreground text-sm">{section.title}</h3>
        <AITutor conceptTitle={section.title} conceptContent={section.content} moduleName="Generative Vision" />
      </div>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">{section.content}</p>
      {section.equations?.map((eq) => (
        <div key={eq.label} className="mb-3">
          <MathEquation tex={eq.tex} label={eq.label} />
          {eq.variables && eq.variables.length > 0 && (
            <div className="mt-1.5 rounded-lg bg-muted/30 border border-border p-3">
              <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">Where</p>
              <div className="space-y-0.5">
                {eq.variables.map((v: any) => (
                  <p key={v.symbol} className="text-xs text-muted-foreground">
                    <span className="font-mono text-foreground">{v.symbol}</span> = {v.meaning}
                  </p>
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function ContentCard({ title, children, accent }: { title: string; children: React.ReactNode; accent?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card/50 p-4">
      <p className="text-[10px] font-semibold uppercase tracking-wider mb-2" style={{ color: accent || `hsl(${color})` }}>{title}</p>
      <div className="text-sm text-muted-foreground leading-relaxed">{children}</div>
    </div>
  );
}

function SectionHeader({ icon: Icon, title, number, subtitle }: { icon: any; title: string; number: number; subtitle?: string }) {
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="flex items-start gap-4 mb-6">
      <div className="h-10 w-10 rounded-xl flex items-center justify-center shrink-0 mt-1" style={{ backgroundColor: `hsl(${color} / 0.12)` }}>
        <Icon className="h-5 w-5" style={{ color: `hsl(${color})` }} />
      </div>
      <div>
        <p className="text-[10px] font-mono font-bold uppercase tracking-widest mb-1" style={{ color: `hsl(${color})` }}>Part {number}</p>
        <h2 className="text-xl font-bold text-foreground tracking-tight">{title}</h2>
        {subtitle && <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{subtitle}</p>}
      </div>
    </motion.div>
  );
}

export default function GenerativeVisionModule() {
  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      <div className="flex items-start gap-4 mb-8">
        <div className="h-12 w-12 rounded-xl flex items-center justify-center shrink-0" style={{ backgroundColor: `hsl(${color} / 0.12)` }}>
          <GraduationCap className="h-6 w-6" style={{ color: `hsl(${color})` }} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground tracking-tight">{generativeVisionModule.title}</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{generativeVisionModule.subtitle}</p>
        </div>
      </div>

      {/* Learning flow nav */}
      <div className="rounded-xl border border-border bg-muted/30 p-4 mb-8">
        <h2 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3">Structured Learning Flow</h2>
        <div className="grid sm:grid-cols-3 lg:grid-cols-9 gap-2">
          {[
            { id: "vae", icon: "🧬", label: "VAE & VQ-VAE" },
            { id: "gan", icon: "⚔️", label: "GANs & WGAN" },
            { id: "score", icon: "📐", label: "Score Functions" },
            { id: "diffusion", icon: "🌊", label: "DDPM" },
            { id: "schedules", icon: "📊", label: "Noise Schedules" },
            { id: "sampling", icon: "⚡", label: "DDIM & Flow" },
            { id: "latent", icon: "🎨", label: "Stable Diffusion" },
            { id: "controlnet", icon: "🎛️", label: "ControlNet" },
            { id: "video", icon: "🎬", label: "Video Gen" },
          ].map((item) => (
            <a key={item.id} href={`#${item.id}`} className="rounded-lg border border-border bg-card p-2 hover:border-primary/40 transition-colors text-center">
              <p className="text-sm mb-0.5">{item.icon}</p>
              <p className="text-[9px] text-foreground font-medium">{item.label}</p>
            </a>
          ))}
        </div>
      </div>

      <div className="space-y-12">

        {/* ═══ Part 1: VAE ═══ */}
        <section id="vae">
          <SectionHeader icon={Sparkles} title="Variational Autoencoders" number={1} subtitle="Learn smooth latent representations via the ELBO — the mathematical foundation of latent generative models." />
          <div className="space-y-4">
            <TheoryInline title="Intuition" />
            <VAEElboViz />
            <TheoryInline title="VAE — Variational Autoencoders" />

            <ContentCard title="Worked Example — VAE Forward Pass" accent="#ec4899">
              <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                <p>Encoder: μ = [0.5, -0.3], log σ² = [-0.2, 0.1]</p>
                <p>σ = [exp(-0.1), exp(0.05)] = [0.905, 1.051]</p>
                <p>ε ~ N(0,I) = [0.8, -0.5]</p>
                <p>z = μ + σ⊙ε = [0.5+0.905×0.8, -0.3+1.051×(-0.5)] = [1.224, -0.826]</p>
                <p>KL = ½·Σ(μ²+σ²-log σ²-1) = ½·[(0.25+0.819+0.2-1)+(0.09+1.105-0.1-1)] = 0.182</p>
              </div>
            </ContentCard>

            <TheoryInline title="VQ-VAE — Discrete Latent Codes" />
          </div>
        </section>

        {/* ═══ Part 2: GANs ═══ */}
        <section id="gan">
          <SectionHeader icon={Zap} title="Generative Adversarial Networks" number={2} subtitle="Two-player minimax game: Generator vs Discriminator — from vanilla GANs to Wasserstein formulation." />
          <div className="space-y-4">
            <GANMinimaxViz />
            <TheoryInline title="GANs — Adversarial Training" />
            <TheoryInline title="WGAN — Wasserstein Distance" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="GAN Failure Modes" accent="#ec4899">
                <strong className="text-foreground">Mode collapse:</strong> G produces limited diversity — maps many z to same output.<br /><br />
                <strong className="text-foreground">Training instability:</strong> D too strong → vanishing gradients for G. D too weak → G gets no learning signal.<br /><br />
                <strong className="text-foreground">WGAN fix:</strong> Wasserstein distance gives continuous gradients even when p_data and p_G have no overlap (unlike JSD = constant log 2).
              </ContentCard>
              <ContentCard title="StyleGAN2 Architecture" accent="#38bdf8">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">z → W-space:</span> 8-layer mapping network → w (more disentangled)</p>
                  <p><span className="text-foreground font-medium">AdaIN:</span> Adaptive instance normalization per layer — style injection</p>
                  <p><span className="text-foreground font-medium">Path length reg:</span> ‖J_w^T · y‖₂ ≈ constant → smooth latent space</p>
                  <p><span className="text-foreground font-medium">Style mixing:</span> different w at each resolution for compositional control</p>
                  <p className="text-primary mt-1">FID=2.84 on FFHQ 1024×1024</p>
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 3: Score Functions ═══ */}
        <section id="score">
          <SectionHeader icon={Target} title="Score Functions & Tweedie's Formula" number={3} subtitle="The mathematical backbone — learn the gradient of log-density. Unifies noise prediction, denoising, and score matching." />
          <div className="space-y-4">
            <ScoreFunctionViz />
            <TheoryInline title="Score Functions & Tweedie's Formula" />

            <div className="grid md:grid-cols-3 gap-4">
              <ContentCard title="Three Equivalent Views" accent="#a855f7">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">ε-prediction:</span> ε_θ(x_t, t) ≈ ε</p>
                  <p><span className="text-foreground font-medium">Score prediction:</span> s_θ = -ε_θ/σ_t</p>
                  <p><span className="text-foreground font-medium">x₀-prediction:</span> x̂₀ = (x_t - σ_t·ε_θ)/α_t</p>
                  <p className="text-primary mt-1">All mathematically equivalent!</p>
                </div>
              </ContentCard>
              <ContentCard title="Why Score Matching Works" accent="#06b6d4">
                <p className="text-xs">Unlike MLE, score matching doesn't need the normalizing constant Z of p(x). The score ∇ log p(x) = ∇ log p̃(x) since ∇ log Z = 0.</p>
                <p className="text-xs mt-1">Denoising score matching: add noise → learn to denoise = learn the score.</p>
              </ContentCard>
              <ContentCard title="Probability Flow ODE" accent="#f59e0b">
                <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  dx = [f(x,t) - ½g(t)²∇ log p_t(x)]dt
                </div>
                <p className="text-xs mt-1">Deterministic ODE with same marginals as the SDE. Enables exact likelihood computation.</p>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 4: DDPM ═══ */}
        <section id="diffusion">
          <SectionHeader icon={Layers} title="Diffusion Models (DDPM)" number={4} subtitle="Learn to reverse a noise-adding Markov chain — powering Stable Diffusion, DALL-E, and Midjourney." />
          <div className="space-y-4">
            <DiffusionTimeline />
            <TheoryInline title="DDPM — Denoising Diffusion Probabilistic Models" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Forward Process — Detailed" accent="#ec4899">
                <p className="mb-2">βₜ increases linearly from β₁ = 1e-4 to β_T = 0.02 over T=1000 steps.</p>
                <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                  <p>q(xₜ|xₜ₋₁) = N(√(1-βₜ)xₜ₋₁, βₜI)</p>
                  <p>Closed-form: xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε</p>
                  <p>{"ᾱₜ = Π_{s=1}^t (1-βₛ) → monotonically ↓"}</p>
                  <p>SNR(t) = ᾱₜ/(1-ᾱₜ) → 0 as t → T</p>
                  <p>At t=T: ᾱ_T ≈ 0, x_T ≈ N(0,I)</p>
                </div>
              </ContentCard>
              <ContentCard title="Reverse Process — Posterior" accent="#38bdf8">
                <p className="mb-2">The true posterior q(x_{t-1}|x_t,x₀) is Gaussian and tractable:</p>
                <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                  <p>q(xₜ₋₁|xₜ,x₀) = N(μ̃ₜ, β̃ₜI)</p>
                  <p>μ̃ₜ = (√ᾱₜ₋₁·βₜ·x₀ + √αₜ·(1-ᾱₜ₋₁)·xₜ)/(1-ᾱₜ)</p>
                  <p>β̃ₜ = βₜ·(1-ᾱₜ₋₁)/(1-ᾱₜ)</p>
                  <p>Substitute x̂₀ = (xₜ−√(1−ᾱₜ)·ε_θ)/√ᾱₜ</p>
                </div>
              </ContentCard>
            </div>

            <ContentCard title="Worked Example — Single Denoising Step" accent="#f59e0b">
              <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                <p>Given: xₜ at t=500, ᾱ₅₀₀ = 0.044, β₅₀₀ = 0.011</p>
                <p>1. Network predicts: ε_θ(xₜ, 500) = [-0.3, 0.7, ...] (noise estimate)</p>
                <p>2. Predict x₀: x̂₀ = (xₜ - √(1-0.044)·ε_θ) / √0.044 = (xₜ - 0.978·ε_θ) / 0.210</p>
                <p>3. Compute posterior mean: μ̃ = f(x̂₀, xₜ, ᾱₜ, ᾱₜ₋₁, βₜ)</p>
                <p>4. Sample: xₜ₋₁ ~ N(μ̃, β̃·I) — add small noise for stochastic generation</p>
              </div>
            </ContentCard>
          </div>
        </section>

        {/* ═══ Part 5: Noise Schedules ═══ */}
        <section id="schedules">
          <SectionHeader icon={SlidersHorizontal} title="Noise Schedules" number={5} subtitle="How signal decays over time — linear, cosine, and scaled-linear schedules with v-prediction." />
          <div className="space-y-4">
            <NoiseScheduleViz />
            <TheoryInline title="Noise Schedules — Linear, Cosine & Scaled-Linear" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Schedule Comparison" accent="#a855f7">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">Linear (DDPM):</span> β ∈ [1e-4, 0.02]. Fast decay — signal near zero by t=600/1000.</p>
                  <p><span className="text-foreground font-medium">Cosine (iDDPM):</span> More gradual. Better intermediate SNR. Default for small images.</p>
                  <p><span className="text-foreground font-medium">Scaled-Linear (SDXL):</span> β_t = (√β_min + t(√β_max−√β_min))². Better for high-res.</p>
                  <p><span className="text-foreground font-medium">SD3 / Flux:</span> Rectified flow (t ∈ [0,1]) with logit-normal timestep sampling.</p>
                </div>
              </ContentCard>
              <ContentCard title="v-Prediction Parameterization" accent="#06b6d4">
                <p className="text-xs mb-2">Instead of predicting ε or x₀, predict the "velocity" v = α_t·ε − σ_t·x₀:</p>
                <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                  <p>v_t = √ᾱ_t · ε − √(1−ᾱ_t) · x₀</p>
                  <p>At SNR→∞: v ≈ ε (noise prediction)</p>
                  <p>At SNR→0: v ≈ −x₀ (signal prediction)</p>
                  <p>Smooth interpolation between regimes</p>
                </div>
                <p className="text-xs mt-1">Used in SD 2.x, SDXL refiner. Better for progressive distillation.</p>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 6: DDIM & Flow ═══ */}
        <section id="sampling">
          <SectionHeader icon={Wand2} title="DDIM, Flow Matching & Guidance" number={6} subtitle="Accelerated deterministic sampling, ODE-based generation, rectified flow, and classifier-free guidance." />
          <div className="space-y-4">
            <TheoryInline title="DDIM & Flow Matching" />

            <ContentCard title="Worked Example — SD Inference (50 DDIM Steps)" accent="#f59e0b">
              <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                <p>1. Tokenize: "astronaut riding horse on Mars" → 12 CLIP tokens (padded to 77)</p>
                <p>2. Encode: CLIP ViT-L/14 → c ∈ ℝ^(77×768); also encode "" → c_∅ (for CFG)</p>
                <p>3. Sample: x_T ~ N(0,I) ∈ ℝ^(4×64×64) in latent space</p>
                <p>4. Loop t=T→0 (50 uniform steps):</p>
                <p>   a) ε_uncond = UNet(xₜ, t, c_∅)</p>
                <p>   b) ε_cond = UNet(xₜ, t, c)</p>
                <p>   c) ε_guided = ε_uncond + 7.5·(ε_cond - ε_uncond)  [CFG]</p>
                <p>   d) x̂₀ = (xₜ - √(1-ᾱₜ)·ε_guided) / √ᾱₜ</p>
                <p>   e) xₜ₋₁ = √ᾱₜ₋₁·x̂₀ + √(1-ᾱₜ₋₁)·ε_guided  [DDIM, σ=0]</p>
                <p>5. Decode: VAE decoder z₀ → image ∈ ℝ^(3×512×512)</p>
              </div>
            </ContentCard>

            <TheoryInline title="Consistency Models & Distillation" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Sampling Speed Comparison" accent="#a855f7">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">DDPM:</span> 1000 steps (~60s on A100)</p>
                  <p><span className="text-foreground font-medium">DDIM:</span> 50 steps (~3s)</p>
                  <p><span className="text-foreground font-medium">DPM-Solver:</span> 20 steps (~1.2s)</p>
                  <p><span className="text-foreground font-medium">LCM:</span> 4 steps (~0.3s)</p>
                  <p><span className="text-foreground font-medium">Consistency:</span> 1-2 steps (~0.15s)</p>
                  <p><span className="text-foreground font-medium">SDXL Turbo:</span> 1 step (~0.1s)</p>
                </div>
              </ContentCard>
              <ContentCard title="Rectified Flow (SD3 / Flux)" accent="#06b6d4">
                <p className="text-xs">Linear interpolation path: x_t = (1-t)x₀ + t·ε</p>
                <p className="text-xs mt-1">Learn velocity: v_θ(x_t,t) ≈ ε − x₀</p>
                <p className="text-xs mt-1"><strong className="text-foreground">Reflow:</strong> Iteratively straighten ODE paths by sampling (x₀, x₁) pairs from the current model, then re-training on straight paths.</p>
                <p className="text-xs mt-1 text-primary">Result: nearly straight paths → 1-4 step generation</p>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 7: Stable Diffusion ═══ */}
        <section id="latent">
          <SectionHeader icon={Paintbrush} title="Latent Diffusion & Stable Diffusion" number={7} subtitle="Diffusion in compressed latent space — 64× compute savings via VQ-VAE encoding." />
          <div className="space-y-4">
            <StableDiffusionPipelineViz />
            <TheoryInline title="Latent Diffusion & Stable Diffusion" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="U-Net Architecture (Deep Dive)" accent="#ec4899">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">Time embedding:</span> sinusoidal → MLP → AdaGN scale/shift per ResBlock</p>
                  <p><span className="text-foreground font-medium">Self-Attn:</span> Q=K=V from spatial features (captures spatial coherence)</p>
                  <p><span className="text-foreground font-medium">Cross-Attn:</span> Q=features, K=V=CLIP tokens (injects text meaning)</p>
                  <p><span className="text-foreground font-medium">Down:</span> strided conv 2× at resolutions 64→32→16→8</p>
                  <p><span className="text-foreground font-medium">Mid:</span> ResBlock + Self-Attn + Cross-Attn at 8×8</p>
                  <p><span className="text-foreground font-medium">Up:</span> bilinear 2× + concat skip + ResBlock at 8→16→32→64</p>
                  <p className="text-primary mt-1">SD 1.5: 860M params · SDXL: 2.6B params</p>
                </div>
              </ContentCard>
              <ContentCard title="SDXL Improvements" accent="#38bdf8">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">Dual encoders:</span> CLIP ViT-L + OpenCLIP ViT-bigG (better text understanding)</p>
                  <p><span className="text-foreground font-medium">Larger U-Net:</span> 2.6B params, attention at 32×32 (not 64)</p>
                  <p><span className="text-foreground font-medium">Refiner:</span> separate model for detail enhancement (img2img at high denoise)</p>
                  <p><span className="text-foreground font-medium">Micro-conditioning:</span> crop coords + target size as extra conditioning</p>
                  <p><span className="text-foreground font-medium">Resolution:</span> native 1024×1024 (vs 512 for SD 1.x)</p>
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 8: ControlNet ═══ */}
        <section id="controlnet">
          <SectionHeader icon={SlidersHorizontal} title="ControlNet & Conditioning Adapters" number={8} subtitle="Zero-convolution architecture for structural control — edges, depth, pose, segmentation maps, and image prompts." />
          <div className="space-y-4">
            <ControlNetViz />
            <TheoryInline title="ControlNet — Structural Conditioning" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Zero-Convolution — Why It Works" accent="#a855f7">
                <p className="text-xs mb-2">Standard fine-tuning destroys pre-trained features. Zero-conv solves this:</p>
                <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                  <p>Step 1: Clone U-Net encoder → trainable copy</p>
                  <p>Step 2: Connect via 1×1 conv with W=0, b=0</p>
                  <p>Step 3: At init: output = locked_model(x) + 0</p>
                  <p>Step 4: Gradients flow → zero-conv learns scale</p>
                  <p>Step 5: Gradually ControlNet contribution grows</p>
                </div>
                <p className="text-xs mt-1 text-primary">Training: only ~300M trainable params (encoder copy + zero-convs)</p>
              </ContentCard>
              <ContentCard title="Conditioning Types" accent="#06b6d4">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">Canny Edge:</span> binary edges → preserves composition, allows restyling</p>
                  <p><span className="text-foreground font-medium">Depth (MiDaS):</span> monocular depth → 3D-aware generation</p>
                  <p><span className="text-foreground font-medium">OpenPose:</span> skeleton keypoints → pose transfer</p>
                  <p><span className="text-foreground font-medium">Segmentation:</span> semantic map → region-level material control</p>
                  <p><span className="text-foreground font-medium">Normal Map:</span> surface normals → lighting-aware generation</p>
                  <p><span className="text-foreground font-medium">Scribble:</span> rough sketch → artistic interpretation</p>
                </div>
              </ContentCard>
            </div>

            <TheoryInline title="IP-Adapter & T2I-Adapter" />

            <ContentCard title="LoRA & DreamBooth" accent="#f59e0b">
              <div className="grid md:grid-cols-2 gap-4 text-xs">
                <div>
                  <p className="text-foreground font-medium mb-1">LoRA (Low-Rank Adaptation)</p>
                  <div className="font-mono text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                    <p>W' = W₀ + B·A where B∈ℝ^{"{d×r}"}, A∈ℝ^{"{r×k}"}</p>
                    <p>Applied to Q,K,V,O projections in cross-attn</p>
                    <p>r=4: 99.9% savings, r=64: 99.2%</p>
                    <p>Merge at inference: zero overhead</p>
                  </div>
                </div>
                <div>
                  <p className="text-foreground font-medium mb-1">DreamBooth</p>
                  <div className="font-mono text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                    <p>Fine-tune on ~5 subject photos</p>
                    <p>Unique token: "photo of [V] dog"</p>
                    <p>Prior preservation loss prevents forgetting</p>
                    <p>LoRA variant: 4MB adapter vs 2GB full</p>
                  </div>
                </div>
              </div>
            </ContentCard>
          </div>
        </section>

        {/* ═══ Part 9: Video Generation ═══ */}
        <section id="video">
          <SectionHeader icon={Film} title="Video Generation Models" number={9} subtitle="From image diffusion to temporal consistency — SVD, Sora, AnimateDiff, and the frontier of video synthesis." />
          <div className="space-y-4">
            <VideoGenPipelineViz />
            <TheoryInline title="Video Diffusion Models" />

            <div className="grid md:grid-cols-3 gap-4">
              <ContentCard title="SVD — Temporal Inflation" accent="#ec4899">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">Method:</span> Insert temporal attention after each spatial block</p>
                  <p><span className="text-foreground font-medium">Init:</span> Pre-train on images → fine-tune on video</p>
                  <p><span className="text-foreground font-medium">Input:</span> Single image → 14-25 frames</p>
                  <p><span className="text-foreground font-medium">Motion:</span> fps + motion bucket conditioning</p>
                  <p className="text-primary">3D U-Net = 2D spatial + 1D temporal</p>
                </div>
              </ContentCard>
              <ContentCard title="Sora — Spacetime DiT" accent="#38bdf8">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">Architecture:</span> DiT (Diffusion Transformer)</p>
                  <p><span className="text-foreground font-medium">Patches:</span> 3D spacetime patches (τ×P×P)</p>
                  <p><span className="text-foreground font-medium">Variable:</span> any duration, resolution, aspect ratio</p>
                  <p><span className="text-foreground font-medium">Emergent:</span> 3D consistency from scale alone</p>
                  <p className="text-primary">No separate spatial/temporal — unified</p>
                </div>
              </ContentCard>
              <ContentCard title="AnimateDiff — Plug & Play" accent="#f59e0b">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">Idea:</span> Motion module added to any SD model</p>
                  <p><span className="text-foreground font-medium">Frozen:</span> All SD weights unchanged</p>
                  <p><span className="text-foreground font-medium">Trained:</span> Only temporal self-attention layers</p>
                  <p><span className="text-foreground font-medium">Compatible:</span> Any LoRA/checkpoint works</p>
                  <p className="text-primary">~25M trainable params (motion only)</p>
                </div>
              </ContentCard>
            </div>

            <ContentCard title="Video Generation Challenges & FVD" accent="#a855f7">
              <div className="grid md:grid-cols-2 gap-4 text-xs">
                <div>
                  <p className="text-foreground font-medium mb-1">Key Challenges</p>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    <li>Temporal consistency: objects shouldn't flicker/morph</li>
                    <li>Long-range coherence: narrative over 60+ seconds</li>
                    <li>Physics: gravity, collisions, fluid dynamics</li>
                    <li>Memory: F frames × 4 latent channels = F× more compute</li>
                    <li>Training data: high-quality video-caption pairs are scarce</li>
                  </ul>
                </div>
                <div>
                  <p className="text-foreground font-medium mb-1">Evaluation: FVD</p>
                  <div className="font-mono text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                    <p>FVD = FID applied to I3D video features</p>
                    <p>Measures both visual quality + temporal coherence</p>
                    <p>SVD FVD≈178 on UCF-101</p>
                    <p>Also: CLIP-FID, VBench (17 dimensions)</p>
                  </div>
                </div>
              </div>
            </ContentCard>
          </div>
        </section>

        {/* ═══ Part 10: Evaluation & Review ═══ */}
        <section id="review">
          <SectionHeader icon={BarChart3} title="Evaluation Metrics, Papers & Practice" number={10} subtitle="FID, Inception Score, CLIPScore, FVD — measuring generative quality across images and video." />
          <div className="space-y-4">
            <TheoryInline title="Evaluation Metrics — FID, IS & CLIPScore" />
            <TheoryInline title="Real-World Applications" />
            <ModulePage content={generativeVisionModule} hideHeader hideTheory />
          </div>
        </section>
      </div>
    </div>
  );
}
