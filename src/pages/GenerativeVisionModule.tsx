import ModulePage from "@/components/ModulePage";
import { generativeVisionModule } from "@/data/generativeVisionModuleData";
import { MathEquation } from "@/components/MathBlock";
import AITutor from "@/components/AITutor";
import { ArrowLeft, GraduationCap, Sparkles, Layers, Zap, Paintbrush, BarChart3, Wand2 } from "lucide-react";
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
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="flex items-start gap-4 mb-6"
    >
      <div
        className="h-10 w-10 rounded-xl flex items-center justify-center shrink-0 mt-1"
        style={{ backgroundColor: `hsl(${color} / 0.12)` }}
      >
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
        <div className="grid sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {[
            { id: "vae", icon: "🧬", label: "VAE & VQ-VAE" },
            { id: "gan", icon: "⚔️", label: "GANs & WGAN" },
            { id: "diffusion", icon: "🌊", label: "Diffusion (DDPM)" },
            { id: "sampling", icon: "⚡", label: "DDIM & Guidance" },
            { id: "latent", icon: "🎨", label: "Stable Diffusion" },
            { id: "review", icon: "📚", label: "Evaluation & Review" },
          ].map((item) => (
            <a
              key={item.id}
              href={`#${item.id}`}
              className="rounded-lg border border-border bg-card p-2.5 hover:border-primary/40 transition-colors text-center"
            >
              <p className="text-sm mb-0.5">{item.icon}</p>
              <p className="text-[10px] text-foreground font-medium">{item.label}</p>
            </a>
          ))}
        </div>
      </div>

      <div className="space-y-12">

        {/* ═══ Part 1: VAE ═══ */}
        <section id="vae">
          <SectionHeader
            icon={Sparkles}
            title="Variational Autoencoders"
            number={1}
            subtitle="Learn smooth latent representations via the ELBO — the mathematical foundation of latent generative models."
          />
          <div className="space-y-4">
            <TheoryInline title="Intuition" />
            <TheoryInline title="VAE — Variational Autoencoders" />

            <ContentCard title="Worked Example — VAE Forward Pass" accent="#ec4899">
              <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                <p>Encoder: μ = [0.5, -0.3], log σ² = [-0.2, 0.1]</p>
                <p>σ = [0.905, 1.051]</p>
                <p>ε ~ N(0,I) = [0.8, -0.5]</p>
                <p>z = μ + σ⊙ε = [1.224, -0.826]</p>
                <p>KL = ½·[(0.25+0.819+0.2-1)+(0.09+1.104-0.1-1)] = 0.182</p>
                <p>Total: L = L_recon + 1.0 × 0.182</p>
              </div>
            </ContentCard>

            <TheoryInline title="VQ-VAE — Discrete Latent Codes" />
          </div>
        </section>

        {/* ═══ Part 2: GANs ═══ */}
        <section id="gan">
          <SectionHeader
            icon={Zap}
            title="Generative Adversarial Networks"
            number={2}
            subtitle="Two-player minimax game: Generator vs Discriminator — from vanilla GANs to the stable Wasserstein formulation."
          />
          <div className="space-y-4">
            <TheoryInline title="GANs — Adversarial Training" />
            <TheoryInline title="WGAN — Wasserstein Distance" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="GAN Failure Modes" accent="#ec4899">
                <strong className="text-foreground">Mode collapse:</strong> G produces limited diversity, cycling through few outputs.<br /><br />
                <strong className="text-foreground">Training instability:</strong> D too strong → vanishing gradients for G. D too weak → no learning signal.<br /><br />
                <strong className="text-foreground">WGAN fix:</strong> Wasserstein distance gives continuous gradients even without overlap.
              </ContentCard>
              <ContentCard title="GAN vs VAE vs Diffusion" accent="#38bdf8">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">GAN:</span> Sharp samples, unstable training, no likelihood</p>
                  <p><span className="text-foreground font-medium">VAE:</span> Smooth latent space, stable, blurry samples</p>
                  <p><span className="text-foreground font-medium">Diffusion:</span> Best quality, stable, but slow sampling</p>
                  <p><span className="text-foreground font-medium">LDM:</span> Diffusion quality + fast via latent space</p>
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 3: DDPM ═══ */}
        <section id="diffusion">
          <SectionHeader
            icon={Layers}
            title="Diffusion Models (DDPM)"
            number={3}
            subtitle="Learn to reverse a noise-adding Markov chain — the mathematical foundation powering Stable Diffusion, DALL-E, and Midjourney."
          />
          <div className="space-y-4">
            <TheoryInline title="DDPM — Denoising Diffusion Probabilistic Models" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="Forward Process" accent="#ec4899">
                <p className="mb-2">βₜ increases linearly from β₁ = 1e-4 to β_T = 0.02 over T=1000 steps.</p>
                <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  q(xₜ|xₜ₋₁) = N(√(1-βₜ)xₜ₋₁, βₜI)<br />
                  As t → T: ᾱ_T → 0, x_T → N(0,I)
                </div>
              </ContentCard>
              <ContentCard title="Reverse Process" accent="#38bdf8">
                <p className="mb-2">Learn to predict the noise ε added at each step — much simpler than predicting the mean.</p>
                <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border">
                  μ_θ(xₜ,t) = (xₜ - √(1-ᾱₜ)·ε_θ) / √ᾱₜ<br />
                  Predict noise directly → stable training
                </div>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 4: DDIM & CFG ═══ */}
        <section id="sampling">
          <SectionHeader
            icon={Wand2}
            title="DDIM, Flow Matching & Guidance"
            number={4}
            subtitle="Accelerated deterministic sampling, ODE-based generation, and classifier-free guidance for steering outputs."
          />
          <div className="space-y-4">
            <TheoryInline title="DDIM & Flow Matching" />

            <ContentCard title="Worked Example — SD Inference (50 DDIM Steps)" accent="#f59e0b">
              <div className="font-mono text-xs text-foreground/70 bg-muted/40 rounded p-2 border border-border space-y-1">
                <p>1. Tokenize: "astronaut riding horse on Mars" → 12 CLIP tokens</p>
                <p>2. Encode: CLIP ViT-L/14 → c ∈ ℝ^(77×768); also encode "" → ∅</p>
                <p>3. Sample: x_T ~ N(0,I) ∈ ℝ^(4×64×64)</p>
                <p>4. Loop: ε_guided = ε(xₜ,∅) + 7.5·(ε(xₜ,c) - ε(xₜ,∅))</p>
                <p>5. Decode: VAE decoder z₀ → image ∈ ℝ^(3×512×512)</p>
              </div>
            </ContentCard>
          </div>
        </section>

        {/* ═══ Part 5: Latent Diffusion ═══ */}
        <section id="latent">
          <SectionHeader
            icon={Paintbrush}
            title="Latent Diffusion & Stable Diffusion"
            number={5}
            subtitle="Diffusion in compressed latent space — 64× compute savings via VQ-VAE encoding, U-Net denoising, and ControlNet conditioning."
          />
          <div className="space-y-4">
            <TheoryInline title="Latent Diffusion & Stable Diffusion" />

            <div className="grid md:grid-cols-2 gap-4">
              <ContentCard title="U-Net Architecture" accent="#ec4899">
                <div className="space-y-1.5 text-xs">
                  <p><span className="text-foreground font-medium">ResBlock:</span> time embedding via AdaGN scale/shift</p>
                  <p><span className="text-foreground font-medium">Self-Attn:</span> Q=K=V from spatial features</p>
                  <p><span className="text-foreground font-medium">Cross-Attn:</span> Q=features, K=V=CLIP text tokens</p>
                  <p><span className="text-foreground font-medium">Down:</span> strided conv 2× (32→16→8→4)</p>
                  <p><span className="text-foreground font-medium">Bottleneck:</span> full attention at 4×4 or 8×8</p>
                  <p><span className="text-foreground font-medium">Up:</span> bilinear 2× + skip connections</p>
                </div>
              </ContentCard>
              <ContentCard title="ControlNet & LoRA" accent="#38bdf8">
                <p className="mb-2 text-xs"><strong className="text-foreground">ControlNet:</strong> zero-conv init → starts as no-op. Trainable encoder copy learns edge/depth/pose conditioning.</p>
                <p className="text-xs"><strong className="text-foreground">LoRA in U-Net:</strong> Low-rank adapters in Q,K,V projections of attention layers.</p>
                <p className="text-xs mt-1"><strong className="text-foreground">DreamBooth:</strong> subject LoRA on ~5 photos.</p>
                <p className="text-xs mt-1"><strong className="text-foreground">IP-Adapter:</strong> image prompt via separate cross-attention K,V.</p>
              </ContentCard>
            </div>
          </div>
        </section>

        {/* ═══ Part 6: Evaluation & Review ═══ */}
        <section id="review">
          <SectionHeader
            icon={BarChart3}
            title="Evaluation Metrics, Papers & Practice"
            number={6}
            subtitle="FID, Inception Score, CLIPScore — measuring generative quality. Plus consolidated algorithms and key papers."
          />
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
