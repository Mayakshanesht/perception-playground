import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft, BookOpen, FlaskConical, FileText, Cpu, GraduationCap, Lightbulb, Calculator, History, Brain, Rocket, ChevronDown } from "lucide-react";
import { Link } from "react-router-dom";
import { ModuleContent } from "@/data/moduleContent";
import { MathEquation } from "@/components/MathBlock";
import Playground from "@/components/Playground";
import { useState } from "react";

interface ModulePageProps {
  content: ModuleContent;
}

// Map theory sections by title keywords to the 6-section structure
function categorizeSections(theory: ModuleContent["theory"]) {
  const intuition: typeof theory = [];
  const math: typeof theory = [];
  const classical: typeof theory = [];
  const deepLearning: typeof theory = [];
  const applications: typeof theory = [];
  const other: typeof theory = [];

  for (const section of theory) {
    const t = section.title.toLowerCase();
    if (t.includes("intuition") || t.includes("what is") || t.includes("types of") || t.includes("challenge") || t.includes("temporal modeling")) {
      intuition.push(section);
    } else if (t.includes("application") || t.includes("real-world")) {
      applications.push(section);
    } else if (
      t.includes("transformer") || t.includes("vit") || t.includes("deep") ||
      t.includes("nerf") || t.includes("gaussian") || t.includes("raft") ||
      t.includes("flownet") || t.includes("hrnet") || t.includes("mask r-cnn") ||
      t.includes("efficientnet") || t.includes("detr") || t.includes("dpt") ||
      t.includes("midas") || t.includes("self-supervised") ||
      t.includes("part affinity") || t.includes("deeplab") || t.includes("u-net") ||
      t.includes("3d convolution") || t.includes("two-stream") ||
      t.includes("video transformer") || t.includes("lvlm") || t.includes("florence") ||
      t.includes("vision-language") || t.includes("visual grounding") ||
      t.includes("residual") || t.includes("batch norm") || t.includes("positional encoding") ||
      t.includes("appearance feature")
    ) {
      deepLearning.push(section);
    } else if (
      t.includes("lucas") || t.includes("horn") || t.includes("kalman") ||
      t.includes("hungarian") || t.includes("nms") || t.includes("brightness") ||
      t.includes("aperture") || t.includes("sift") || t.includes("feature detection") ||
      t.includes("epipolar") || t.includes("triangulation") || t.includes("bundle") ||
      t.includes("stereo") || t.includes("calibration") || t.includes("pinhole") ||
      t.includes("lens") || t.includes("image formation") || t.includes("sensor")
    ) {
      classical.push(section);
    } else if (section.equations && section.equations.length > 0) {
      math.push(section);
    } else {
      other.push(section);
    }
  }

  // Distribute "other" into intuition if no equations, math otherwise
  for (const s of other) {
    if (s.equations && s.equations.length > 0) math.push(s);
    else intuition.push(s);
  }

  return { intuition, math, classical, deepLearning, applications };
}

function CollapsibleSection({ title, icon: Icon, color, children, defaultOpen = true }: {
  title: string;
  icon: any;
  color: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="rounded-xl border border-border bg-card/50 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 p-4 hover:bg-muted/30 transition-colors text-left"
      >
        <div
          className="h-8 w-8 rounded-lg flex items-center justify-center shrink-0"
          style={{ backgroundColor: `hsl(${color} / 0.12)` }}
        >
          <Icon className="h-4 w-4" style={{ color: `hsl(${color})` }} />
        </div>
        <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider flex-1">{title}</h2>
        <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 space-y-4">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}

function TheoryCard({ section, color, index }: { section: ModuleContent["theory"][0]; color: string; index: number }) {
  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <h3 className="font-semibold text-foreground mb-3 flex items-center gap-2 text-sm">
        <span
          className="h-5 w-5 rounded-md flex items-center justify-center text-[10px] font-mono font-bold shrink-0"
          style={{ backgroundColor: `hsl(${color} / 0.15)`, color: `hsl(${color})` }}
        >
          {index + 1}
        </span>
        {section.title}
      </h3>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">{section.content}</p>
      {section.equations?.map((eq) => (
        <MathEquation key={eq.label} tex={eq.tex} label={eq.label} />
      ))}
    </div>
  );
}

export default function ModulePage({ content }: ModulePageProps) {
  const playgrounds = content.playgrounds ?? (content.playground ? [content.playground] : []);
  const { intuition, math, classical, deepLearning, applications } = categorizeSections(content.theory);

  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      {/* Header */}
      <div className="flex items-start gap-4 mb-8">
        <div
          className="h-12 w-12 rounded-xl flex items-center justify-center shrink-0"
          style={{ backgroundColor: `hsl(${content.color} / 0.12)` }}
        >
          <GraduationCap className="h-6 w-6" style={{ color: `hsl(${content.color})` }} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground tracking-tight">{content.title}</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{content.subtitle}</p>
        </div>
      </div>

      {/* Learning flow nav */}
      <div className="rounded-xl border border-border bg-muted/40 p-4 mb-8">
        <h2 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3">Structured Learning Flow</h2>
        <div className="grid sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {[
            { id: "intuition", label: "Intuition", count: intuition.length },
            { id: "math", label: "Math", count: math.length },
            { id: "classical", label: "Classical", count: classical.length },
            { id: "deep-learning", label: "Deep Learning", count: deepLearning.length },
            { id: "playground", label: "Playground", count: playgrounds.length },
            { id: "applications", label: "Applications", count: applications.length },
          ].filter(s => s.count > 0).map((item, idx) => (
            <a
              key={item.id}
              href={`#${item.id}`}
              className="rounded-lg border border-border bg-card p-2.5 hover:border-primary/40 transition-colors text-center"
            >
              <p className="text-[10px] text-muted-foreground font-mono mb-0.5">Step {idx + 1}</p>
              <p className="text-xs text-foreground font-medium">{item.label}</p>
              <p className="text-[10px] text-muted-foreground">{item.count} section(s)</p>
            </a>
          ))}
        </div>
      </div>

      <div className="space-y-6">
        {/* Intuition */}
        {intuition.length > 0 && (
          <CollapsibleSection title="Intuition" icon={Lightbulb} color={content.color}>
            {intuition.map((s, i) => (
              <TheoryCard key={s.title} section={s} color={content.color} index={i} />
            ))}
          </CollapsibleSection>
        )}

        {/* Mathematical Formulation */}
        {math.length > 0 && (
          <CollapsibleSection title="Mathematical Formulation" icon={Calculator} color={content.color}>
            {math.map((s, i) => (
              <TheoryCard key={s.title} section={s} color={content.color} index={i} />
            ))}
          </CollapsibleSection>
        )}

        {/* Classical Approaches */}
        {classical.length > 0 && (
          <CollapsibleSection title="Classical Approaches" icon={History} color={content.color}>
            {classical.map((s, i) => (
              <TheoryCard key={s.title} section={s} color={content.color} index={i} />
            ))}
          </CollapsibleSection>
        )}

        {/* Deep Learning Approaches */}
        {deepLearning.length > 0 && (
          <CollapsibleSection title="Deep Learning Approaches" icon={Brain} color={content.color}>
            {deepLearning.map((s, i) => (
              <TheoryCard key={s.title} section={s} color={content.color} index={i} />
            ))}
          </CollapsibleSection>
        )}

        {/* Algorithms */}
        {content.algorithms.length > 0 && (
          <CollapsibleSection title="Algorithms & Pipelines" icon={Cpu} color={content.color} defaultOpen={false}>
            {content.algorithms.map((algo) => (
              <div key={algo.name} className="rounded-lg border border-border bg-card p-5">
                <h3 className="font-semibold text-foreground text-sm mb-4">{algo.name}</h3>
                <div className="space-y-2">
                  {algo.steps.map((step, i) => (
                    <div key={step.step} className="flex items-start gap-3 rounded-lg bg-muted/40 p-3">
                      <div
                        className="h-6 w-6 rounded-md flex items-center justify-center text-[10px] font-mono font-bold shrink-0"
                        style={{ backgroundColor: `hsl(${content.color} / 0.12)`, color: `hsl(${content.color})` }}
                      >
                        {String(i + 1).padStart(2, "0")}
                      </div>
                      <div>
                        <p className="text-sm font-medium text-foreground">{step.step}</p>
                        <p className="text-xs text-muted-foreground mt-0.5">{step.detail}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </CollapsibleSection>
        )}

        {/* Playground */}
        {playgrounds.length > 0 && (
          <section id="playground">
            <div className="flex items-center gap-2 mb-4">
              <FlaskConical className="h-4 w-4 text-primary" />
              <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">Interactive Playground</h2>
            </div>
            <div className="space-y-4">
              {playgrounds.map((pg) => (
                <Playground
                  key={`${content.id}-${pg.taskType}-${pg.title}`}
                  title={pg.title}
                  description={pg.description}
                  taskType={pg.taskType}
                  acceptVideo={pg.acceptVideo}
                  acceptImage={pg.acceptImage}
                  modelName={pg.modelName}
                  learningFocus={pg.learningFocus}
                />
              ))}
            </div>
          </section>
        )}

        {/* Key Papers */}
        {content.papers.length > 0 && (
          <CollapsibleSection title="Key Papers" icon={FileText} color={content.color} defaultOpen={false}>
            <div className="relative">
              <div className="absolute left-[72px] top-0 bottom-0 w-px bg-border" />
              <div className="space-y-2">
                {content.papers.map((paper, i) => (
                  <div key={paper.title} className="flex gap-4 group">
                    <div className="w-[60px] shrink-0 text-right">
                      <span className="text-xs font-mono text-muted-foreground">{paper.year}</span>
                    </div>
                    <div className="relative shrink-0 mt-1.5">
                      <div className="h-3 w-3 rounded-full border-2 border-border bg-background group-hover:border-primary transition-colors" />
                    </div>
                    <div className="rounded-lg border border-border bg-card p-3 flex-1 group-hover:border-primary/30 transition-colors">
                      <div className="flex items-baseline gap-2 mb-1">
                        <h4 className="text-sm font-semibold text-foreground">{paper.title}</h4>
                        <span className="text-[10px] font-mono text-muted-foreground">{paper.venue}</span>
                      </div>
                      <p className="text-[10px] text-primary/70 mb-1">{paper.authors}</p>
                      <p className="text-xs text-muted-foreground leading-relaxed">{paper.summary}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CollapsibleSection>
        )}

        {/* Applications */}
        {applications.length > 0 && (
          <CollapsibleSection title="Real-World Applications" icon={Rocket} color={content.color}>
            {applications.map((s, i) => (
              <TheoryCard key={s.title} section={s} color={content.color} index={i} />
            ))}
          </CollapsibleSection>
        )}
      </div>
    </div>
  );
}
