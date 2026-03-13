import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft, BookOpen, FlaskConical, FileText, Cpu, GraduationCap, Lightbulb, Calculator, History, Brain, Rocket, ChevronDown, AlertTriangle, HelpCircle, Zap, Image, Beaker, Link2 } from "lucide-react";
import { Link } from "react-router-dom";
import { ModuleContent } from "@/data/moduleContent";
import { MathEquation } from "@/components/MathBlock";
import Playground from "@/components/Playground";
import ConceptQuiz from "@/components/ConceptQuiz";
import FailureModesComponent from "@/components/FailureModes";
import PaperAgent from "@/components/PaperAgent";
import ConceptLab from "@/components/ConceptLab";
import ModuleImage from "@/components/ModuleImage";
import { moduleQuizzes, moduleFailureModes } from "@/data/moduleQuizData";
import { moduleLabs } from "@/data/conceptLabs";
import { moduleImages } from "@/data/moduleImages";
import { useState } from "react";

interface ModulePageProps {
  content: ModuleContent;
  hideHeader?: boolean;
  hideTheory?: boolean;
}

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
      t.includes("lens") || t.includes("image formation") || t.includes("sensor") ||
      t.includes("two-stage") || t.includes("one-stage") || t.includes("fcn") ||
      t.includes("feature pyramid") || t.includes("atrous") || t.includes("monocular") ||
      t.includes("2d pose") || t.includes("3d pose") || t.includes("tracking by")
    ) {
      classical.push(section);
    } else if (section.equations && section.equations.length > 0) {
      math.push(section);
    } else {
      other.push(section);
    }
  }

  for (const s of other) {
    if (s.equations && s.equations.length > 0) math.push(s);
    else intuition.push(s);
  }

  return { intuition, math, classical, deepLearning, applications };
}

function CollapsibleSection({ title, icon: Icon, color, children, defaultOpen = true, id }: {
  title: string;
  icon: any;
  color: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  id?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section id={id} className="rounded-xl border border-border bg-card/50 overflow-hidden">
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
    <div className="concept-card">
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

// Module connection map
const moduleConnections: Record<string, { label: string; path: string; relation: string }[]> = {
  camera: [
    { label: "Semantic Information", path: "/module/semantic", relation: "Camera images are input to all semantic tasks" },
    { label: "Geometric Information", path: "/module/geometric", relation: "Calibration is essential for depth and stereo" },
    { label: "3D Reconstruction", path: "/module/reconstruction", relation: "SfM uses camera intrinsics for triangulation" },
  ],
  semantic: [
    { label: "Camera", path: "/module/camera", relation: "Requires calibrated camera images as input" },
    { label: "Motion Estimation", path: "/module/motion", relation: "Detection feeds into multi-object tracking" },
    { label: "Scene Reasoning", path: "/module/scene-reasoning", relation: "Visual features enable multimodal understanding" },
  ],
  geometric: [
    { label: "Camera", path: "/module/camera", relation: "Stereo depth requires camera calibration" },
    { label: "3D Reconstruction", path: "/module/reconstruction", relation: "Depth maps feed into dense reconstruction" },
    { label: "Motion Estimation", path: "/module/motion", relation: "Pose estimation connects to action recognition" },
  ],
  motion: [
    { label: "Semantic Information", path: "/module/semantic", relation: "Detection provides input for tracking" },
    { label: "Camera", path: "/module/camera", relation: "Ego-motion requires camera calibration" },
    { label: "3D Reconstruction", path: "/module/reconstruction", relation: "Multi-view motion helps SfM" },
  ],
  reconstruction: [
    { label: "Camera", path: "/module/camera", relation: "Requires calibrated cameras with known intrinsics" },
    { label: "Geometric Information", path: "/module/geometric", relation: "Depth estimation provides dense geometry" },
    { label: "Scene Reasoning", path: "/module/scene-reasoning", relation: "3D scenes enable spatial reasoning" },
  ],
  "scene-reasoning": [
    { label: "Semantic Information", path: "/module/semantic", relation: "Visual features from detection/segmentation" },
    { label: "Geometric Information", path: "/module/geometric", relation: "Depth and pose provide spatial context" },
    { label: "3D Reconstruction", path: "/module/reconstruction", relation: "3D scene understanding for embodied AI" },
  ],
};

export default function ModulePage({ content, hideHeader, hideTheory }: ModulePageProps) {
  const playgrounds = content.playgrounds ?? (content.playground ? [content.playground] : []);
  const { intuition, math, classical, deepLearning, applications } = categorizeSections(content.theory);
  const quizQuestions = content.quizQuestions || moduleQuizzes[content.id] || [];
  const failureModes = content.failureModes || moduleFailureModes[content.id] || [];
  const labs = moduleLabs[content.id] || [];
  const images = moduleImages[content.id] || [];
  const connections = moduleConnections[content.id] || [];
  const [exploringPaper, setExploringPaper] = useState<string | null>(null);

  const navItems = [
    { id: "intuition", label: "Intuition", count: intuition.length, icon: "💡" },
    { id: "images", label: "Visual", count: images.length, icon: "🖼️" },
    { id: "math", label: "Math", count: math.length, icon: "📐" },
    { id: "classical", label: "Classical", count: classical.length, icon: "📚" },
    { id: "deep-learning", label: "Deep Learning", count: deepLearning.length, icon: "🧠" },
    { id: "labs", label: "Labs", count: labs.length, icon: "🧪" },
    { id: "playground", label: "Playground", count: playgrounds.length, icon: "🎮" },
    { id: "quiz", label: "Quiz", count: quizQuestions.length, icon: "❓" },
    { id: "failures", label: "Failures", count: failureModes.length, icon: "⚠️" },
    { id: "applications", label: "Applications", count: applications.length, icon: "🚀" },
    { id: "connections", label: "Connections", count: connections.length, icon: "🔗" },
  ].filter(s => s.count > 0);

  return (
    <div className={hideHeader ? "" : "p-6 md:p-8 max-w-5xl mx-auto"}>
      {!hideHeader && (
        <>
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
        </>
      )}

      {/* Learning flow nav */}
      <div className="rounded-xl border border-border bg-muted/30 p-4 mb-8">
        <h2 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3">Structured Learning Flow</h2>
        <div className="grid sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6 gap-2">
          {navItems.map((item) => (
            <a
              key={item.id}
              href={`#${item.id}`}
              className="rounded-lg border border-border bg-card p-2.5 hover:border-primary/40 transition-colors text-center"
            >
              <p className="text-sm mb-0.5">{item.icon}</p>
              <p className="text-xs text-foreground font-medium">{item.label}</p>
              <p className="text-[10px] text-muted-foreground">{item.count}</p>
            </a>
          ))}
        </div>
      </div>

      <div className="space-y-6">
        {/* Intuition */}
        {intuition.length > 0 && (
          <CollapsibleSection title="Concept Overview & Intuition" icon={Lightbulb} color={content.color} id="intuition">
            {intuition.map((s, i) => (
              <TheoryCard key={s.title} section={s} color={content.color} index={i} />
            ))}
          </CollapsibleSection>
        )}

        {/* Visual Diagrams */}
        {images.length > 0 && (
          <CollapsibleSection title="Visual Diagrams" icon={Image} color={content.color} id="images">
            {images.map((img) => (
              <ModuleImage key={img.src} src={img.src} alt={img.alt} caption={img.caption} />
            ))}
          </CollapsibleSection>
        )}

        {/* Mathematical Formulation */}
        {math.length > 0 && (
          <CollapsibleSection title="Mathematical Formulation" icon={Calculator} color={content.color} id="math">
            {math.map((s, i) => (
              <TheoryCard key={s.title} section={s} color={content.color} index={i} />
            ))}
          </CollapsibleSection>
        )}

        {/* Classical Approaches */}
        {classical.length > 0 && (
          <CollapsibleSection title="Classical Methods" icon={History} color={content.color} id="classical">
            {classical.map((s, i) => (
              <TheoryCard key={s.title} section={s} color={content.color} index={i} />
            ))}
          </CollapsibleSection>
        )}

        {/* Deep Learning Approaches */}
        {deepLearning.length > 0 && (
          <CollapsibleSection title="Modern Deep Learning Methods" icon={Brain} color={content.color} id="deep-learning">
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

        {/* Conceptual Labs */}
        {labs.length > 0 && (
          <CollapsibleSection title="Conceptual Labs" icon={Beaker} color="168 80% 58%" id="labs" defaultOpen={false}>
            <p className="text-xs text-muted-foreground mb-3">
              Hands-on exercises to deepen understanding through reasoning and computation. No GPU required.
            </p>
            <ConceptLab labs={labs} />
          </CollapsibleSection>
        )}

        {/* Playground */}
        {playgrounds.length > 0 && (
          <section id="playground">
            <div className="flex items-center gap-2 mb-2">
              <FlaskConical className="h-4 w-4 text-primary" />
              <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">Interactive Playground</h2>
            </div>
            <div className="rounded-lg border border-accent/20 bg-accent/5 p-3 mb-4">
              <div className="flex items-center gap-2">
                <Zap className="h-3.5 w-3.5 text-accent" />
                <p className="text-[11px] text-muted-foreground">
                  Playgrounds run real CV models on GPU infrastructure. Learning modules are free; playground runs may require GPU credits.
                </p>
              </div>
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
          <CollapsibleSection title="Key Research Papers" icon={FileText} color={content.color} defaultOpen={false}>
            <div className="relative">
              <div className="absolute left-[72px] top-0 bottom-0 w-px bg-border" />
              <div className="space-y-2">
                {content.papers.map((paper) => (
                  <div key={paper.title}>
                    <div className="flex gap-4 group">
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
                        <p className="text-xs text-muted-foreground leading-relaxed mb-2">{paper.summary}</p>
                        <button
                          onClick={() => setExploringPaper(exploringPaper === paper.title ? null : paper.title)}
                          className="text-[10px] font-medium text-primary hover:text-primary/80 transition-colors flex items-center gap-1"
                        >
                          <BookOpen className="h-3 w-3" />
                          {exploringPaper === paper.title ? "Close" : "Explore Paper"}
                        </button>
                      </div>
                    </div>
                    <AnimatePresence>
                      {exploringPaper === paper.title && (
                        <div className="ml-[88px]">
                          <PaperAgent paperTitle={paper.title} onClose={() => setExploringPaper(null)} />
                        </div>
                      )}
                    </AnimatePresence>
                  </div>
                ))}
              </div>
            </div>
          </CollapsibleSection>
        )}

        {/* Failure Modes */}
        {failureModes.length > 0 && (
          <CollapsibleSection title="Failure Modes" icon={AlertTriangle} color="0 72% 51%" defaultOpen={false} id="failures">
            <FailureModesComponent failures={failureModes} />
          </CollapsibleSection>
        )}

        {/* Quiz */}
        {quizQuestions.length > 0 && (
          <section id="quiz">
            <ConceptQuiz questions={quizQuestions} />
          </section>
        )}

        {/* Applications */}
        {applications.length > 0 && (
          <CollapsibleSection title="Real-World Applications" icon={Rocket} color={content.color} id="applications">
            {applications.map((s, i) => (
              <TheoryCard key={s.title} section={s} color={content.color} index={i} />
            ))}
          </CollapsibleSection>
        )}

        {/* Connections to Other Modules */}
        {connections.length > 0 && (
          <CollapsibleSection title="Connection to Other Modules" icon={Link2} color={content.color} id="connections" defaultOpen={false}>
            <div className="space-y-2">
              {connections.map((conn) => (
                <Link
                  key={conn.label}
                  to={conn.path}
                  className="flex items-center gap-3 rounded-lg border border-border bg-muted/30 p-3 hover:border-primary/40 transition-colors group"
                >
                  <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                    <Link2 className="h-3.5 w-3.5 text-primary" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-foreground group-hover:text-primary transition-colors">{conn.label}</p>
                    <p className="text-xs text-muted-foreground">{conn.relation}</p>
                  </div>
                </Link>
              ))}
            </div>
          </CollapsibleSection>
        )}
      </div>
    </div>
  );
}
