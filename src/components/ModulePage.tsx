import { motion } from "framer-motion";
import { ArrowLeft, BookOpen, FlaskConical, FileText, Cpu, GraduationCap } from "lucide-react";
import { Link } from "react-router-dom";
import { ModuleContent } from "@/data/moduleContent";
import { MathEquation } from "@/components/MathBlock";
import Playground from "@/components/Playground";

interface ModulePageProps {
  content: ModuleContent;
}

export default function ModulePage({ content }: ModulePageProps) {
  const playgrounds = content.playgrounds ?? (content.playground ? [content.playground] : []);
  const flowItems = [
    { id: "theory", label: "Theory", detail: `${content.theory.length} concepts` },
    { id: "algorithms", label: "Algorithms", detail: `${content.algorithms.length} pipeline(s)` },
    { id: "papers", label: "Papers", detail: `${content.papers.length} references` },
    { id: "playground", label: "Playground", detail: `${playgrounds.length} lab(s)` },
  ];

  return (
    <div className="p-8 max-w-5xl mx-auto">
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

      <div className="space-y-10">
        <section className="rounded-xl border border-border bg-muted/40 p-4">
          <h2 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3">Structured Learning Flow</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-2">
            {flowItems.map((item, idx) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                className="rounded-lg border border-border bg-card p-3 hover:border-primary/40 transition-colors"
              >
                <p className="text-[10px] text-muted-foreground font-mono mb-1">Step {idx + 1}</p>
                <p className="text-sm text-foreground font-medium">{item.label}</p>
                <p className="text-xs text-muted-foreground">{item.detail}</p>
              </a>
            ))}
          </div>
        </section>

        {/* Theory Section */}
        <section id="theory">
          <SectionHeader icon={BookOpen} title="Theory & Foundations" />
          <div className="space-y-6">
            {content.theory.map((section, i) => (
              <motion.div
                key={section.title}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="rounded-xl border border-border bg-card p-6"
              >
                <h3 className="font-semibold text-foreground mb-3 flex items-center gap-2">
                  <span
                    className="h-5 w-5 rounded-md flex items-center justify-center text-[10px] font-mono font-bold"
                    style={{
                      backgroundColor: `hsl(${content.color} / 0.15)`,
                      color: `hsl(${content.color})`,
                    }}
                  >
                    {i + 1}
                  </span>
                  {section.title}
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed mb-4">{section.content}</p>
                {section.equations?.map((eq) => (
                  <MathEquation key={eq.label} tex={eq.tex} label={eq.label} />
                ))}
              </motion.div>
            ))}
          </div>
        </section>

        {/* Algorithms Section */}
        {content.algorithms.length > 0 && (
          <section id="algorithms">
            <SectionHeader icon={Cpu} title="Algorithms & Pipelines" />
            <div className="space-y-6">
              {content.algorithms.map((algo) => (
                <div key={algo.name} className="rounded-xl border border-border bg-card p-6">
                  <h3 className="font-semibold text-foreground text-sm mb-4">{algo.name}</h3>
                  <div className="space-y-3">
                    {algo.steps.map((step, i) => (
                      <motion.div
                        key={step.step}
                        initial={{ opacity: 0, x: -12 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.04 }}
                        className="flex items-start gap-4 rounded-lg bg-muted/40 p-3"
                      >
                        <div
                          className="h-7 w-7 rounded-md flex items-center justify-center text-[10px] font-mono font-bold shrink-0"
                          style={{
                            backgroundColor: `hsl(${content.color} / 0.12)`,
                            color: `hsl(${content.color})`,
                          }}
                        >
                          {String(i + 1).padStart(2, "0")}
                        </div>
                        <div>
                          <p className="text-sm font-medium text-foreground">{step.step}</p>
                          <p className="text-xs text-muted-foreground mt-0.5">{step.detail}</p>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Key Papers */}
        <section id="papers">
          <SectionHeader icon={FileText} title="Key Papers" />
          <div className="relative">
            <div className="absolute left-[72px] top-0 bottom-0 w-px bg-border" />
            <div className="space-y-3">
              {content.papers.map((paper, i) => (
                <motion.div
                  key={paper.title}
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.04 }}
                  className="flex gap-4 group"
                >
                  <div className="w-[60px] shrink-0 text-right">
                    <span className="text-xs font-mono text-muted-foreground">{paper.year}</span>
                  </div>
                  <div className="relative shrink-0 mt-1.5">
                    <div
                      className="h-3 w-3 rounded-full border-2 border-border bg-background group-hover:border-primary transition-colors"
                      style={{ borderColor: undefined }}
                    />
                  </div>
                  <div className="rounded-lg border border-border bg-card p-4 flex-1 group-hover:border-primary/30 transition-colors">
                    <div className="flex items-baseline gap-2 mb-1">
                      <h4 className="text-sm font-semibold text-foreground">{paper.title}</h4>
                      <span className="text-[10px] font-mono text-muted-foreground">{paper.venue}</span>
                    </div>
                    <p className="text-[10px] text-primary/70 mb-1">{paper.authors}</p>
                    <p className="text-xs text-muted-foreground leading-relaxed">{paper.summary}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Playground */}
        {playgrounds.length > 0 && (
          <section id="playground">
            <SectionHeader icon={FlaskConical} title="Inference Playground" />
            <div className="mb-4 rounded-xl border border-border bg-muted/40 p-4">
              <p className="text-xs font-semibold text-foreground mb-1">Student Workflow</p>
              <p className="text-xs text-muted-foreground">
                Run each model with at least two contrasting inputs, compare confidence outputs, and note one failure mode.
              </p>
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
      </div>
    </div>
  );
}

function SectionHeader({ icon: Icon, title }: { icon: any; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <Icon className="h-4 w-4 text-primary" />
      <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">{title}</h2>
    </div>
  );
}
