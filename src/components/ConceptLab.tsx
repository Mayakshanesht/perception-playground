import { useState } from "react";
import { motion } from "framer-motion";
import { FlaskConical, ChevronDown, CheckCircle2, Play } from "lucide-react";
import { MathEquation } from "@/components/MathBlock";

export interface ConceptLabExercise {
  id: string;
  title: string;
  description: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  task: string;
  hints: string[];
  solution: string;
  equation?: { label: string; tex: string };
  variables?: { symbol: string; meaning: string }[];
}

export default function ConceptLab({ labs }: { labs: ConceptLabExercise[] }) {
  const [expandedLab, setExpandedLab] = useState<string | null>(null);
  const [showSolution, setShowSolution] = useState<Record<string, boolean>>({});
  const [showHints, setShowHints] = useState<Record<string, number>>({});

  const difficultyColor = {
    beginner: "168 80% 58%",
    intermediate: "38 92% 60%",
    advanced: "0 72% 51%",
  };

  return (
    <div className="space-y-3">
      {labs.map((lab) => {
        const isExpanded = expandedLab === lab.id;
        const hintsShown = showHints[lab.id] || 0;
        const solutionShown = showSolution[lab.id] || false;

        return (
          <div key={lab.id} className="rounded-xl border border-border bg-card/80 overflow-hidden">
            <button
              onClick={() => setExpandedLab(isExpanded ? null : lab.id)}
              className="w-full flex items-center gap-3 p-4 hover:bg-muted/30 transition-colors text-left"
            >
              <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                <FlaskConical className="h-4 w-4 text-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <h4 className="text-sm font-semibold text-foreground truncate">{lab.title}</h4>
                  <span
                    className="text-[9px] font-mono px-1.5 py-0.5 rounded-full uppercase tracking-wider"
                    style={{
                      backgroundColor: `hsl(${difficultyColor[lab.difficulty]} / 0.12)`,
                      color: `hsl(${difficultyColor[lab.difficulty]})`,
                    }}
                  >
                    {lab.difficulty}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground mt-0.5 truncate">{lab.description}</p>
              </div>
              <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform shrink-0 ${isExpanded ? "rotate-180" : ""}`} />
            </button>

            {isExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                transition={{ duration: 0.25 }}
                className="px-4 pb-4 space-y-4"
              >
                {/* Task */}
                <div className="rounded-lg bg-muted/40 border border-border p-4">
                  <h5 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-2 flex items-center gap-2">
                    <Play className="h-3 w-3 text-primary" /> Exercise
                  </h5>
                  <p className="text-sm text-foreground leading-relaxed">{lab.task}</p>
                </div>

                {/* Equation */}
                {lab.equation && (
                  <div>
                    <MathEquation tex={lab.equation.tex} label={lab.equation.label} />
                    {lab.variables && lab.variables.length > 0 && (
                      <div className="mt-2 rounded-lg bg-muted/30 border border-border p-3">
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5">Where</p>
                        <div className="space-y-1">
                          {lab.variables.map((v) => (
                            <p key={v.symbol} className="text-xs text-muted-foreground">
                              <span className="font-mono text-foreground">{v.symbol}</span> = {v.meaning}
                            </p>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Hints */}
                <div>
                  <button
                    onClick={() => setShowHints({ ...showHints, [lab.id]: Math.min(hintsShown + 1, lab.hints.length) })}
                    disabled={hintsShown >= lab.hints.length}
                    className="text-xs font-medium text-accent hover:text-accent/80 transition-colors disabled:opacity-40 disabled:cursor-default"
                  >
                    {hintsShown >= lab.hints.length ? "All hints shown" : `Show Hint (${hintsShown}/${lab.hints.length})`}
                  </button>
                  {hintsShown > 0 && (
                    <div className="mt-2 space-y-1.5">
                      {lab.hints.slice(0, hintsShown).map((hint, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, y: 4 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="text-xs text-muted-foreground bg-accent/5 border border-accent/10 rounded-lg px-3 py-2"
                        >
                          💡 {hint}
                        </motion.div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Solution */}
                <div>
                  <button
                    onClick={() => setShowSolution({ ...showSolution, [lab.id]: !solutionShown })}
                    className="text-xs font-medium text-primary hover:text-primary/80 transition-colors flex items-center gap-1"
                  >
                    <CheckCircle2 className="h-3 w-3" />
                    {solutionShown ? "Hide Solution" : "Show Solution"}
                  </button>
                  {solutionShown && (
                    <motion.div
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-2 rounded-lg bg-primary/5 border border-primary/20 p-4"
                    >
                      <p className="text-sm text-foreground leading-relaxed whitespace-pre-line">{lab.solution}</p>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            )}
          </div>
        );
      })}
    </div>
  );
}
