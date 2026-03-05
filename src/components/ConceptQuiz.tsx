import { useState } from "react";
import { motion } from "framer-motion";
import { CheckCircle2, XCircle, HelpCircle } from "lucide-react";

export interface QuizQuestion {
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
}

export default function ConceptQuiz({ questions }: { questions: QuizQuestion[] }) {
  const [currentQ, setCurrentQ] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);

  const q = questions[currentQ];
  if (!q) return null;

  const handleSelect = (idx: number) => {
    if (showResult) return;
    setSelected(idx);
    setShowResult(true);
    if (idx === q.correctIndex) setScore((s) => s + 1);
  };

  const next = () => {
    setSelected(null);
    setShowResult(false);
    setCurrentQ((c) => c + 1);
  };

  const isComplete = currentQ >= questions.length;

  if (isComplete) {
    return (
      <div className="rounded-xl border border-primary/30 bg-card p-6 text-center">
        <CheckCircle2 className="h-8 w-8 text-primary mx-auto mb-3" />
        <h3 className="text-sm font-semibold text-foreground mb-1">Quiz Complete!</h3>
        <p className="text-xs text-muted-foreground">
          You scored {score}/{questions.length}
        </p>
        <button
          onClick={() => { setCurrentQ(0); setScore(0); setSelected(null); setShowResult(false); }}
          className="mt-3 px-4 py-2 rounded-lg bg-primary/10 text-primary text-xs font-medium hover:bg-primary/20 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="flex items-center gap-2 mb-3">
        <HelpCircle className="h-4 w-4 text-accent" />
        <h3 className="text-sm font-semibold text-foreground">Quick Concept Check</h3>
        <span className="text-[10px] text-muted-foreground font-mono ml-auto">
          {currentQ + 1}/{questions.length}
        </span>
      </div>

      <p className="text-sm text-foreground mb-4 leading-relaxed">{q.question}</p>

      <div className="space-y-2">
        {q.options.map((opt, i) => {
          const isCorrect = i === q.correctIndex;
          const isSelected = i === selected;
          let borderColor = "border-border";
          let bgColor = "bg-muted/30";

          if (showResult) {
            if (isCorrect) { borderColor = "border-primary"; bgColor = "bg-primary/10"; }
            else if (isSelected && !isCorrect) { borderColor = "border-destructive"; bgColor = "bg-destructive/10"; }
          }

          return (
            <motion.button
              key={i}
              onClick={() => handleSelect(i)}
              disabled={showResult}
              className={`w-full text-left px-4 py-3 rounded-lg border ${borderColor} ${bgColor} text-sm text-foreground transition-colors hover:border-primary/40 disabled:cursor-default flex items-center gap-3`}
              whileTap={!showResult ? { scale: 0.98 } : {}}
            >
              <span className="h-6 w-6 rounded-md bg-muted flex items-center justify-center text-[10px] font-mono text-muted-foreground shrink-0">
                {String.fromCharCode(65 + i)}
              </span>
              <span className="flex-1">{opt}</span>
              {showResult && isCorrect && <CheckCircle2 className="h-4 w-4 text-primary shrink-0" />}
              {showResult && isSelected && !isCorrect && <XCircle className="h-4 w-4 text-destructive shrink-0" />}
            </motion.button>
          );
        })}
      </div>

      {showResult && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 rounded-lg bg-muted/40 border border-border p-3"
        >
          <p className="text-xs text-muted-foreground leading-relaxed">{q.explanation}</p>
          {currentQ < questions.length - 1 && (
            <button
              onClick={next}
              className="mt-3 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 transition-colors"
            >
              Next Question
            </button>
          )}
          {currentQ === questions.length - 1 && (
            <button
              onClick={next}
              className="mt-3 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 transition-colors"
            >
              See Results
            </button>
          )}
        </motion.div>
      )}
    </div>
  );
}
