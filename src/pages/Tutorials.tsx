import { motion } from "framer-motion";
import { ExternalLink, Clock, Target, BookOpen, ChevronRight } from "lucide-react";
import { tutorialsContent, TutorialCard } from "@/data/tutorialsContent";
import { Badge } from "@/components/ui/badge";
import { useState, useMemo } from "react";

const difficultyColor: Record<string, string> = {
  Beginner: "160 84% 39%",
  Intermediate: "32 95% 55%",
  Advanced: "0 72% 51%",
};

function TutorialCardComponent({ tutorial, index }: { tutorial: TutorialCard; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const accent = difficultyColor[tutorial.difficulty] ?? "192 94% 56%";

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06, duration: 0.35 }}
      className="rounded-xl border border-border bg-card overflow-hidden group hover:border-primary/40 transition-colors"
    >
      <div className="relative h-40 overflow-hidden bg-muted">
        <img
          src={tutorial.imageUrl}
          alt={tutorial.title}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
          loading="lazy"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-card/90 via-transparent to-transparent" />
        <div className="absolute top-3 left-3 flex gap-2">
          <span
            className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-md"
            style={{
              backgroundColor: `hsl(${accent} / 0.2)`,
              color: `hsl(${accent})`,
            }}
          >
            {tutorial.difficulty}
          </span>
        </div>
        <div className="absolute bottom-3 right-3 flex items-center gap-1 text-[10px] text-muted-foreground bg-card/80 backdrop-blur-sm px-2 py-1 rounded-md">
          <Clock className="h-3 w-3" />
          {tutorial.duration}
        </div>
      </div>

      <div className="p-5">
        <h3 className="font-semibold text-foreground text-sm mb-1.5">{tutorial.title}</h3>
        <p className="text-xs text-muted-foreground leading-relaxed mb-3">{tutorial.description}</p>

        <div className="flex flex-wrap gap-1.5 mb-4">
          {tutorial.tags.slice(0, 3).map((tag) => (
            <Badge key={tag} variant="secondary" className="text-[10px] font-mono">
              {tag}
            </Badge>
          ))}
          {tutorial.tags.length > 3 && (
            <Badge variant="secondary" className="text-[10px] font-mono">
              +{tutorial.tags.length - 3}
            </Badge>
          )}
        </div>

        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors mb-3"
        >
          <ChevronRight className={`h-3 w-3 transition-transform ${expanded ? "rotate-90" : ""}`} />
          Learning Objectives
        </button>
        {expanded && (
          <motion.ul
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="space-y-1.5 mb-4"
          >
            {tutorial.learningObjectives.map((obj, i) => (
              <li key={i} className="text-xs text-muted-foreground flex items-start gap-2">
                <Target className="h-3 w-3 mt-0.5 shrink-0 text-primary/60" />
                {obj}
              </li>
            ))}
          </motion.ul>
        )}

        <div className="flex flex-wrap gap-1.5 mb-4">
          {tutorial.topics.map((topic) => (
            <span
              key={topic}
              className="text-[10px] px-2 py-0.5 rounded-md bg-muted text-muted-foreground"
            >
              {topic}
            </span>
          ))}
        </div>

        <a
          href={tutorial.colabUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 text-xs font-medium text-primary hover:text-primary/80 transition-colors"
        >
          <BookOpen className="h-3.5 w-3.5" />
          Open in Google Colab
          <ExternalLink className="h-3 w-3" />
        </a>
      </div>
    </motion.div>
  );
}

export default function Tutorials() {
  const grouped = useMemo(() => {
    const general: TutorialCard[] = [];
    const categories: Record<string, TutorialCard[]> = {};

    for (const t of tutorialsContent) {
      if (t.category) {
        if (!categories[t.category]) categories[t.category] = [];
        categories[t.category].push(t);
      } else {
        general.push(t);
      }
    }

    return { general, categories };
  }, []);

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-foreground tracking-tight mb-2">Hands-On Tutorials</h1>
        <p className="text-sm text-muted-foreground max-w-2xl leading-relaxed">
          Guided Colab notebooks aligned with the perception pipeline — from detection to multimodal reasoning.
        </p>
      </div>

      {grouped.general.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 mb-10">
          {grouped.general.map((tutorial, i) => (
            <TutorialCardComponent key={tutorial.id} tutorial={tutorial} index={i} />
          ))}
        </div>
      )}

      {Object.entries(grouped.categories).map(([category, tutorials]) => (
        <div key={category} className="mb-10">
          <h2 className="text-lg font-semibold text-foreground tracking-tight mb-4 border-b border-border pb-2">
            {category}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {tutorials.map((tutorial, i) => (
              <TutorialCardComponent key={tutorial.id} tutorial={tutorial} index={i} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
