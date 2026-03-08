import { Calendar, FolderOpen, Beaker, Box, FileText, Sparkles } from "lucide-react";

interface TimelineProps {
  projects: any[];
  experiments: any[];
  models: any[];
}

interface TimelineEvent {
  date: string;
  type: "project" | "experiment" | "model";
  title: string;
  subtitle?: string;
}

export default function ResearchTimeline({ projects, experiments, models }: TimelineProps) {
  const events: TimelineEvent[] = [
    ...projects.map(p => ({ date: p.created_at, type: "project" as const, title: p.name, subtitle: p.research_question || "New project" })),
    ...experiments.map(e => ({ date: e.created_at, type: "experiment" as const, title: e.title, subtitle: `Status: ${e.status?.replace(/_/g, " ")}` })),
    ...models.map(m => ({ date: m.created_at, type: "model" as const, title: m.name, subtitle: m.dataset_used ? `Dataset: ${m.dataset_used}` : "Registered model" })),
  ].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  const iconMap = { project: FolderOpen, experiment: Beaker, model: Box };
  const colorMap = {
    project: "bg-primary/15 text-primary border-primary/30",
    experiment: "bg-yellow-500/15 text-yellow-600 dark:text-yellow-400 border-yellow-500/30",
    model: "bg-green-500/15 text-green-600 dark:text-green-400 border-green-500/30",
  };

  if (events.length === 0) {
    return (
      <div className="text-center py-16">
        <Calendar className="h-10 w-10 text-muted-foreground/30 mx-auto mb-4" />
        <h3 className="text-sm font-semibold text-foreground mb-1">No Activity Yet</h3>
        <p className="text-xs text-muted-foreground">Create projects and run experiments to see your research timeline.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
        <Calendar className="h-4 w-4 text-primary" /> Research Timeline
      </h3>
      <div className="relative">
        <div className="absolute left-[19px] top-0 bottom-0 w-px bg-border" />
        <div className="space-y-4">
          {events.map((ev, i) => {
            const Icon = iconMap[ev.type];
            return (
              <div key={i} className="flex gap-4 items-start relative">
                <div className={`shrink-0 h-10 w-10 rounded-full flex items-center justify-center border z-10 ${colorMap[ev.type]}`}>
                  <Icon className="h-4 w-4" />
                </div>
                <div className="flex-1 rounded-xl border border-border bg-card p-3 min-w-0">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-semibold text-foreground line-clamp-1">{ev.title}</h4>
                    <span className="text-[10px] text-muted-foreground font-mono shrink-0 ml-2">{new Date(ev.date).toLocaleDateString()}</span>
                  </div>
                  {ev.subtitle && <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">{ev.subtitle}</p>}
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground/60 font-semibold mt-1 block">{ev.type}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
