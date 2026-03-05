import { AlertTriangle } from "lucide-react";

export interface FailureMode {
  title: string;
  causes: string[];
}

export default function FailureModes({ failures }: { failures: FailureMode[] }) {
  return (
    <div className="space-y-3">
      {failures.map((f) => (
        <div key={f.title} className="rounded-lg border border-destructive/20 bg-destructive/5 p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="h-4 w-4 text-destructive" />
            <h4 className="text-sm font-semibold text-foreground">{f.title}</h4>
          </div>
          <ul className="space-y-1.5 ml-6">
            {f.causes.map((c, i) => (
              <li key={i} className="text-xs text-muted-foreground list-disc">{c}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
