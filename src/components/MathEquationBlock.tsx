import { MathEquation } from "@/components/MathBlock";

export function MathEquationBlock({ tex, label }: { tex: string; label?: string }) {
  return (
    <div className="equation-block">
      {label && (
        <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-2 font-mono">{label}</p>
      )}
      <MathEquation tex={tex} label={undefined} />
    </div>
  );
}
