import { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

interface MathBlockProps {
  tex: string;
  display?: boolean;
  className?: string;
}

export default function MathBlock({ tex, display = false, className = "" }: MathBlockProps) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (ref.current) {
      katex.render(tex, ref.current, {
        displayMode: display,
        throwOnError: false,
        trust: true,
      });
    }
  }, [tex, display]);

  return (
    <span
      ref={ref}
      className={`${display ? "block my-4 overflow-x-auto py-2" : "inline"} ${className}`}
    />
  );
}

export function MathEquation({ tex, label }: { tex: string; label?: string }) {
  return (
    <div className="my-4 rounded-lg bg-muted/50 border border-border p-4">
      {label && (
        <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-2 font-mono">{label}</p>
      )}
      <MathBlock tex={tex} display />
    </div>
  );
}
