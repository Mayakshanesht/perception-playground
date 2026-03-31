import { useEffect, useRef } from "react";
import { useModuleProgress } from "./useModuleProgress";

/**
 * Drop this hook into any module page. Pass moduleId and sectionIds.
 * It will observe each section and mark it visited when 40%+ visible.
 */
export function useSectionObserver(moduleId: string, sectionIds: string[]) {
  const { markSectionVisited, currentPercent } = useModuleProgress(moduleId, sectionIds.length);
  const observerRef = useRef<IntersectionObserver | null>(null);

  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            markSectionVisited(entry.target.id);
          }
        });
      },
      { threshold: 0.4 }
    );

    // Small delay to ensure DOM is ready
    const timer = setTimeout(() => {
      sectionIds.forEach((id) => {
        const el = document.getElementById(id);
        if (el && observerRef.current) {
          observerRef.current.observe(el);
        }
      });
    }, 500);

    return () => {
      clearTimeout(timer);
      observerRef.current?.disconnect();
    };
  }, [moduleId, sectionIds, markSectionVisited]);

  return currentPercent;
}
