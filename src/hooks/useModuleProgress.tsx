import { useState, useEffect, useCallback } from "react";

const STORAGE_KEY = "perception-lab-progress";

export interface ModuleProgress {
  [moduleId: string]: {
    sectionsVisited: string[];
    totalSections: number;
    lastVisited?: string;
  };
}

function loadProgress(): ModuleProgress {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveProgress(progress: ModuleProgress) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
}

export function useModuleProgress(moduleId?: string, totalSections?: number) {
  const [progress, setProgress] = useState<ModuleProgress>(loadProgress);

  useEffect(() => {
    saveProgress(progress);
  }, [progress]);

  const markSectionVisited = useCallback(
    (sectionId: string) => {
      if (!moduleId) return;
      setProgress((prev) => {
        const mod = prev[moduleId] || { sectionsVisited: [], totalSections: totalSections || 1 };
        if (mod.sectionsVisited.includes(sectionId)) return prev;
        return {
          ...prev,
          [moduleId]: {
            ...mod,
            totalSections: totalSections || mod.totalSections,
            sectionsVisited: [...mod.sectionsVisited, sectionId],
            lastVisited: new Date().toISOString(),
          },
        };
      });
    },
    [moduleId, totalSections]
  );

  const getModulePercent = useCallback(
    (id: string) => {
      const mod = progress[id];
      if (!mod || mod.totalSections === 0) return 0;
      return Math.min(100, Math.round((mod.sectionsVisited.length / mod.totalSections) * 100));
    },
    [progress]
  );

  const currentPercent = moduleId ? getModulePercent(moduleId) : 0;

  return { progress, markSectionVisited, getModulePercent, currentPercent };
}
