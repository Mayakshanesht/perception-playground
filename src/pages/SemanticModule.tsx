import ModulePage from "@/components/ModulePage";
import { ModuleContent } from "@/data/moduleContent";
import { moduleContents } from "@/data/moduleContent";

// Merge classification + detection + segmentation into one module
const semanticModule: ModuleContent = {
  id: "semantic",
  title: "Semantic Information",
  subtitle: "Extract meaning from images — classify scenes, detect objects, and segment regions at the pixel level.",
  color: "187 85% 53%",
  theory: [
    ...moduleContents.classification.theory,
    ...moduleContents.detection.theory,
    ...moduleContents.segmentation.theory,
  ],
  algorithms: [
    ...moduleContents.classification.algorithms,
    ...moduleContents.detection.algorithms,
    ...moduleContents.segmentation.algorithms,
  ],
  papers: [
    ...moduleContents.classification.papers,
    ...moduleContents.detection.papers,
    ...moduleContents.segmentation.papers,
  ].sort((a, b) => a.year - b.year),
  playgrounds: [
    ...(moduleContents.detection.playground ? [moduleContents.detection.playground] : []),
    ...(moduleContents.segmentation.playgrounds ?? []),
  ],
};

export default function SemanticModule() {
  return <ModulePage content={semanticModule} />;
}
