import ModulePage from "@/components/ModulePage";
import { ModuleContent } from "@/data/moduleContent";
import { moduleContents } from "@/data/moduleContent";

// Merge depth + pose into geometric information
const geometricModule: ModuleContent = {
  id: "geometric",
  title: "Geometric Information",
  subtitle: "Recover 3D geometry from images — estimate depth, detect human poses, and understand spatial structure.",
  color: "32 95% 55%",
  theory: [
    ...moduleContents.depth.theory,
    ...moduleContents.pose.theory,
  ],
  algorithms: [
    ...moduleContents.depth.algorithms,
    ...moduleContents.pose.algorithms,
  ],
  papers: [
    ...moduleContents.depth.papers,
    ...moduleContents.pose.papers,
  ].sort((a, b) => a.year - b.year),
  playgrounds: [
    ...(moduleContents.depth.playground ? [moduleContents.depth.playground] : []),
    ...(moduleContents.pose.playground ? [moduleContents.pose.playground] : []),
  ],
};

export default function GeometricModule() {
  return <ModulePage content={geometricModule} />;
}
