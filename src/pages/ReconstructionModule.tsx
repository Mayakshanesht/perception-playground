import ModulePage from "@/components/ModulePage";
import { ModuleContent } from "@/data/moduleContent";
import { moduleContents } from "@/data/moduleContent";

// Merge SfM + NeRF into 3D reconstruction
const reconstructionModule: ModuleContent = {
  id: "reconstruction",
  title: "3D Reconstruction & Rendering",
  subtitle: "Reconstruct 3D scenes from images and render novel views — from classical Structure from Motion to neural radiance fields.",
  color: "340 75% 55%",
  theory: [
    ...moduleContents.sfm.theory,
    ...moduleContents.nerf.theory,
  ],
  algorithms: [
    ...moduleContents.sfm.algorithms,
    ...moduleContents.nerf.algorithms,
  ],
  papers: [
    ...moduleContents.sfm.papers,
    ...moduleContents.nerf.papers,
  ].sort((a, b) => a.year - b.year),
};

export default function ReconstructionModule() {
  return <ModulePage content={reconstructionModule} />;
}
