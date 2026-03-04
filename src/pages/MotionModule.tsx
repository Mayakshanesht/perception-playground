import ModulePage from "@/components/ModulePage";
import { ModuleContent, PlaygroundConfig } from "@/data/moduleContent";
import { moduleContents } from "@/data/moduleContent";

const velocityPlayground: PlaygroundConfig = {
  title: "Speed Estimation Playground",
  description: "Upload a video to estimate object speeds using YOLO tracking with velocity estimation.",
  taskType: "velocity-estimation",
  acceptVideo: true,
  acceptImage: false,
  modelName: "yolo26n.pt (tracking + speed estimation)",
  learningFocus: "Compare estimated speeds across different objects and analyze how tracking quality affects velocity accuracy.",
};

// Merge tracking + action + optical flow into motion
const motionModule: ModuleContent = {
  id: "motion",
  title: "Motion Estimation",
  subtitle: "Understand visual motion — track objects across frames, recognize actions, and estimate optical flow and velocity.",
  color: "280 70% 55%",
  theory: [
    ...moduleContents.tracking.theory,
    ...moduleContents.opticalflow.theory,
    ...moduleContents.action.theory,
  ],
  algorithms: [
    ...moduleContents.tracking.algorithms,
    ...moduleContents.opticalflow.algorithms,
    ...moduleContents.action.algorithms,
  ],
  papers: [
    ...moduleContents.tracking.papers,
    ...moduleContents.opticalflow.papers,
    ...moduleContents.action.papers,
  ].sort((a, b) => a.year - b.year),
  playgrounds: [velocityPlayground],
};

export default function MotionModule() {
  return <ModulePage content={motionModule} />;
}
