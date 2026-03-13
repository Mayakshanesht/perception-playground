import ModulePage from "@/components/ModulePage";
import CameraAnimations from "@/components/CameraAnimations";
import { cameraModule } from "@/data/consolidatedModules";
import { ArrowLeft, GraduationCap } from "lucide-react";
import { Link } from "react-router-dom";

export default function CameraModule() {
  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      {/* Header */}
      <div className="flex items-start gap-4 mb-8">
        <div
          className="h-12 w-12 rounded-xl flex items-center justify-center shrink-0"
          style={{ backgroundColor: `hsl(${cameraModule.color} / 0.12)` }}
        >
          <GraduationCap className="h-6 w-6" style={{ color: `hsl(${cameraModule.color})` }} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground tracking-tight">{cameraModule.title}</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{cameraModule.subtitle}</p>
        </div>
      </div>

      {/* Interactive Animations — placed prominently before theory */}
      <div className="mb-8">
        <CameraAnimations />
      </div>

      {/* Rest of the module content */}
      <ModulePage content={cameraModule} hideHeader />
    </div>
  );
}
