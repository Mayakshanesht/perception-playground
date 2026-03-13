import { Projection3DScene, PinholeScene, PerspectiveScene, IntrinsicScene, LensScene, SensorScene } from "@/components/CameraAnimations";
import ModulePage from "@/components/ModulePage";
import { cameraModule } from "@/data/consolidatedModules";
import { MathEquation } from "@/components/MathBlock";
import { ArrowLeft, GraduationCap, Lightbulb, BookOpen, Cpu, Aperture } from "lucide-react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

// Categorize camera module theory sections for inline rendering
const theoryByTitle: Record<string, typeof cameraModule.theory[0]> = {};
cameraModule.theory.forEach(s => { theoryByTitle[s.title] = s; });

function TheoryInline({ section }: { section: typeof cameraModule.theory[0] }) {
  return (
    <div className="concept-card">
      <h3 className="font-semibold text-foreground mb-3 text-sm">{section.title}</h3>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">{section.content}</p>
      {section.equations?.map((eq) => (
        <div key={eq.label} className="mb-3">
          <MathEquation tex={eq.tex} label={eq.label} />
          {eq.variables && eq.variables.length > 0 && (
            <div className="mt-1.5 rounded-lg bg-muted/30 border border-border p-3">
              <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">Where</p>
              <div className="space-y-0.5">
                {eq.variables.map((v: any) => (
                  <p key={v.symbol} className="text-xs text-muted-foreground">
                    <span className="font-mono text-foreground">{v.symbol}</span> = {v.meaning}
                  </p>
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function SectionHeader({ icon: Icon, title, number }: { icon: any; title: string; number: number }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <div
        className="h-9 w-9 rounded-lg flex items-center justify-center shrink-0"
        style={{ backgroundColor: `hsl(220 70% 55% / 0.12)` }}
      >
        <Icon className="h-4 w-4" style={{ color: `hsl(220, 70%, 55%)` }} />
      </div>
      <div>
        <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">Part {number}</p>
        <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">{title}</h2>
      </div>
    </div>
  );
}

export default function CameraModule() {
  const color = cameraModule.color;

  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-start gap-4 mb-10"
      >
        <div
          className="h-12 w-12 rounded-xl flex items-center justify-center shrink-0"
          style={{ backgroundColor: `hsl(${color} / 0.12)` }}
        >
          <GraduationCap className="h-6 w-6" style={{ color: `hsl(${color})` }} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground tracking-tight">{cameraModule.title}</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{cameraModule.subtitle}</p>
        </div>
      </motion.div>

      <div className="space-y-10">
        {/* ═══ PART 1: Image Formation & Projection ═══ */}
        <motion.section initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <SectionHeader icon={Lightbulb} title="Image Formation & Projection" number={1} />

          <div className="space-y-6">
            {/* Intuition theory */}
            {theoryByTitle["Intuition"] && <TheoryInline section={theoryByTitle["Intuition"]} />}

            {/* 3D → 2D Projection animation */}
            <Projection3DScene />

            {/* Pinhole Camera Model theory */}
            {theoryByTitle["Pinhole Camera Model"] && <TheoryInline section={theoryByTitle["Pinhole Camera Model"]} />}

            {/* Pinhole animation */}
            <PinholeScene />
          </div>
        </motion.section>

        {/* ═══ PART 2: Perspective Projection ═══ */}
        <motion.section initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <SectionHeader icon={BookOpen} title="Perspective Projection & Similar Triangles" number={2} />

          <div className="space-y-6">
            <PerspectiveScene />
          </div>
        </motion.section>

        {/* ═══ PART 3: Lens & Distortion ═══ */}
        <motion.section initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <SectionHeader icon={Aperture} title="Lenses, Distortion & Depth of Field" number={3} />

          <div className="space-y-6">
            {theoryByTitle["Lens Distortion"] && <TheoryInline section={theoryByTitle["Lens Distortion"]} />}

            <LensScene />
          </div>
        </motion.section>

        {/* ═══ PART 4: Intrinsic Matrix, Sensor & Calibration ═══ */}
        <motion.section initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
          <SectionHeader icon={Cpu} title="Intrinsic Matrix, Sensor & Calibration" number={4} />

          <div className="space-y-6">
            {/* Intrinsic matrix animation */}
            <IntrinsicScene />

            {/* Image Formation & Sensor theory */}
            {theoryByTitle["Image Formation & Sensor"] && <TheoryInline section={theoryByTitle["Image Formation & Sensor"]} />}

            {/* Sensor animation */}
            <SensorScene />

            {/* Calibration theory */}
            {theoryByTitle["Camera Calibration (Zhang's Method)"] && <TheoryInline section={theoryByTitle["Camera Calibration (Zhang's Method)"]} />}
          </div>
        </motion.section>

        {/* ═══ Rest of module content (Algorithms, Papers, Quiz, etc.) ═══ */}
        <ModulePage content={cameraModule} hideHeader />
      </div>
    </div>
  );
}
