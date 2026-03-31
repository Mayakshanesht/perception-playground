import { Projection3DScene, PinholeScene, PerspectiveScene, IntrinsicScene, LensScene, SensorScene, ColorSpaceScene } from "@/components/CameraAnimations";
import ModulePage from "@/components/ModulePage";
import { cameraModule } from "@/data/consolidatedModules";
import { MathEquation } from "@/components/MathBlock";
import AITutor from "@/components/AITutor";
import { ArrowLeft, GraduationCap, Lightbulb, BookOpen, Cpu, Aperture, Palette, Eye } from "lucide-react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useSectionObserver } from "@/hooks/useSectionObserver";
import { Progress } from "@/components/ui/progress";

// Categorize camera module theory sections for inline rendering
const theoryByTitle: Record<string, typeof cameraModule.theory[0]> = {};
cameraModule.theory.forEach(s => { theoryByTitle[s.title] = s; });

function TheoryInline({ section }: { section: typeof cameraModule.theory[0] }) {
  return (
    <div className="concept-card">
      <div className="flex items-center flex-wrap gap-y-1 mb-3">
        <h3 className="font-semibold text-foreground text-sm">{section.title}</h3>
        <AITutor conceptTitle={section.title} conceptContent={section.content} moduleName="Camera Image Formation" />
      </div>
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

function SectionHeader({ icon: Icon, title, number, subtitle }: { icon: any; title: string; number: number; subtitle?: string }) {
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
        {subtitle && <p className="text-[10px] text-muted-foreground mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}

const cameraSections = ["cam-s1", "cam-s2", "cam-s3", "cam-s4", "cam-s5", "cam-s6"];

export default function CameraModule() {
  const color = cameraModule.color;
  const progressPct = useSectionObserver("camera", cameraSections);

  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-start gap-4 mb-4"
      >
        <div
          className="h-12 w-12 rounded-xl flex items-center justify-center shrink-0"
          style={{ backgroundColor: `hsl(${color} / 0.12)` }}
        >
          <GraduationCap className="h-6 w-6" style={{ color: `hsl(${color})` }} />
        </div>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-foreground tracking-tight">{cameraModule.title}</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{cameraModule.subtitle}</p>
        </div>
      </motion.div>

      {/* Progress bar */}
      <div className="flex items-center gap-3 mb-10">
        <Progress value={progressPct} className="h-2 flex-1" />
        <span className="text-xs font-mono text-muted-foreground">{progressPct}%</span>
      </div>

      <div className="space-y-12">
        {/* ═══ PART 1: Image Formation & Projection ═══ */}
        <motion.section id="cam-s1" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <SectionHeader icon={Lightbulb} title="Image Formation & Projection" number={1} subtitle="How does a camera turn the 3D world into a flat image?" />

          <div className="space-y-6">
            {/* Start with the intuition */}
            {theoryByTitle["Intuition"] && <TheoryInline section={theoryByTitle["Intuition"]} />}

            {/* Immediately show the 3D → 2D animation so students visualize the concept */}
            <Projection3DScene />

            {/* Now the formal pinhole model */}
            {theoryByTitle["Pinhole Camera Model"] && <TheoryInline section={theoryByTitle["Pinhole Camera Model"]} />}

            {/* Interactive pinhole to let them play with the parameters */}
            <PinholeScene />
          </div>
        </motion.section>

        {/* ═══ PART 2: Perspective Projection & Similar Triangles ═══ */}
        <motion.section id="cam-s2" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <SectionHeader icon={BookOpen} title="Perspective Projection & Similar Triangles" number={2} subtitle="The geometry that makes projection work — connect f, Z, and image coordinates" />

          <div className="space-y-6">
            <div className="concept-card">
              <h3 className="font-semibold text-foreground mb-3 text-sm">Why Similar Triangles?</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                The pinhole model creates two similar triangles: one from the camera center to the 3D point (base = Z, height = Y), 
                and one from the camera center to the image plane (base = f, height = y'). Since these triangles share the same angle 
                at the camera center, their ratios are equal: <strong>y'/f = Y/Z</strong>. This single relationship is the foundation 
                of all perspective projection — it explains why distant objects appear smaller and why focal length controls field of view.
              </p>
            </div>

            {/* The improved perspective animation with split-view */}
            <PerspectiveScene />
          </div>
        </motion.section>

        {/* ═══ PART 3: Lens, Distortion & Depth of Field ═══ */}
        <motion.section id="cam-s3" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <SectionHeader icon={Aperture} title="Lenses, Distortion & Depth of Field" number={3} subtitle="Real cameras use lenses — introducing blur, distortion, and optical trade-offs" />

          <div className="space-y-6">
            {theoryByTitle["Lens Distortion"] && <TheoryInline section={theoryByTitle["Lens Distortion"]} />}

            {/* Lens animation showing thin lens equation, DOF, CoC */}
            <LensScene />
          </div>
        </motion.section>

        {/* ═══ PART 4: Intrinsic Matrix, Sensor & Calibration ═══ */}
        <motion.section id="cam-s4" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
          <SectionHeader icon={Cpu} title="Intrinsic Matrix, Sensor & Calibration" number={4} subtitle="From camera geometry to pixel coordinates — the K matrix and digital sensors" />

          <div className="space-y-6">
            {/* Intrinsic matrix animation — normalized coords → K → pixel coords */}
            <IntrinsicScene />

            {/* Sensor theory */}
            {theoryByTitle["Image Formation & Sensor"] && <TheoryInline section={theoryByTitle["Image Formation & Sensor"]} />}

            {/* Sensor animation with corrected Bayer pattern */}
            <SensorScene />

            {/* Calibration theory */}
            {theoryByTitle["Camera Calibration (Zhang's Method)"] && <TheoryInline section={theoryByTitle["Camera Calibration (Zhang's Method)"]} />}
          </div>
        </motion.section>

        {/* ═══ PART 5: Color Spaces & Image Manipulation ═══ */}
        <motion.section id="cam-s5" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
          <SectionHeader icon={Palette} title="Color Spaces & Image Manipulation" number={5} subtitle="RGB, HSV, gamma correction, and the pixel operations that power every CV pipeline" />

          <div className="space-y-6">
            {theoryByTitle["Color Spaces & Representations"] && <TheoryInline section={theoryByTitle["Color Spaces & Representations"]} />}

            {/* Interactive color space visualization */}
            <ColorSpaceScene />

            {theoryByTitle["Gamma Correction & Dynamic Range"] && <TheoryInline section={theoryByTitle["Gamma Correction & Dynamic Range"]} />}

            {theoryByTitle["Basic Image Operations"] && <TheoryInline section={theoryByTitle["Basic Image Operations"]} />}
          </div>
        </motion.section>

        {/* ═══ PART 6: Applications ═══ */}
        <motion.section initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}>
          <SectionHeader icon={Eye} title="Real-World Applications" number={6} subtitle="Where camera models and image formation matter in practice" />

          <div className="space-y-6">
            {theoryByTitle["Real-World Applications"] && <TheoryInline section={theoryByTitle["Real-World Applications"]} />}
          </div>
        </motion.section>

        {/* ═══ Rest of module content (Algorithms, Papers, Quiz, etc.) ═══ */}
        <ModulePage content={cameraModule} hideHeader hideTheory />
      </div>
    </div>
  );
}
