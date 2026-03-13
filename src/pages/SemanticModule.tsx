import ModulePage from "@/components/ModulePage";
import { ModuleContent } from "@/data/moduleContent";
import { moduleContents } from "@/data/moduleContent";
import { MathEquation } from "@/components/MathBlock";
import SemanticSceneVisualizer from "@/components/SemanticAnimations";
import { ArrowLeft, GraduationCap, Lightbulb, Target, Grid3X3, Layers, Brain, Eye } from "lucide-react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

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

const color = semanticModule.color;

// Index theory by title for inline access
const theoryByTitle: Record<string, typeof semanticModule.theory[0]> = {};
semanticModule.theory.forEach(s => { theoryByTitle[s.title] = s; });

function TheoryInline({ title }: { title: string }) {
  const section = theoryByTitle[title];
  if (!section) return null;
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

function SectionHeader({ icon: Icon, title, number, subtitle }: { icon: any; title: string; number: number; subtitle?: string }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <div
        className="h-9 w-9 rounded-lg flex items-center justify-center shrink-0"
        style={{ backgroundColor: `hsl(${color} / 0.12)` }}
      >
        <Icon className="h-4 w-4" style={{ color: `hsl(${color})` }} />
      </div>
      <div>
        <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">Part {number}</p>
        <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">{title}</h2>
        {subtitle && <p className="text-[10px] text-muted-foreground mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}

export default function SemanticModule() {
  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto">
      {/* Header */}
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      <div className="flex items-start gap-4 mb-8">
        <div
          className="h-12 w-12 rounded-xl flex items-center justify-center shrink-0"
          style={{ backgroundColor: `hsl(${color} / 0.12)` }}
        >
          <GraduationCap className="h-6 w-6" style={{ color: `hsl(${color})` }} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground tracking-tight">{semanticModule.title}</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">{semanticModule.subtitle}</p>
        </div>
      </div>

      <div className="space-y-10">
        {/* ═══ Part 1: The Semantic Hierarchy ═══ */}
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
          <SectionHeader icon={Eye} title="The Semantic Hierarchy" number={1} subtitle="From scene labels to pixel-level understanding" />

          <div className="rounded-xl border border-border bg-card/50 p-4 mb-4">
            <p className="text-sm text-muted-foreground leading-relaxed">
              Semantic understanding in computer vision follows a natural hierarchy of increasing spatial precision:
              <strong className="text-foreground"> Classification</strong> (one label per image) →
              <strong className="text-foreground"> Detection</strong> (locate + classify objects) →
              <strong className="text-foreground"> Semantic Segmentation</strong> (label every pixel by class) →
              <strong className="text-foreground"> Instance Segmentation</strong> (separate individual objects at pixel level).
              Each step adds spatial granularity to the model's understanding.
            </p>
          </div>

          {/* Interactive scene visualizer */}
          <SemanticSceneVisualizer />
        </motion.section>

        {/* ═══ Part 2: Classification ═══ */}
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1, duration: 0.4 }}>
          <SectionHeader icon={Lightbulb} title="Image Classification" number={2} subtitle="Assign a single label to an entire image" />
          <div className="space-y-4">
            <TheoryInline title="What is Image Classification?" />
            <TheoryInline title="Convolutional Neural Networks (CNNs)" />
            <TheoryInline title="Batch Normalization" />
            <TheoryInline title="Residual Learning (ResNet)" />
            <TheoryInline title="Vision Transformers (ViT)" />
          </div>
        </motion.section>

        {/* ═══ Part 3: Object Detection ═══ */}
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15, duration: 0.4 }}>
          <SectionHeader icon={Target} title="Object Detection" number={3} subtitle="Localize and classify multiple objects" />
          <div className="space-y-4">
            <TheoryInline title="What is Object Detection?" />
            <TheoryInline title="Two-Stage Detectors: R-CNN Family" />
            <TheoryInline title="One-Stage Detectors: YOLO & SSD" />
            <TheoryInline title="Non-Maximum Suppression (NMS)" />
            <TheoryInline title="Feature Pyramid Networks (FPN)" />
          </div>
        </motion.section>

        {/* ═══ Part 4: Segmentation ═══ */}
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2, duration: 0.4 }}>
          <SectionHeader icon={Grid3X3} title="Segmentation" number={4} subtitle="Pixel-level scene understanding" />
          <div className="space-y-4">
            <TheoryInline title="Types of Segmentation" />
            <TheoryInline title="Fully Convolutional Networks (FCN)" />
            <TheoryInline title="U-Net Architecture" />
            <TheoryInline title="Atrous/Dilated Convolutions (DeepLab)" />
            <TheoryInline title="Instance Segmentation: Mask R-CNN" />
          </div>
        </motion.section>

        {/* ═══ Part 5: Remaining module content (Algorithms, Papers, Playground, Quiz, etc.) ═══ */}
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25, duration: 0.4 }}>
          <SectionHeader icon={Layers} title="Algorithms, Papers & Practice" number={5} subtitle="Pipelines, key papers, playgrounds, and quizzes" />
          <ModulePage content={semanticModule} hideHeader hideTheory />
        </motion.section>
      </div>
    </div>
  );
}
