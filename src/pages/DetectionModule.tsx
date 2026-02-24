import { motion } from "framer-motion";
import { Scan, BookOpen, FlaskConical, FileText, ArrowLeft } from "lucide-react";
import { Link } from "react-router-dom";
import ArchitectureVisualizer from "@/components/ArchitectureVisualizer";
import ComparisonTable from "@/components/ComparisonTable";
import PapersTimeline from "@/components/PapersTimeline";

export default function DetectionModule() {
  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      <div className="flex items-start gap-4 mb-8">
        <div className="h-12 w-12 rounded-xl flex items-center justify-center shrink-0" style={{ backgroundColor: "hsl(160 84% 39% / 0.12)" }}>
          <Scan className="h-6 w-6" style={{ color: "hsl(160 84% 39%)" }} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground tracking-tight">Object Detection</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">
            Learn how detection models localize and classify objects using region proposals (two-stage) 
            or direct grid predictions (one-stage).
          </p>
        </div>
      </div>

      {/* Tabs-like sections */}
      <div className="space-y-8">
        {/* Concept Panel */}
        <section>
          <SectionHeader icon={BookOpen} title="Core Concepts" />
          <div className="grid md:grid-cols-2 gap-4">
            <ConceptCard
              title="Two-Stage Detectors"
              items={[
                "Stage 1: Region Proposal Network generates candidate boxes",
                "Stage 2: Classifier refines proposals and predicts classes",
                "Higher accuracy, slower inference",
                "Examples: R-CNN → Fast R-CNN → Faster R-CNN",
              ]}
              accent="265, 70%, 60%"
            />
            <ConceptCard
              title="One-Stage Detectors"
              items={[
                "Direct prediction: divide image into grid cells",
                "Each cell predicts bounding boxes + class scores",
                "Faster inference, real-time capable",
                "Examples: YOLO, SSD, RetinaNet",
              ]}
              accent="187, 85%, 53%"
            />
          </div>
        </section>

        {/* Architecture Visualizer */}
        <section>
          <SectionHeader icon={FlaskConical} title="Interactive Architecture Explorer" />
          <ArchitectureVisualizer />
        </section>

        {/* Comparison */}
        <section>
          <SectionHeader icon={Scan} title="YOLO vs Faster R-CNN" />
          <ComparisonTable />
        </section>

        {/* Papers */}
        <section>
          <SectionHeader icon={FileText} title="Key Papers Timeline" />
          <PapersTimeline />
        </section>
      </div>
    </div>
  );
}

function SectionHeader({ icon: Icon, title }: { icon: any; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <Icon className="h-4 w-4 text-primary" />
      <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">{title}</h2>
    </div>
  );
}

function ConceptCard({ title, items, accent }: { title: string; items: string[]; accent: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-border bg-card p-5"
    >
      <div className="flex items-center gap-2 mb-3">
        <div className="h-2 w-2 rounded-full" style={{ backgroundColor: `hsl(${accent})` }} />
        <h3 className="font-semibold text-sm text-foreground">{title}</h3>
      </div>
      <ul className="space-y-2">
        {items.map((item, i) => (
          <li key={i} className="text-xs text-muted-foreground flex items-start gap-2">
            <span className="text-muted-foreground/40 mt-0.5">→</span>
            {item}
          </li>
        ))}
      </ul>
    </motion.div>
  );
}
