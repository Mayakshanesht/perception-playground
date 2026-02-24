import { motion } from "framer-motion";

const papers = [
  { year: 2014, title: "R-CNN", authors: "Girshick et al.", venue: "CVPR", summary: "Introduced region-based CNN for object detection using selective search proposals." },
  { year: 2015, title: "Fast R-CNN", authors: "Girshick", venue: "ICCV", summary: "Shared convolutional features across proposals with RoI pooling for efficiency." },
  { year: 2015, title: "Faster R-CNN", authors: "Ren et al.", venue: "NeurIPS", summary: "Replaced selective search with a learned Region Proposal Network (RPN)." },
  { year: 2016, title: "YOLOv1", authors: "Redmon et al.", venue: "CVPR", summary: "First real-time single-stage detector treating detection as regression." },
  { year: 2016, title: "SSD", authors: "Liu et al.", venue: "ECCV", summary: "Multi-scale feature maps for single-shot detection at different resolutions." },
  { year: 2017, title: "RetinaNet", authors: "Lin et al.", venue: "ICCV", summary: "Focal loss to address class imbalance in one-stage detectors." },
  { year: 2018, title: "YOLOv3", authors: "Redmon & Farhadi", venue: "arXiv", summary: "Multi-scale predictions with Darknet-53 backbone for better small object detection." },
  { year: 2020, title: "DETR", authors: "Carion et al.", venue: "ECCV", summary: "End-to-end detection with transformers, eliminating NMS and anchors." },
];

export default function PapersTimeline() {
  return (
    <div className="relative">
      {/* Timeline line */}
      <div className="absolute left-[72px] top-0 bottom-0 w-px bg-border" />

      <div className="space-y-4">
        {papers.map((paper, i) => (
          <motion.div
            key={paper.title}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.06 }}
            className="flex gap-4 group"
          >
            <div className="w-[60px] shrink-0 text-right">
              <span className="text-xs font-mono text-muted-foreground">{paper.year}</span>
            </div>
            <div className="relative shrink-0 mt-1.5">
              <div className="h-3 w-3 rounded-full border-2 border-border bg-background group-hover:border-primary transition-colors" />
            </div>
            <div className="rounded-lg border border-border bg-card p-4 flex-1 group-hover:border-primary/30 transition-colors">
              <div className="flex items-baseline gap-2 mb-1">
                <h4 className="text-sm font-semibold text-foreground">{paper.title}</h4>
                <span className="text-[10px] font-mono text-muted-foreground">{paper.venue}</span>
              </div>
              <p className="text-[10px] text-primary/70 mb-1">{paper.authors}</p>
              <p className="text-xs text-muted-foreground leading-relaxed">{paper.summary}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
