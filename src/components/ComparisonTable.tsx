const comparisons = [
  { feature: "Approach", yolo: "Single-stage (grid-based)", fasterrcnn: "Two-stage (proposal + classify)" },
  { feature: "Speed", yolo: "~45 FPS (real-time)", fasterrcnn: "~5-7 FPS" },
  { feature: "mAP (COCO)", yolo: "~33-57 mAP", fasterrcnn: "~37-42 mAP" },
  { feature: "Small Objects", yolo: "Weaker (improved in v3+)", fasterrcnn: "Better with FPN" },
  { feature: "Backbone", yolo: "Darknet-53", fasterrcnn: "ResNet-50 + FPN" },
  { feature: "Proposals", yolo: "None (direct prediction)", fasterrcnn: "~2000 per image" },
  { feature: "Loss Function", yolo: "Multi-part (coord + obj + cls)", fasterrcnn: "Smooth L1 + Cross-Entropy" },
  { feature: "Use Case", yolo: "Real-time: robotics, video", fasterrcnn: "High accuracy: medical, satellite" },
];

export default function ComparisonTable() {
  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left px-5 py-3 text-muted-foreground font-medium uppercase tracking-wider">Feature</th>
              <th className="text-left px-5 py-3 font-medium uppercase tracking-wider" style={{ color: "hsl(187 85% 53%)" }}>YOLOv3</th>
              <th className="text-left px-5 py-3 font-medium uppercase tracking-wider" style={{ color: "hsl(265 70% 60%)" }}>Faster R-CNN</th>
            </tr>
          </thead>
          <tbody>
            {comparisons.map((row, i) => (
              <tr key={row.feature} className={`border-b border-border/50 ${i % 2 === 0 ? "" : "bg-muted/30"}`}>
                <td className="px-5 py-3 text-foreground font-medium">{row.feature}</td>
                <td className="px-5 py-3 text-muted-foreground font-mono">{row.yolo}</td>
                <td className="px-5 py-3 text-muted-foreground font-mono">{row.fasterrcnn}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
