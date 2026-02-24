import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Play, Pause, Info, X } from "lucide-react";

interface ArchBlock {
  id: string;
  label: string;
  type: "input" | "conv" | "pool" | "fc" | "output" | "special";
  details: {
    operation: string;
    inputShape: string;
    outputShape: string;
    params?: string;
    description: string;
  };
}

interface ArchitectureData {
  name: string;
  blocks: ArchBlock[];
}

const architectures: Record<string, ArchitectureData> = {
  yolo: {
    name: "YOLOv3",
    blocks: [
      { id: "input", label: "Input Image", type: "input", details: { operation: "Input", inputShape: "416×416×3", outputShape: "416×416×3", description: "RGB image resized to fixed dimensions for consistent processing." } },
      { id: "darknet1", label: "Darknet-53\nBlock 1", type: "conv", details: { operation: "Conv2D + BN + LeakyReLU", inputShape: "416×416×3", outputShape: "208×208×32", params: "928", description: "First convolutional block with batch normalization and LeakyReLU activation." } },
      { id: "darknet2", label: "Darknet-53\nBlock 2-5", type: "conv", details: { operation: "Residual Blocks ×8", inputShape: "208×208×32", outputShape: "52×52×256", params: "~8.7M", description: "Stacked residual blocks progressively downsample and extract hierarchical features." } },
      { id: "darknet3", label: "Darknet-53\nBlock 6", type: "conv", details: { operation: "Residual Blocks ×8", inputShape: "52×52×256", outputShape: "13×13×1024", params: "~18M", description: "Final backbone blocks producing deep semantic features at low resolution." } },
      { id: "detect1", label: "Detection\nScale 1", type: "special", details: { operation: "1×1 Conv → Predictions", inputShape: "13×13×1024", outputShape: "13×13×255", description: "Large object detection head. Predicts bounding boxes, objectness, and class probabilities for 3 anchor boxes." } },
      { id: "detect2", label: "Detection\nScale 2", type: "special", details: { operation: "Upsample + Concat + Conv", inputShape: "26×26×768", outputShape: "26×26×255", description: "Medium object detection with feature fusion from earlier layers via upsampling." } },
      { id: "detect3", label: "Detection\nScale 3", type: "special", details: { operation: "Upsample + Concat + Conv", inputShape: "52×52×384", outputShape: "52×52×255", description: "Small object detection with highest resolution feature maps." } },
      { id: "nms", label: "NMS", type: "output", details: { operation: "Non-Max Suppression", inputShape: "All scale predictions", outputShape: "Final detections", description: "Filters overlapping predictions using IoU thresholding to produce final bounding boxes." } },
    ],
  },
  fasterrcnn: {
    name: "Faster R-CNN",
    blocks: [
      { id: "input", label: "Input Image", type: "input", details: { operation: "Input", inputShape: "~800×600×3", outputShape: "~800×600×3", description: "Variable-size input image. Resized with aspect ratio preservation." } },
      { id: "backbone", label: "ResNet-50\nBackbone", type: "conv", details: { operation: "Conv + Residual Blocks", inputShape: "~800×600×3", outputShape: "~50×38×1024", params: "~23.5M", description: "Deep residual network extracts multi-scale feature maps. Shared between RPN and detection head." } },
      { id: "fpn", label: "FPN", type: "conv", details: { operation: "Feature Pyramid Network", inputShape: "Multi-scale features", outputShape: "P2-P5 pyramids", params: "~0.5M", description: "Builds top-down pathway with lateral connections for multi-scale feature representation." } },
      { id: "rpn", label: "Region Proposal\nNetwork", type: "special", details: { operation: "3×3 Conv → cls + reg", inputShape: "Feature maps", outputShape: "~2000 proposals", params: "~1.2M", description: "Slides over feature map to predict object/non-object scores and refine anchor box coordinates." } },
      { id: "roi", label: "RoI Pooling", type: "pool", details: { operation: "RoI Align", inputShape: "Feature map + proposals", outputShape: "7×7×256 per RoI", description: "Extracts fixed-size feature vectors from each proposal region using bilinear interpolation." } },
      { id: "head", label: "Detection\nHead", type: "fc", details: { operation: "FC layers", inputShape: "7×7×256", outputShape: "1024", params: "~12.8M", description: "Two fully-connected layers process pooled features for final classification and regression." } },
      { id: "cls", label: "Classification\n+ BBox Reg", type: "output", details: { operation: "Softmax + Linear", inputShape: "1024", outputShape: "N classes + 4 coords", description: "Final prediction: class probabilities and bounding box refinement for each proposal." } },
    ],
  },
};

const typeColors: Record<string, string> = {
  input: "var(--primary)",
  conv: "160, 84%, 39%",
  pool: "32, 95%, 55%",
  fc: "265, 70%, 60%",
  output: "340, 75%, 55%",
  special: "200, 80%, 55%",
};

function getColor(type: string) {
  const c = typeColors[type];
  if (c?.startsWith("var")) return `hsl(var(--primary))`;
  return `hsl(${c})`;
}
function getColorBg(type: string) {
  const c = typeColors[type];
  if (c?.startsWith("var")) return `hsl(var(--primary) / 0.12)`;
  return `hsl(${c} / 0.12)`;
}

export default function ArchitectureVisualizer() {
  const [selected, setSelected] = useState<string>("yolo");
  const [activeBlock, setActiveBlock] = useState<ArchBlock | null>(null);
  const [animating, setAnimating] = useState(false);
  const [flowIndex, setFlowIndex] = useState(-1);

  const arch = architectures[selected];

  const startAnimation = () => {
    if (animating) {
      setAnimating(false);
      setFlowIndex(-1);
      return;
    }
    setAnimating(true);
    setFlowIndex(0);
    let i = 0;
    const interval = setInterval(() => {
      i++;
      if (i >= arch.blocks.length) {
        clearInterval(interval);
        setAnimating(false);
        setFlowIndex(-1);
      } else {
        setFlowIndex(i);
      }
    }, 800);
  };

  return (
    <div>
      {/* Controls */}
      <div className="flex items-center gap-3 mb-6">
        <div className="flex gap-1 p-1 rounded-lg bg-muted">
          {Object.entries(architectures).map(([key, val]) => (
            <button
              key={key}
              onClick={() => { setSelected(key); setActiveBlock(null); setFlowIndex(-1); setAnimating(false); }}
              className={`px-4 py-1.5 rounded-md text-xs font-medium transition-all ${
                selected === key ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {val.name}
            </button>
          ))}
        </div>
        <button
          onClick={startAnimation}
          className="flex items-center gap-2 px-4 py-1.5 rounded-md text-xs font-medium bg-card border border-border text-foreground hover:border-primary/50 transition-all"
        >
          {animating ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
          {animating ? "Stop" : "Forward Pass"}
        </button>
      </div>

      {/* Architecture Diagram */}
      <div className="relative rounded-xl border border-border bg-card p-6 overflow-x-auto">
        <div className="grid-pattern absolute inset-0 rounded-xl opacity-30" />
        <div className="relative flex items-center gap-3 min-w-max">
          {arch.blocks.map((block, i) => (
            <div key={block.id} className="flex items-center gap-3">
              <motion.div
                layout
                onClick={() => setActiveBlock(activeBlock?.id === block.id ? null : block)}
                className={`node-block relative min-w-[120px] text-center whitespace-pre-line ${
                  activeBlock?.id === block.id ? "active" : ""
                } ${flowIndex === i ? "active" : ""}`}
                style={{
                  borderColor: flowIndex === i ? getColor(block.type) : undefined,
                  boxShadow: flowIndex === i ? `0 0 20px -4px ${getColor(block.type)}` : undefined,
                }}
                animate={flowIndex === i ? { scale: [1, 1.05, 1] } : {}}
                transition={{ duration: 0.3 }}
              >
                <div
                  className="absolute top-0 left-0 right-0 h-0.5 rounded-t-lg"
                  style={{ backgroundColor: getColor(block.type) }}
                />
                <p className="text-xs font-medium text-foreground leading-tight">{block.label}</p>
                <p className="text-[10px] font-mono text-muted-foreground mt-1">{block.details.outputShape}</p>
              </motion.div>
              {i < arch.blocks.length - 1 && (
                <div className="relative">
                  <ArrowRight className="h-4 w-4 text-muted-foreground/40" />
                  {flowIndex === i && (
                    <motion.div
                      className="absolute inset-0 flex items-center justify-center"
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 8 }}
                    >
                      <ArrowRight className="h-4 w-4 text-primary" />
                    </motion.div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Block Detail Panel */}
      <AnimatePresence>
        {activeBlock && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-4 rounded-xl border border-border bg-card p-5">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: getColor(activeBlock.type) }} />
                  <h4 className="font-semibold text-foreground text-sm">{activeBlock.label.replace("\n", " ")}</h4>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full text-muted-foreground" style={{ backgroundColor: getColorBg(activeBlock.type) }}>
                    {activeBlock.type}
                  </span>
                </div>
                <button onClick={() => setActiveBlock(null)} className="text-muted-foreground hover:text-foreground">
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Operation</p>
                  <p className="text-xs font-mono text-foreground">{activeBlock.details.operation}</p>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Input Shape</p>
                  <p className="text-xs font-mono text-foreground">{activeBlock.details.inputShape}</p>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Output Shape</p>
                  <p className="text-xs font-mono text-foreground">{activeBlock.details.outputShape}</p>
                </div>
                {activeBlock.details.params && (
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Parameters</p>
                    <p className="text-xs font-mono text-foreground">{activeBlock.details.params}</p>
                  </div>
                )}
              </div>
              <div className="flex items-start gap-2 rounded-lg bg-muted p-3">
                <Info className="h-4 w-4 text-primary shrink-0 mt-0.5" />
                <p className="text-xs text-muted-foreground leading-relaxed">{activeBlock.details.description}</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
