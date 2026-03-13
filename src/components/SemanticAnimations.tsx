import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Mode = "clf" | "det" | "sem" | "inst";

const TAGS: Record<Mode, string> = {
  clf: "f(x) → class label",
  det: "f(x) → {class, bbox, conf}",
  sem: "f(x) → pixel-class map (H×W)",
  inst: "f(x) → {class, mask, id} × N",
};

const TAB_LABELS: { id: Mode; num: string; label: string }[] = [
  { id: "clf", num: "01", label: "Classification" },
  { id: "det", num: "02", label: "Detection" },
  { id: "sem", num: "03", label: "Semantic Seg" },
  { id: "inst", num: "04", label: "Instance Seg" },
];

/* ══════ BASE SCENE (shared street SVG) ══════ */
function BaseScene() {
  return (
    <svg className="block w-full" viewBox="0 0 600 280" xmlns="http://www.w3.org/2000/svg">
      {/* Sky */}
      <rect x="0" y="0" width="600" height="162" fill="#87CEEB" />
      <rect x="0" y="130" width="600" height="32" fill="#9AD4E8" opacity=".45" />
      {/* Clouds */}
      <ellipse cx="72" cy="40" rx="40" ry="17" fill="white" opacity=".88" />
      <ellipse cx="102" cy="38" rx="29" ry="13" fill="white" opacity=".88" />
      <ellipse cx="48" cy="43" rx="24" ry="11" fill="white" opacity=".88" />
      <ellipse cx="310" cy="30" rx="34" ry="14" fill="white" opacity=".80" />
      <ellipse cx="340" cy="29" rx="25" ry="11" fill="white" opacity=".80" />
      <ellipse cx="285" cy="33" rx="20" ry="9" fill="white" opacity=".80" />
      {/* Sun */}
      <circle cx="545" cy="36" r="21" fill="#FFD700" opacity=".92" />
      <circle cx="545" cy="36" r="27" fill="#FFE082" opacity=".18" />

      {/* Building */}
      <rect x="374" y="22" width="222" height="207" fill="#C9B89A" />
      <rect x="368" y="14" width="234" height="13" fill="#B8A585" />
      <rect x="378" y="70" width="214" height="2" fill="#B0A082" opacity=".55" />
      <rect x="378" y="112" width="214" height="2" fill="#B0A082" opacity=".55" />
      <rect x="378" y="154" width="214" height="2" fill="#B0A082" opacity=".55" />
      {/* Windows */}
      {[30, 78, 120, 162].map((row, ri) =>
        [393, 435, 477, 519].map((col, ci) => (
          <rect key={`w-${ri}-${ci}`} x={col} y={row} width={ci === 3 ? 26 : 28} height="22"
            fill={ri < 2 ? (ci === 3 && ri === 0 ? "#FFE082" : "#90CAF9") : "#B3E5FC"}
            opacity={ri < 2 ? ".88" : ri === 2 ? ".75" : ".65"} rx="1" />
        ))
      )}
      {/* Door */}
      <rect x="462" y="185" width="32" height="46" fill="#8D6E63" />
      <rect x="462" y="185" width="32" height="6" fill="#7A5E56" />
      <circle cx="490" cy="210" r="2.5" fill="#FFD700" />

      {/* Tree */}
      <rect x="138" y="176" width="13" height="50" fill="#6D4C41" />
      <ellipse cx="144" cy="148" rx="44" ry="40" fill="#1B5E20" />
      <ellipse cx="128" cy="162" rx="29" ry="25" fill="#2E7D32" />
      <ellipse cx="161" cy="156" rx="31" ry="27" fill="#388E3C" />
      <ellipse cx="144" cy="137" rx="36" ry="30" fill="#43A047" />
      <ellipse cx="130" cy="143" rx="21" ry="17" fill="#4CAF50" />

      {/* Ground/Grass */}
      <rect x="0" y="208" width="600" height="16" fill="#558B2F" />
      <rect x="0" y="208" width="600" height="4" fill="#33691E" opacity=".45" />
      {/* Sidewalk */}
      <rect x="0" y="220" width="600" height="22" fill="#B0BEC5" />
      {[75, 150, 225, 300, 375, 450, 525].map(x => (
        <line key={x} x1={x} y1="220" x2={x} y2="242" stroke="#9E9E9E" strokeWidth=".7" opacity=".55" />
      ))}
      {/* Road */}
      <rect x="0" y="242" width="600" height="52" fill="#37474F" />
      <rect x="0" y="242" width="600" height="3" fill="#263238" opacity=".6" />
      {Array.from({ length: 10 }, (_, i) => (
        <rect key={i} x={5 + i * 58} y="267" width="42" height="4" fill="#FFEE58" opacity=".8" />
      ))}

      {/* Person 1 */}
      <ellipse cx="186" cy="242" rx="12" ry="3" fill="#263238" opacity=".35" />
      <rect x="179" y="236" width="6" height="5" fill="#212121" rx="1" />
      <rect x="187" y="236" width="6" height="5" fill="#212121" rx="1" />
      <rect x="180" y="222" width="5" height="15" fill="#1A237E" />
      <rect x="187" y="223" width="5" height="14" fill="#1A237E" />
      <rect x="175" y="208" width="18" height="16" fill="#1565C0" rx="2" />
      <rect x="181" y="208" width="6" height="5" fill="#1976D2" rx="1" />
      <line x1="175" y1="214" x2="166" y2="223" stroke="#FDBCB4" strokeWidth="4" strokeLinecap="round" />
      <line x1="193" y1="213" x2="201" y2="221" stroke="#FDBCB4" strokeWidth="4" strokeLinecap="round" />
      <rect x="183" y="200" width="6" height="9" fill="#FDBCB4" />
      <circle cx="186" cy="196" r="9.5" fill="#FDBCB4" />
      <ellipse cx="186" cy="189" rx="9.5" ry="5.5" fill="#3E2723" />
      <circle cx="183" cy="195" r="1.2" fill="#455A64" />
      <circle cx="189" cy="195" r="1.2" fill="#455A64" />

      {/* Person 2 */}
      <ellipse cx="432" cy="242" rx="11" ry="3" fill="#263238" opacity=".35" />
      <rect x="425" y="236" width="6" height="5" fill="#212121" rx="1" />
      <rect x="433" y="236" width="5" height="5" fill="#212121" rx="1" />
      <rect x="426" y="221" width="5" height="16" fill="#37474F" />
      <rect x="433" y="222" width="5" height="15" fill="#37474F" />
      <rect x="422" y="208" width="16" height="15" fill="#C62828" rx="2" />
      <rect x="427" y="208" width="6" height="5" fill="#D32F2F" rx="1" />
      <line x1="422" y1="213" x2="413" y2="221" stroke="#FFCCBC" strokeWidth="3.5" strokeLinecap="round" />
      <line x1="438" y1="213" x2="446" y2="220" stroke="#FFCCBC" strokeWidth="3.5" strokeLinecap="round" />
      <rect x="428" y="199" width="6" height="9" fill="#FFCCBC" />
      <circle cx="431" cy="194" r="9" fill="#FFCCBC" />
      <ellipse cx="431" cy="187" rx="9" ry="5" fill="#212121" />
      <circle cx="428" cy="193" r="1.2" fill="#455A64" />
      <circle cx="434" cy="193" r="1.2" fill="#455A64" />

      {/* Car 1 (red) */}
      <ellipse cx="110" cy="294" rx="73" ry="5.5" fill="#161824" opacity=".55" />
      <rect x="28" y="249" width="162" height="39" fill="#E53935" rx="5" />
      <rect x="54" y="237" width="95" height="18" fill="#C62828" rx="6" />
      <rect x="59" y="239" width="39" height="13" fill="#B3E5FC" opacity=".88" rx="2" />
      <rect x="104" y="239" width="38" height="13" fill="#B3E5FC" opacity=".88" rx="2" />
      <rect x="41" y="278" width="42" height="11" fill="#B71C1C" rx="2" />
      <rect x="131" y="278" width="42" height="11" fill="#B71C1C" rx="2" />
      <circle cx="63" cy="289" r="14" fill="#212121" />
      <circle cx="63" cy="289" r="8.5" fill="#424242" />
      <circle cx="63" cy="289" r="4" fill="#616161" />
      <circle cx="151" cy="289" r="14" fill="#212121" />
      <circle cx="151" cy="289" r="8.5" fill="#424242" />
      <circle cx="151" cy="289" r="4" fill="#616161" />
      <rect x="28" y="259" width="11" height="7" fill="#FFF9C4" opacity=".95" rx="1" />
      <rect x="179" y="259" width="11" height="7" fill="#FF5252" opacity=".9" rx="1" />
      <rect x="28" y="273" width="162" height="3" fill="#B71C1C" />

      {/* Car 2 (blue) */}
      <ellipse cx="305" cy="294" rx="69" ry="5" fill="#161824" opacity=".55" />
      <rect x="238" y="251" width="132" height="37" fill="#1565C0" rx="5" />
      <rect x="260" y="239" width="90" height="18" fill="#0D47A1" rx="6" />
      <rect x="264" y="241" width="36" height="13" fill="#B3E5FC" opacity=".88" rx="2" />
      <rect x="306" y="241" width="37" height="13" fill="#B3E5FC" opacity=".88" rx="2" />
      <rect x="251" y="278" width="40" height="10" fill="#0A3880" rx="2" />
      <rect x="341" y="278" width="40" height="10" fill="#0A3880" rx="2" />
      <circle cx="271" cy="289" r="13.5" fill="#212121" />
      <circle cx="271" cy="289" r="8" fill="#424242" />
      <circle cx="271" cy="289" r="3.5" fill="#616161" />
      <circle cx="355" cy="289" r="13.5" fill="#212121" />
      <circle cx="355" cy="289" r="8" fill="#424242" />
      <circle cx="355" cy="289" r="3.5" fill="#616161" />
      <rect x="238" y="261" width="11" height="7" fill="#FFF9C4" opacity=".95" rx="1" />
      <rect x="359" y="261" width="11" height="7" fill="#FF5252" opacity=".9" rx="1" />
      <rect x="238" y="274" width="132" height="3" fill="#0A3880" />
    </svg>
  );
}

/* ══════ ANIMATED OVERLAYS ══════ */

const fadeIn = (delay: number) => ({
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  transition: { delay, duration: 0.4, ease: "easeOut" },
});

const drawBox = (delay: number) => ({
  initial: { pathLength: 0, opacity: 0 },
  animate: { pathLength: 1, opacity: 1 },
  transition: { delay, duration: 0.65, ease: [0.15, 0, 0.3, 1] },
});

const popIn = (delay: number) => ({
  initial: { opacity: 0, y: 4 },
  animate: { opacity: 1, y: 0 },
  transition: { delay, duration: 0.25, ease: "easeOut" },
});

function ClassificationOverlay() {
  return (
    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 600 280" xmlns="http://www.w3.org/2000/svg">
      {/* Scan grid */}
      {[56, 112, 168, 224].map((y, i) => (
        <motion.line key={`h-${y}`} x1="0" y1={y} x2="600" y2={y} stroke="#4E86FF" strokeWidth=".5" {...fadeIn(0.05 * (i + 1))} />
      ))}
      {[100, 200, 300, 400, 500].map((x, i) => (
        <motion.line key={`v-${x}`} x1={x} y1="0" x2={x} y2="280" stroke="#4E86FF" strokeWidth=".5" {...fadeIn(0.08 + 0.04 * i)} />
      ))}
      {/* Pulsing border */}
      <motion.rect x="2" y="2" width="596" height="276" fill="none" stroke="#4E86FF" strokeWidth="2" rx="4"
        {...fadeIn(0.1)} animate={{ opacity: [0.7, 1, 0.7] }} transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }} />
      {/* Corner brackets */}
      {[
        "M2 24L2 2L24 2",
        "M576 2L598 2L598 24",
        "M2 256L2 278L24 278",
        "M576 278L598 278L598 256",
      ].map((d, i) => (
        <motion.path key={i} d={d} fill="none" stroke="#4E86FF" strokeWidth="2.5" strokeLinecap="square" {...fadeIn(0.3)} />
      ))}
      {/* Softmax panel */}
      <motion.g {...fadeIn(0.35)}>
        <rect x="4" y="4" width="172" height="116" fill="#08090F" opacity=".93" rx="4" />
        <rect x="4" y="4" width="172" height="116" fill="none" stroke="#1A1D2E" strokeWidth=".8" rx="4" />
      </motion.g>
      <motion.text x="12" y="21" fill="#4E86FF" fontFamily="'Courier New',monospace" fontSize="9" letterSpacing=".12em" {...fadeIn(0.4)}>SOFTMAX P(y|x)</motion.text>
      <motion.line x1="4" y1="28" x2="176" y2="28" stroke="#1A1D2E" strokeWidth=".6" {...fadeIn(0.4)} />
      {/* Bars */}
      {[
        { label: "urban", w: 90, pct: "94%", y: 34, delay: 0.45, color: "#4E86FF", textColor: "#D0D4EE", lx: 12 },
        { label: "suburban", w: 5, pct: "3%", y: 50, delay: 0.5, color: "#2A4070", textColor: "#3E4260", lx: 12 },
        { label: "highway", w: 2, pct: "1.4%", y: 66, delay: 0.55, color: "#2A4070", textColor: "#3E4260", lx: 12 },
        { label: "rural", w: 1.5, pct: "0.8%", y: 82, delay: 0.6, color: "#2A4070", textColor: "#3E4260", lx: 12 },
        { label: "indoor", w: 1, pct: "0.5%", y: 98, delay: 0.65, color: "#2A4070", textColor: "#3E4260", lx: 12 },
      ].map((bar) => (
        <motion.g key={bar.label} {...fadeIn(bar.delay)}>
          <text x={bar.lx} y={bar.y + 10} fill={bar.textColor} fontFamily="'Courier New',monospace" fontSize="10">{bar.label}</text>
          <rect x={bar.label.length > 5 ? 76 : 64} y={bar.y} width={bar.label.length > 5 ? 88 : 96} height={bar.y === 34 ? 11 : 10} fill="#131623" rx="2" />
          <rect x={bar.label.length > 5 ? 76 : 64} y={bar.y} width={bar.w} height={bar.y === 34 ? 11 : 10} fill={bar.color} rx="2" opacity=".88" />
          <text x="170" y={bar.y + 9} fill={bar.y === 34 ? "#4E86FF" : "#3E4260"} fontFamily="'Courier New',monospace" fontSize="9" textAnchor="end">{bar.pct}</text>
        </motion.g>
      ))}
      {/* Central prediction */}
      <motion.g {...fadeIn(0.5)}>
        <rect x="174" y="98" width="252" height="82" fill="#08090F" opacity=".93" rx="6" />
        <rect x="174" y="98" width="252" height="82" fill="none" stroke="#4E86FF" strokeWidth="1.2" rx="6" />
      </motion.g>
      <motion.text x="300" y="118" textAnchor="middle" fill="#3E4260" fontFamily="'Courier New',monospace" fontSize="9" letterSpacing=".15em" {...fadeIn(0.55)}>PREDICTED CLASS</motion.text>
      <motion.text x="300" y="152" textAnchor="middle" fill="#D0D4EE" fontFamily="'Courier New',monospace" fontSize="24" fontWeight="bold" {...fadeIn(0.6)}>"urban"</motion.text>
      <motion.g {...fadeIn(0.65)}>
        <rect x="194" y="161" width="212" height="7" fill="#131623" rx="3" />
        <rect x="194" y="161" width="200" height="7" fill="#4E86FF" rx="3" opacity=".82" />
      </motion.g>
      <motion.text x="410" y="170" fill="#4E86FF" fontFamily="'Courier New',monospace" fontSize="9" textAnchor="end" {...fadeIn(0.7)}>94.2%</motion.text>
    </svg>
  );
}

function DetectionOverlay() {
  const boxes: { x: number; y: number; w: number; h: number; label: string; conf: string; color: string; dBox: number; dLbl: number }[] = [
    { x: 160, y: 184, w: 48, h: 62, label: "person", conf: "0.96", color: "#FF6B6B", dBox: 0.05, dLbl: 0.55 },
    { x: 412, y: 182, w: 42, h: 62, label: "person", conf: "0.92", color: "#FF6B6B", dBox: 0.2, dLbl: 0.7 },
    { x: 24, y: 232, w: 172, h: 52, label: "car", conf: "0.98", color: "#FFA940", dBox: 0.35, dLbl: 0.85 },
    { x: 233, y: 234, w: 142, h: 52, label: "car", conf: "0.89", color: "#FFA940", dBox: 0.5, dLbl: 1.0 },
    { x: 370, y: 14, w: 228, h: 215, label: "building", conf: "0.94", color: "#FFB347", dBox: 0.65, dLbl: 1.15 },
    { x: 92, y: 108, w: 108, h: 116, label: "tree", conf: "0.81", color: "#4DD07A", dBox: 0.8, dLbl: 1.3 },
  ];

  return (
    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 600 280" xmlns="http://www.w3.org/2000/svg">
      {/* HUD corners */}
      {["M2 24L2 2L24 2", "M576 2L598 2L598 24", "M2 256L2 278L24 278", "M576 278L598 278L598 256"].map((d, i) => (
        <path key={i} d={d} fill="none" stroke="#3E4260" strokeWidth="1" strokeLinecap="square" opacity=".6" />
      ))}
      {boxes.map((b) => (
        <g key={`${b.label}-${b.x}`}>
          <motion.rect x={b.x} y={b.y} width={b.w} height={b.h} fill="none" stroke={b.color} strokeWidth="1.8" rx="1"
            initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }}
            transition={{ delay: b.dBox, duration: 0.65, ease: [0.15, 0, 0.3, 1] }} />
          <motion.g {...popIn(b.dLbl)}>
            <rect x={b.x - 2} y={b.y - 15} width={76} height="14" fill={b.color} rx="2" />
            <text x={b.x + 2} y={b.y - 5} fill="#08090F" fontFamily="'Courier New',monospace" fontSize="9" fontWeight="bold">
              {b.label}  {b.conf}
            </text>
          </motion.g>
        </g>
      ))}
      {/* Stats HUD */}
      <motion.g {...popIn(1.4)}>
        <rect x="466" y="194" width="130" height="50" fill="#08090F" opacity=".9" rx="4" />
        <rect x="466" y="194" width="130" height="50" fill="none" stroke="#1A1D2E" strokeWidth=".6" rx="4" />
        <text x="474" y="210" fill="#3E4260" fontFamily="'Courier New',monospace" fontSize="9" letterSpacing=".1em">DETECTIONS</text>
        <text x="474" y="226" fill="#D0D4EE" fontFamily="'Courier New',monospace" fontSize="10">objects  :  6</text>
        <text x="474" y="240" fill="#D0D4EE" fontFamily="'Courier New',monospace" fontSize="10">IoU thr  : .50</text>
      </motion.g>
    </svg>
  );
}

function SemanticSegOverlay() {
  const masks: { el: JSX.Element; delay: number; op: number }[] = [
    { el: <rect x="0" y="0" width="600" height="162" fill="#5BB8F5" />, delay: 0, op: 0.58 },
    { el: <rect x="368" y="14" width="230" height="215" fill="#FFB347" />, delay: 0.12, op: 0.60 },
    { el: <><ellipse cx="144" cy="152" rx="56" ry="52" fill="#4DD07A" /><rect x="136" y="174" width="16" height="52" fill="#4DD07A" /></>, delay: 0.24, op: 0.65 },
    { el: <><ellipse cx="186" cy="218" rx="23" ry="34" fill="#FF6B6B" /><ellipse cx="431" cy="216" rx="21" ry="32" fill="#FF6B6B" /></>, delay: 0.36, op: 0.70 },
    { el: <><rect x="22" y="230" width="178" height="58" fill="#FFA940" /><rect x="230" y="232" width="148" height="58" fill="#FFA940" /></>, delay: 0.48, op: 0.65 },
    { el: <rect x="0" y="208" width="600" height="14" fill="#66BB6A" />, delay: 0.58, op: 0.50 },
    { el: <rect x="0" y="220" width="600" height="22" fill="#BABFDB" />, delay: 0.68, op: 0.50 },
    { el: <rect x="0" y="242" width="600" height="52" fill="#8B96C8" />, delay: 0.78, op: 0.52 },
  ];

  const legend = [
    { color: "#5BB8F5", label: "sky" }, { color: "#FFB347", label: "building" },
    { color: "#4DD07A", label: "vegetation" }, { color: "#FF6B6B", label: "person" },
    { color: "#FFA940", label: "car" }, { color: "#8B96C8", label: "road" },
    { color: "#BABFDB", label: "sidewalk" }, { color: "#66BB6A", label: "terrain" },
  ];

  return (
    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 600 280" xmlns="http://www.w3.org/2000/svg">
      {masks.map((m, i) => (
        <motion.g key={i} initial={{ opacity: 0 }} animate={{ opacity: m.op }} transition={{ delay: m.delay, duration: 0.4, ease: "easeOut" }}>
          {m.el}
        </motion.g>
      ))}
      {/* Callout */}
      <motion.g {...popIn(0.9)}>
        <rect x="458" y="184" width="138" height="44" fill="#08090F" opacity=".9" rx="4" />
        <rect x="458" y="184" width="138" height="44" fill="none" stroke="#FF6B6B" strokeWidth=".8" rx="4" opacity=".7" />
        <text x="466" y="200" fill="#FF6B6B" fontFamily="'Courier New',monospace" fontSize="9">both persons =</text>
        <text x="466" y="215" fill="#FF6B6B" fontFamily="'Courier New',monospace" fontSize="9">same color!</text>
        <text x="466" y="225" fill="#3E4260" fontFamily="'Courier New',monospace" fontSize="8">(class, not instance)</text>
      </motion.g>
      {/* Legend */}
      <rect x="0" y="252" width="600" height="28" fill="#08090F" opacity=".88" />
      {legend.map((l, i) => {
        const cx = 11 + i * 53;
        return (
          <g key={l.label}>
            <circle cx={cx} cy="266" r="5" fill={l.color} />
            <text x={cx + 9} y="270" fill="#D0D4EE" fontFamily="'Courier New',monospace" fontSize="9">{l.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function InstanceSegOverlay() {
  const instances: { el: JSX.Element; delay: number; op: number }[] = [
    { el: <rect x="368" y="14" width="230" height="215" fill="#4DCAFF" />, delay: 0, op: 0.60 },
    { el: <><ellipse cx="144" cy="152" rx="56" ry="52" fill="#A3FF4D" /><rect x="136" y="174" width="16" height="52" fill="#A3FF4D" /></>, delay: 0.15, op: 0.65 },
    { el: <ellipse cx="186" cy="218" rx="23" ry="34" fill="#FF4DC4" />, delay: 0.3, op: 0.70 },
    { el: <ellipse cx="431" cy="216" rx="21" ry="32" fill="#4DFF9B" />, delay: 0.45, op: 0.70 },
    { el: <rect x="22" y="230" width="178" height="58" fill="#FFD166" />, delay: 0.6, op: 0.65 },
    { el: <rect x="230" y="232" width="148" height="58" fill="#C77DFF" />, delay: 0.75, op: 0.65 },
  ];

  const labels: { x: number; y: number; text: string; color: string; delay: number }[] = [
    { x: 484, y: 110, text: "bldg #001", color: "#4DCAFF", delay: 0.15 },
    { x: 96, y: 148, text: "tree #001", color: "#A3FF4D", delay: 0.30 },
    { x: 158, y: 183, text: "pers #001", color: "#FF4DC4", delay: 0.45 },
    { x: 412, y: 180, text: "pers #002", color: "#4DFF9B", delay: 0.60 },
    { x: 24, y: 230, text: "car #001", color: "#FFD166", delay: 0.75 },
    { x: 232, y: 232, text: "car #002", color: "#C77DFF", delay: 0.90 },
  ];

  return (
    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 600 280" xmlns="http://www.w3.org/2000/svg">
      {instances.map((m, i) => (
        <motion.g key={i} initial={{ opacity: 0 }} animate={{ opacity: m.op }} transition={{ delay: m.delay, duration: 0.4, ease: "easeOut" }}>
          {m.el}
        </motion.g>
      ))}
      {labels.map((l) => (
        <motion.g key={l.text} {...popIn(l.delay)}>
          <rect x={l.x} y={l.y} width={l.text.length * 7 + 10} height="13" fill={l.color} rx="2" opacity=".95" />
          <text x={l.x + 4} y={l.y + 10} fill="#08090F" fontFamily="'Courier New',monospace" fontSize="9" fontWeight="bold">{l.text}</text>
        </motion.g>
      ))}
      {/* Instance count HUD */}
      <motion.g {...popIn(1.0)}>
        <rect x="4" y="4" width="140" height="76" fill="#08090F" opacity=".92" rx="4" />
        <rect x="4" y="4" width="140" height="76" fill="none" stroke="#1A1D2E" strokeWidth=".6" rx="4" />
        <text x="10" y="21" fill="#4E86FF" fontFamily="'Courier New',monospace" fontSize="9" letterSpacing=".1em">INSTANCES</text>
        <line x1="4" y1="27" x2="144" y2="27" stroke="#1A1D2E" strokeWidth=".6" />
        <circle cx="14" cy="40" r="4" fill="#FF4DC4" /><text x="22" y="44" fill="#D0D4EE" fontFamily="'Courier New',monospace" fontSize="10">person  × 2</text>
        <circle cx="14" cy="56" r="4" fill="#FFD166" /><text x="22" y="60" fill="#D0D4EE" fontFamily="'Courier New',monospace" fontSize="10">car     × 2</text>
        <circle cx="82" cy="40" r="4" fill="#4DCAFF" /><text x="90" y="44" fill="#D0D4EE" fontFamily="'Courier New',monospace" fontSize="10">bldg × 1</text>
        <circle cx="82" cy="56" r="4" fill="#A3FF4D" /><text x="90" y="60" fill="#D0D4EE" fontFamily="'Courier New',monospace" fontSize="10">tree × 1</text>
        <text x="10" y="76" fill="#3E4260" fontFamily="'Courier New',monospace" fontSize="8">total: 6 instances</text>
      </motion.g>
      {/* Callout */}
      <motion.g {...popIn(1.1)}>
        <rect x="456" y="184" width="140" height="52" fill="#08090F" opacity=".9" rx="4" />
        <rect x="456" y="184" width="140" height="52" fill="none" stroke="#C77DFF" strokeWidth=".8" rx="4" opacity=".7" />
        <text x="464" y="200" fill="#C77DFF" fontFamily="'Courier New',monospace" fontSize="9">each instance =</text>
        <text x="464" y="214" fill="#C77DFF" fontFamily="'Courier New',monospace" fontSize="9">unique color!</text>
        <text x="464" y="228" fill="#3E4260" fontFamily="'Courier New',monospace" fontSize="8">cf. semantic: class→color</text>
      </motion.g>
    </svg>
  );
}

/* ══════ INFO PANELS ══════ */

const infoPanels: Record<Mode, { left: { title: string; body: string; formula?: string }; right: { title: string; body: string; formula?: string; chips?: { color: string; label: string }[] } }> = {
  clf: {
    left: {
      title: "Task Overview",
      body: "Given image x ∈ ℝ^(H×W×C), predict a single class label y ∈ {1,…,K}. The model learns a mapping f(x) → y from a labeled dataset D = {(xᵢ, yᵢ)}.",
      formula: "P(y=k|x) = exp(wₖᵀx + bₖ)\n           ─────────────────────\n           Σⱼ exp(wⱼᵀx + bⱼ)",
    },
    right: {
      title: "Training Objective",
      body: "Minimize cross-entropy loss between the softmax predictions ŷ and the one-hot ground truth y.",
      formula: "L = -Σᵢ Σₖ  yᵢₖ · log(ŷᵢₖ)\n\nMetrics: Top-1 accuracy, Top-5 accuracy",
      chips: [
        { color: "#4E86FF", label: "softmax output" },
        { color: "#FF6B6B", label: "cross-entropy loss" },
        { color: "#4DD07A", label: "top-5 accuracy" },
      ],
    },
  },
  det: {
    left: {
      title: "Task Overview",
      body: "For each object, predict class c AND bounding box b = (bₓ, bᵧ, b_w, b_h). Output is a set of tuples {(cᵢ, bᵢ, sᵢ)} where s is confidence.",
      formula: "IoU(A,B) = |A∩B| / |A∪B|\n\nNMS: suppress boxes with IoU > threshold",
    },
    right: {
      title: "Architectures",
      body: "Two-stage detectors (Faster R-CNN) use a Region Proposal Network first. One-stage (YOLO, RetinaNet) detect in a single pass — faster but historically less accurate.",
      formula: "L = L_cls + λ · L_reg (Smooth L1)\nFocal: FL(pₜ) = -α(1-pₜ)^γ log(pₜ)",
      chips: [
        { color: "#FF6B6B", label: "person" },
        { color: "#FFA940", label: "car" },
        { color: "#FFB347", label: "building" },
        { color: "#4DD07A", label: "tree" },
      ],
    },
  },
  sem: {
    left: {
      title: "Task Overview",
      body: "Assign a class label to every pixel. All instances of the same class receive the same color — two cars are indistinguishable. Critical for scene parsing in autonomous driving.",
      formula: "L = -1/HW · Σᵢⱼ Σc yᵢⱼc log(ŷᵢⱼc)\n\nMetric: mean IoU (mIoU) over all classes",
    },
    right: {
      title: "Class Legend",
      body: "Cityscapes-style class palette. Colors are assigned per class, not per instance.",
      formula: "FCN → U-Net → DeepLab (ASPP + atrous conv)",
      chips: [
        { color: "#5BB8F5", label: "sky" }, { color: "#FFB347", label: "building" },
        { color: "#4DD07A", label: "vegetation" }, { color: "#FF6B6B", label: "person" },
        { color: "#FFA940", label: "car" }, { color: "#8B96C8", label: "road" },
        { color: "#BABFDB", label: "sidewalk" }, { color: "#66BB6A", label: "terrain" },
      ],
    },
  },
  inst: {
    left: {
      title: "Task Overview",
      body: "Separates individual instances: car#001 ≠ car#002, even though both are \"car.\" Adds a mask branch to detection — each detected RoI gets a per-pixel binary mask.",
      formula: "L = L_cls + L_box + L_mask\n\nL_mask = binary CE per pixel (k-th class only)",
    },
    right: {
      title: "Instance Palette",
      body: "Colors assigned per-instance, not per-class. Same class, different color = different individual object.",
      formula: "Mask R-CNN: RoI Align → per-instance FCN\nPanoptic = semantic (stuff) + instance (things)",
      chips: [
        { color: "#FF4DC4", label: "person #001" }, { color: "#4DFF9B", label: "person #002" },
        { color: "#FFD166", label: "car #001" }, { color: "#C77DFF", label: "car #002" },
        { color: "#4DCAFF", label: "bldg #001" }, { color: "#A3FF4D", label: "tree #001" },
      ],
    },
  },
};

function InfoPanel({ mode }: { mode: Mode }) {
  const data = infoPanels[mode];
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-3">
      {[data.left, data.right].map((card: any) => (
        <div key={card.title} className="rounded-lg border border-border bg-card p-3">
          <p className="text-[8px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">{card.title}</p>
          <p className="text-[11px] text-muted-foreground leading-relaxed">{card.body}</p>
          {card.formula && (
            <pre className="mt-2 rounded bg-muted/40 border border-border border-l-2 border-l-primary p-2 text-[10px] font-mono text-primary/80 leading-relaxed whitespace-pre">
              {card.formula}
            </pre>
          )}
          {card.chips && (
            <div className="flex flex-wrap gap-1.5 mt-2">
              {card.chips.map((c) => (
                <span key={c.label} className="inline-flex items-center gap-1 text-[9px] text-muted-foreground bg-muted/30 border border-border rounded px-1.5 py-0.5">
                  <span className="w-[7px] h-[7px] rounded-sm shrink-0" style={{ backgroundColor: c.color }} />
                  {c.label}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/* ══════ MAIN EXPORTED COMPONENT ══════ */

const OVERLAYS: Record<Mode, () => JSX.Element> = {
  clf: ClassificationOverlay,
  det: DetectionOverlay,
  sem: SemanticSegOverlay,
  inst: InstanceSegOverlay,
};

export default function SemanticSceneVisualizer() {
  const [mode, setMode] = useState<Mode>("clf");
  const [animKey, setAnimKey] = useState(0);

  const switchMode = useCallback((m: Mode) => {
    setMode(m);
    setAnimKey((k) => k + 1);
  }, []);

  const OverlayComponent = OVERLAYS[mode];

  return (
    <div className="space-y-0">
      {/* Tabs */}
      <div className="flex border border-border border-b-0 rounded-t-lg overflow-hidden">
        {TAB_LABELS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => switchMode(tab.id)}
            className={`flex-1 py-2.5 px-1 text-center font-mono text-[9px] uppercase tracking-wider leading-snug transition-colors relative border-r border-border last:border-r-0 ${
              mode === tab.id
                ? "bg-card/80 text-primary"
                : "bg-muted/20 text-muted-foreground hover:bg-muted/40 hover:text-foreground"
            }`}
          >
            <span className="block text-[7px] text-muted-foreground/50 mb-0.5">{tab.num}</span>
            {tab.label}
            {mode === tab.id && (
              <motion.div layoutId="sem-tab-indicator" className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
            )}
          </button>
        ))}
      </div>

      {/* Scene */}
      <div className="relative border border-border rounded-b-lg overflow-hidden bg-[#0A0C15]">
        <BaseScene />
        <AnimatePresence mode="wait">
          <motion.div
            key={`${mode}-${animKey}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="absolute inset-0"
          >
            <OverlayComponent />
          </motion.div>
        </AnimatePresence>
        {/* Output tag */}
        <span className="absolute bottom-2 right-3 text-[8px] font-mono text-muted-foreground/40 uppercase tracking-wider pointer-events-none">
          {TAGS[mode]}
        </span>
      </div>

      {/* Info panels */}
      <InfoPanel mode={mode} />

      {/* Task flow mini-nav */}
      <div className="flex items-center gap-1.5 mt-3 flex-wrap">
        {TAB_LABELS.map((tab, i) => (
          <div key={tab.id} className="flex items-center gap-1.5">
            <button
              onClick={() => switchMode(tab.id)}
              className={`text-[9px] font-mono uppercase tracking-wider px-2 py-1 rounded border transition-colors ${
                mode === tab.id
                  ? "text-primary border-primary"
                  : "text-muted-foreground border-border hover:text-foreground hover:border-muted-foreground"
              }`}
            >
              {tab.label}
            </button>
            {i < TAB_LABELS.length - 1 && (
              <span className="text-muted-foreground/30 text-[10px]">→</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
