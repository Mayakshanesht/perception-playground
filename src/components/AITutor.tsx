import { useState, useRef, useCallback, useEffect } from "react";
import { Volume2, Pause, RotateCcw, VolumeX, Loader2 } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { motion, AnimatePresence } from "framer-motion";

interface AITutorProps {
  conceptTitle: string;
  conceptContent: string;
  moduleName: string;
}

type TutorState = "idle" | "loading" | "speaking" | "paused" | "muted" | "done";

export default function AITutor({ conceptTitle, conceptContent, moduleName }: AITutorProps) {
  const [state, setState] = useState<TutorState>("idle");
  const [explanation, setExplanation] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isMuted, setIsMuted] = useState(false);
  const [progress, setProgress] = useState(0);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const explanationCache = useRef<Map<string, string>>(new Map());

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopSpeaking();
      if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
    };
  }, []);

  const fetchExplanation = useCallback(async (): Promise<string | null> => {
    const cacheKey = `${moduleName}::${conceptTitle}`;
    if (explanationCache.current.has(cacheKey)) {
      return explanationCache.current.get(cacheKey)!;
    }

    try {
      const { data, error: fnError } = await supabase.functions.invoke("ai-tutor", {
        body: { conceptTitle, conceptContent, moduleName },
      });

      if (fnError) throw new Error(fnError.message);
      if (data?.error) throw new Error(data.error);

      const text = data?.explanation || "";
      explanationCache.current.set(cacheKey, text);
      return text;
    } catch (err: any) {
      console.error("AI Tutor fetch error:", err);
      setError(err.message || "Failed to generate explanation");
      return null;
    }
  }, [conceptTitle, conceptContent, moduleName]);

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis.cancel();
    utteranceRef.current = null;
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  }, []);

  const speak = useCallback((text: string) => {
    stopSpeaking();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.volume = isMuted ? 0 : 1;

    // Try to pick a good voice
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(
      (v) =>
        v.lang.startsWith("en") &&
        (v.name.toLowerCase().includes("google") ||
          v.name.toLowerCase().includes("samantha") ||
          v.name.toLowerCase().includes("daniel") ||
          v.name.toLowerCase().includes("natural"))
    );
    if (preferred) utterance.voice = preferred;

    // Track progress
    const words = text.split(/\s+/).length;
    const estimatedDuration = (words / 2.5) * 1000; // ~150 wpm
    const startTime = Date.now();
    progressIntervalRef.current = setInterval(() => {
      const elapsed = Date.now() - startTime;
      setProgress(Math.min(1, elapsed / estimatedDuration));
    }, 100);

    utterance.onend = () => {
      setState("done");
      setProgress(1);
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
    };

    utterance.onerror = (e) => {
      if (e.error !== "canceled") {
        setState("idle");
        setError("Speech synthesis failed");
      }
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
    };

    utteranceRef.current = utterance;
    setState(isMuted ? "muted" : "speaking");
    window.speechSynthesis.speak(utterance);
  }, [isMuted, stopSpeaking]);

  const handlePlay = useCallback(async () => {
    setError(null);

    if (explanation) {
      speak(explanation);
      return;
    }

    setState("loading");
    const text = await fetchExplanation();
    if (text) {
      setExplanation(text);
      speak(text);
    } else {
      setState("idle");
    }
  }, [explanation, speak, fetchExplanation]);

  const handlePause = useCallback(() => {
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.pause();
      setState("paused");
    }
  }, []);

  const handleResume = useCallback(() => {
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
      setState(isMuted ? "muted" : "speaking");
    }
  }, [isMuted]);

  const handleReplay = useCallback(() => {
    if (explanation) {
      setProgress(0);
      speak(explanation);
    }
  }, [explanation, speak]);

  const handleMuteToggle = useCallback(() => {
    const next = !isMuted;
    setIsMuted(next);
    // Update current utterance volume
    if (utteranceRef.current && window.speechSynthesis.speaking) {
      // SpeechSynthesis doesn't support live volume change, so we restart
      stopSpeaking();
      if (explanation) {
        const utterance = new SpeechSynthesisUtterance(explanation);
        utterance.volume = next ? 0 : 1;
        utterance.rate = 0.95;
        utteranceRef.current = utterance;
        utterance.onend = () => setState("done");
        setState(next ? "muted" : "speaking");
        window.speechSynthesis.speak(utterance);
      }
    }
  }, [isMuted, explanation, stopSpeaking]);

  const handleStop = useCallback(() => {
    stopSpeaking();
    setState("idle");
    setProgress(0);
  }, [stopSpeaking]);

  // Hover trigger (800ms dwell)
  const handleMouseEnter = useCallback(() => {
    if (state !== "idle" && state !== "done") return;
    hoverTimerRef.current = setTimeout(() => {
      handlePlay();
    }, 800);
  }, [state, handlePlay]);

  const handleMouseLeave = useCallback(() => {
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
  }, []);

  const isActive = state !== "idle" && state !== "done";

  return (
    <div className="inline-flex items-center gap-1.5 ml-2">
      {/* Main trigger / Listen button */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onClick={() => {
          if (state === "idle" || state === "done") handlePlay();
          else if (state === "speaking" || state === "muted") handlePause();
          else if (state === "paused") handleResume();
        }}
        disabled={state === "loading"}
        className={`
          inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium
          transition-all duration-200 border
          ${isActive
            ? "bg-primary/15 border-primary/30 text-primary"
            : "bg-muted/50 border-border hover:bg-primary/10 hover:border-primary/20 text-muted-foreground hover:text-primary"
          }
          disabled:opacity-50 disabled:cursor-wait
        `}
        title={state === "idle" || state === "done" ? "Listen to AI explanation" : state === "paused" ? "Resume" : "Pause"}
      >
        {state === "loading" ? (
          <Loader2 className="h-3 w-3 animate-spin" />
        ) : state === "speaking" || state === "muted" ? (
          <Pause className="h-3 w-3" />
        ) : (
          <Volume2 className="h-3 w-3" />
        )}
        <span>
          {state === "loading"
            ? "Generating..."
            : state === "speaking" || state === "muted"
            ? "Pause"
            : state === "paused"
            ? "Resume"
            : "Listen"}
        </span>
      </motion.button>

      {/* Active controls */}
      <AnimatePresence>
        {isActive && (
          <motion.div
            initial={{ opacity: 0, width: 0 }}
            animate={{ opacity: 1, width: "auto" }}
            exit={{ opacity: 0, width: 0 }}
            className="flex items-center gap-1 overflow-hidden"
          >
            {/* Progress bar */}
            <div className="w-12 h-1 rounded-full bg-border overflow-hidden">
              <motion.div
                className="h-full bg-primary rounded-full"
                style={{ width: `${progress * 100}%` }}
                transition={{ duration: 0.1 }}
              />
            </div>

            {/* Replay */}
            <button
              onClick={handleReplay}
              className="p-1 rounded-full hover:bg-muted/50 text-muted-foreground hover:text-primary transition-colors"
              title="Replay"
            >
              <RotateCcw className="h-3 w-3" />
            </button>

            {/* Mute */}
            <button
              onClick={handleMuteToggle}
              className={`p-1 rounded-full hover:bg-muted/50 transition-colors ${isMuted ? "text-destructive" : "text-muted-foreground hover:text-primary"}`}
              title={isMuted ? "Unmute" : "Mute"}
            >
              <VolumeX className="h-3 w-3" />
            </button>

            {/* Stop */}
            <button
              onClick={handleStop}
              className="p-1 rounded-full hover:bg-muted/50 text-muted-foreground hover:text-foreground transition-colors text-[10px] font-mono"
              title="Stop"
            >
              ✕
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error */}
      {error && (
        <span className="text-[10px] text-destructive ml-1">{error}</span>
      )}
    </div>
  );
}
