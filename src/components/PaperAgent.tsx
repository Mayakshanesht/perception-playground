import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Bot, Send, X, Loader2, BookOpen } from "lucide-react";
import ReactMarkdown from "react-markdown";

type Msg = { role: "user" | "assistant"; content: string };

interface PaperAgentProps {
  paperTitle: string;
  onClose: () => void;
}

export default function PaperAgent({ paperTitle, onClose }: PaperAgentProps) {
  const [messages, setMessages] = useState<Msg[]>([
    { role: "user", content: `Explain the ${paperTitle} paper in detail. Cover: the problem it solves, core idea, architecture/method, key equations, experimental results, limitations, and its impact on later research.` }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [started, setStarted] = useState(false);

  const sendMessages = async (msgs: Msg[]) => {
    setLoading(true);
    let assistantSoFar = "";

    try {
      const resp = await fetch(
        `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/cv-assistant`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
          },
          body: JSON.stringify({ messages: msgs, mode: "paper" }),
        }
      );

      if (!resp.ok || !resp.body) throw new Error("Failed");

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let textBuffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        textBuffer += decoder.decode(value, { stream: true });
        let newlineIndex: number;
        while ((newlineIndex = textBuffer.indexOf("\n")) !== -1) {
          let line = textBuffer.slice(0, newlineIndex);
          textBuffer = textBuffer.slice(newlineIndex + 1);
          if (line.endsWith("\r")) line = line.slice(0, -1);
          if (!line.startsWith("data: ") || line.trim() === "" || line.startsWith(":")) continue;
          const jsonStr = line.slice(6).trim();
          if (jsonStr === "[DONE]") break;
          try {
            const parsed = JSON.parse(jsonStr);
            const content = parsed.choices?.[0]?.delta?.content;
            if (content) {
              assistantSoFar += content;
              setMessages((prev) => {
                const last = prev[prev.length - 1];
                if (last?.role === "assistant") {
                  return prev.map((m, i) => i === prev.length - 1 ? { ...m, content: assistantSoFar } : m);
                }
                return [...prev, { role: "assistant", content: assistantSoFar }];
              });
            }
          } catch {
            textBuffer = line + "\n" + textBuffer;
            break;
          }
        }
      }
    } catch (e: any) {
      setMessages((prev) => [...prev, { role: "assistant", content: `⚠️ ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const startExploration = () => {
    setStarted(true);
    sendMessages(messages);
  };

  const askFollowUp = () => {
    if (!input.trim() || loading) return;
    const newMsgs = [...messages, { role: "user" as const, content: input.trim() }];
    setMessages(newMsgs);
    setInput("");
    sendMessages(newMsgs);
  };

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      exit={{ opacity: 0, height: 0 }}
      className="rounded-xl border border-primary/30 bg-card overflow-hidden mt-3"
    >
      <div className="px-4 py-3 border-b border-border flex items-center justify-between bg-primary/5">
        <div className="flex items-center gap-2">
          <BookOpen className="h-4 w-4 text-primary" />
          <h4 className="text-sm font-semibold text-foreground">Paper Explorer: {paperTitle}</h4>
        </div>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="p-4 max-h-96 overflow-y-auto scrollbar-thin space-y-3">
        {!started ? (
          <div className="text-center py-4">
            <p className="text-sm text-muted-foreground mb-3">
              Get a structured AI-powered explanation of this paper
            </p>
            <button
              onClick={startExploration}
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
            >
              Explore Paper
            </button>
          </div>
        ) : (
          <>
            {messages.filter(m => m.role === "assistant").map((msg, i) => (
              <div key={i} className="prose prose-sm prose-invert max-w-none text-xs leading-relaxed [&_p]:mb-2 [&_code]:text-[10px]">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
            ))}
            {loading && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                Analyzing paper...
              </div>
            )}
            {!loading && started && (
              <form onSubmit={(e) => { e.preventDefault(); askFollowUp(); }} className="flex gap-2 pt-2 border-t border-border">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask a follow-up question..."
                  className="flex-1 px-3 py-2 rounded-lg bg-muted border border-border text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                />
                <button
                  type="submit"
                  disabled={!input.trim()}
                  className="h-8 w-8 rounded-lg bg-primary text-primary-foreground flex items-center justify-center disabled:opacity-50"
                >
                  <Send className="h-3 w-3" />
                </button>
              </form>
            )}
          </>
        )}
      </div>
    </motion.div>
  );
}
