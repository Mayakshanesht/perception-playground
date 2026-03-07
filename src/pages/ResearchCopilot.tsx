import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate, Link } from "react-router-dom";
import {
  Search, FileText, GitBranch, Cpu, BookOpen, Download, Plus,
  ArrowLeft, Loader2, Sparkles, FlaskConical, Database, Box,
  ChevronRight, ExternalLink, Trash2, FolderOpen
} from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import ReactMarkdown from "react-markdown";

type Tab = "research" | "projects" | "experiments" | "models";

interface Project {
  id: string;
  name: string;
  research_question: string;
  papers: any[];
  architecture: any;
  notebooks: any[];
  status: string;
  created_at: string;
  updated_at: string;
}

export default function ResearchCopilot() {
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [activeTab, setActiveTab] = useState<Tab>("research");
  const [query, setQuery] = useState("");
  const [analysisResult, setAnalysisResult] = useState<string>("");
  const [analyzing, setAnalyzing] = useState(false);
  const [projects, setProjects] = useState<Project[]>([]);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [showNewProject, setShowNewProject] = useState(false);

  useEffect(() => {
    if (!authLoading && !user) {
      navigate("/sign-in");
    }
  }, [user, authLoading, navigate]);

  useEffect(() => {
    if (user) fetchProjects();
  }, [user]);

  const fetchProjects = async () => {
    setLoadingProjects(true);
    const { data, error } = await supabase
      .from("research_projects")
      .select("*")
      .order("updated_at", { ascending: false });
    if (!error && data) setProjects(data as Project[]);
    setLoadingProjects(false);
  };

  const createProject = async () => {
    if (!newProjectName.trim()) return;
    const { error } = await supabase.from("research_projects").insert({
      user_id: user!.id,
      name: newProjectName.trim(),
      research_question: query || "",
    });
    if (error) {
      toast({ title: "Error creating project", description: error.message, variant: "destructive" });
    } else {
      setNewProjectName("");
      setShowNewProject(false);
      fetchProjects();
      toast({ title: "Project created" });
    }
  };

  const deleteProject = async (id: string) => {
    await supabase.from("research_projects").delete().eq("id", id);
    fetchProjects();
  };

  const analyzeResearch = async () => {
    if (!query.trim()) return;
    setAnalyzing(true);
    setAnalysisResult("");

    try {
      const endpoint = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/research-copilot`;
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
        },
        body: JSON.stringify({ query: query.trim(), action: "analyze" }),
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.error || `Request failed (${resp.status})`);
      }

      if (!resp.body) throw new Error("No response body");

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let newlineIdx: number;
        while ((newlineIdx = buffer.indexOf("\n")) !== -1) {
          let line = buffer.slice(0, newlineIdx);
          buffer = buffer.slice(newlineIdx + 1);
          if (line.endsWith("\r")) line = line.slice(0, -1);
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.slice(6).trim();
          if (jsonStr === "[DONE]") break;
          try {
            const parsed = JSON.parse(jsonStr);
            const content = parsed.choices?.[0]?.delta?.content;
            if (content) {
              fullText += content;
              setAnalysisResult(fullText);
            }
          } catch { /* partial json, continue */ }
        }
      }
    } catch (err: any) {
      toast({ title: "Analysis failed", description: err.message, variant: "destructive" });
    } finally {
      setAnalyzing(false);
    }
  };

  const generateNotebook = async () => {
    if (!query.trim()) return;
    setAnalyzing(true);
    setAnalysisResult("");

    try {
      const endpoint = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/research-copilot`;
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
        },
        body: JSON.stringify({ query: query.trim(), action: "notebook" }),
      });

      if (!resp.ok) throw new Error("Request failed");
      if (!resp.body) throw new Error("No body");

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let idx: number;
        while ((idx = buffer.indexOf("\n")) !== -1) {
          let line = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 1);
          if (line.endsWith("\r")) line = line.slice(0, -1);
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.slice(6).trim();
          if (jsonStr === "[DONE]") break;
          try {
            const p = JSON.parse(jsonStr);
            const c = p.choices?.[0]?.delta?.content;
            if (c) { fullText += c; setAnalysisResult(fullText); }
          } catch {}
        }
      }
    } catch (err: any) {
      toast({ title: "Generation failed", description: err.message, variant: "destructive" });
    } finally {
      setAnalyzing(false);
    }
  };

  if (authLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
      </div>
    );
  }

  if (!user) return null;

  const tabs: { id: Tab; label: string; icon: any }[] = [
    { id: "research", label: "Research", icon: Search },
    { id: "projects", label: "Projects", icon: FolderOpen },
    { id: "experiments", label: "Experiments", icon: FlaskConical },
    { id: "models", label: "Models", icon: Box },
  ];

  return (
    <div className="p-6 md:p-8 max-w-6xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6">
        <ArrowLeft className="h-3 w-3" /> Back to Dashboard
      </Link>

      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
            <Sparkles className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-foreground tracking-tight">Cloudbee Research Copilot</h1>
            <p className="text-xs text-muted-foreground">by Cloudbee Robotics — Analyze papers, design architectures, generate notebooks</p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-6 p-1 rounded-xl bg-muted/50 border border-border w-fit">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all ${
              activeTab === t.id
                ? "bg-primary text-primary-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground hover:bg-muted"
            }`}
          >
            <t.icon className="h-3.5 w-3.5" />
            {t.label}
          </button>
        ))}
      </div>

      {/* Research Tab */}
      {activeTab === "research" && (
        <div className="space-y-6">
          {/* Query Input */}
          <div className="rounded-xl border border-border bg-card p-6">
            <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
              <Search className="h-4 w-4 text-primary" />
              Research Question
            </h3>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. Build a robust optical flow model for autonomous driving"
              rows={3}
              className="w-full rounded-lg border border-border bg-muted/30 px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 resize-none"
            />
            <div className="flex gap-3 mt-4">
              <button
                onClick={analyzeResearch}
                disabled={analyzing || !query.trim()}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
              >
                {analyzing ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileText className="h-4 w-4" />}
                Analyze Research
              </button>
              <button
                onClick={generateNotebook}
                disabled={analyzing || !query.trim()}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg border border-border bg-card text-foreground text-sm font-medium hover:bg-muted/50 transition-colors disabled:opacity-50"
              >
                {analyzing ? <Loader2 className="h-4 w-4 animate-spin" /> : <BookOpen className="h-4 w-4" />}
                Generate Notebook
              </button>
              <button
                onClick={() => {
                  if (analysisResult) {
                    const blob = new Blob([analysisResult], { type: "text/markdown" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = "research-analysis.md";
                    a.click();
                    URL.revokeObjectURL(url);
                  }
                }}
                disabled={!analysisResult}
                className="flex items-center gap-2 px-4 py-2.5 rounded-lg border border-border bg-card text-foreground text-sm font-medium hover:bg-muted/50 transition-colors disabled:opacity-50"
              >
                <Download className="h-4 w-4" />
                Export
              </button>
            </div>
          </div>

          {/* Results */}
          {(analysisResult || analyzing) && (
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="text-sm font-semibold text-foreground mb-4 flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-accent" />
                {analyzing ? "Analyzing..." : "Analysis Results"}
              </h3>
              <div className="prose prose-sm dark:prose-invert max-w-none text-sm">
                <ReactMarkdown>{analysisResult || "Thinking..."}</ReactMarkdown>
              </div>
            </div>
          )}

          {/* Quick Actions */}
          {!analysisResult && !analyzing && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                { icon: FileText, title: "Paper Analysis", desc: "Analyze papers related to your research topic", action: () => { setQuery("Analyze recent papers on monocular depth estimation"); } },
                { icon: GitBranch, title: "Architecture Design", desc: "Generate model architecture proposals", action: () => { setQuery("Design an architecture for real-time object detection"); } },
                { icon: Cpu, title: "Training Pipeline", desc: "Generate training notebooks", action: () => { setQuery("Build a training pipeline for semantic segmentation on Cityscapes"); } },
              ].map((card) => (
                <button
                  key={card.title}
                  onClick={card.action}
                  className="rounded-xl border border-border bg-card/80 p-5 text-left hover:border-primary/30 hover:bg-muted/30 transition-all group"
                >
                  <card.icon className="h-5 w-5 text-primary mb-3" />
                  <h4 className="text-sm font-semibold text-foreground mb-1">{card.title}</h4>
                  <p className="text-xs text-muted-foreground">{card.desc}</p>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Projects Tab */}
      {activeTab === "projects" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-foreground">My Projects</h3>
            <button
              onClick={() => setShowNewProject(!showNewProject)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90"
            >
              <Plus className="h-3 w-3" /> New Project
            </button>
          </div>

          {showNewProject && (
            <div className="rounded-xl border border-primary/30 bg-card p-4 flex gap-3">
              <input
                value={newProjectName}
                onChange={(e) => setNewProjectName(e.target.value)}
                placeholder="Project name"
                className="flex-1 rounded-lg border border-border bg-muted/30 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                onKeyDown={(e) => e.key === "Enter" && createProject()}
              />
              <button onClick={createProject} className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-medium">Create</button>
            </div>
          )}

          {loadingProjects ? (
            <div className="flex justify-center py-12"><Loader2 className="h-5 w-5 animate-spin text-primary" /></div>
          ) : projects.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground text-sm">No projects yet. Create your first research project!</div>
          ) : (
            <div className="grid gap-3">
              {projects.map((p) => (
                <div key={p.id} className="rounded-xl border border-border bg-card p-4 flex items-center justify-between group hover:border-primary/20 transition-colors">
                  <div>
                    <h4 className="text-sm font-semibold text-foreground">{p.name}</h4>
                    {p.research_question && <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">{p.research_question}</p>}
                    <p className="text-[10px] text-muted-foreground/60 font-mono mt-1">
                      {new Date(p.updated_at).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button onClick={() => deleteProject(p.id)} className="p-1.5 rounded-lg hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors">
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Experiments Tab */}
      {activeTab === "experiments" && (
        <div className="text-center py-16">
          <FlaskConical className="h-10 w-10 text-muted-foreground/30 mx-auto mb-4" />
          <h3 className="text-sm font-semibold text-foreground mb-1">Experiments</h3>
          <p className="text-xs text-muted-foreground max-w-md mx-auto">
            Run research analyses and save experiments from the Research tab. Your hypotheses, architectures, and notebooks will appear here.
          </p>
        </div>
      )}

      {/* Models Tab */}
      {activeTab === "models" && (
        <div className="text-center py-16">
          <Box className="h-10 w-10 text-muted-foreground/30 mx-auto mb-4" />
          <h3 className="text-sm font-semibold text-foreground mb-1">Saved Models</h3>
          <p className="text-xs text-muted-foreground max-w-md mx-auto">
            Register trained models with HuggingFace links, datasets, and metrics. Models registered from your experiments will appear here.
          </p>
        </div>
      )}
    </div>
  );
}
