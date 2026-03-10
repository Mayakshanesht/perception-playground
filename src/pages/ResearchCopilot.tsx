import { useState, useEffect, useCallback } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  Search, FileText, GitBranch, Download, Plus, ArrowLeft, Loader2,
  Sparkles, FlaskConical, Box, Trash2, FolderOpen, ChevronRight,
  ExternalLink, BookOpen, Database, Check, Copy, Globe, Clock,
  Share2, FileDown, BarChart3, Beaker, Eye, Calendar, TrendingUp,
  Network, AlertCircle
} from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import ResearchGraph from "@/components/ResearchGraph";
import ResearchTimeline from "@/components/ResearchTimeline";

type Tab = "projects" | "experiments" | "models" | "timeline" | "graph";

interface Paper {
  title: string;
  authors: string;
  year: number;
  problem: string;
  method: string;
  results: string;
  github: string | null;
  dataset: string;
}

interface Hypothesis {
  id: string;
  name: string;
  architecture: string;
  expected_accuracy: string;
  compute: string;
  dataset: string;
  reasoning: string;
}

interface Proposal {
  title: string;
  summary: string;
  pipeline: string[];
  key_insight: string;
}

interface AnalysisData {
  papers: Paper[];
  hypotheses: Hypothesis[];
  proposal: Proposal;
  datasets: { name: string; size: string; source: string; url: string }[];
}

interface Project {
  id: string;
  name: string;
  research_question: string;
  papers: any;
  architecture: any;
  notebooks: any;
  model_links: any;
  status: string;
  created_at: string;
  updated_at: string;
}

interface SavedModel {
  id: string;
  name: string;
  huggingface_link: string;
  dataset_used: string;
  metrics: any;
  project_id: string | null;
  created_at: string;
}

interface Experiment {
  id: string;
  title: string;
  hypothesis: string;
  architecture: any;
  metrics: any;
  status: string;
  notebook: string;
  project_id: string | null;
  created_at: string;
  updated_at: string;
}

type Stage = "query" | "analyzing" | "results" | "notebook-gen" | "notebook-ready";

export default function ResearchCopilot() {
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [activeTab, setActiveTab] = useState<Tab>("projects");
  const [projects, setProjects] = useState<Project[]>([]);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [showNewProject, setShowNewProject] = useState(false);

  const [activeProject, setActiveProject] = useState<Project | null>(null);
  const [stage, setStage] = useState<Stage>("query");
  const [query, setQuery] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [selectedHypothesis, setSelectedHypothesis] = useState<Hypothesis | null>(null);
  const [notebookJson, setNotebookJson] = useState("");
  const [generatingNotebook, setGeneratingNotebook] = useState(false);
  const [expandedPaper, setExpandedPaper] = useState<number | null>(null);

  const [models, setModels] = useState<SavedModel[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [showNewModel, setShowNewModel] = useState(false);
  const [newModel, setNewModel] = useState({ name: "", huggingface_link: "", dataset_used: "", project_id: "" });

  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loadingExperiments, setLoadingExperiments] = useState(false);

  useEffect(() => {
    if (!authLoading && !user) navigate("/sign-in");
  }, [user, authLoading, navigate]);

  useEffect(() => {
    if (user) { fetchProjects(); fetchModels(); fetchExperiments(); }
  }, [user]);

  const fetchProjects = async () => {
    setLoadingProjects(true);
    const { data } = await supabase.from("research_projects").select("*").order("updated_at", { ascending: false });
    if (data) setProjects(data as Project[]);
    setLoadingProjects(false);
  };

  const fetchModels = async () => {
    setLoadingModels(true);
    const { data } = await supabase.from("saved_models").select("*").order("created_at", { ascending: false });
    if (data) setModels(data as SavedModel[]);
    setLoadingModels(false);
  };

  const fetchExperiments = async () => {
    setLoadingExperiments(true);
    const { data } = await supabase.from("experiments").select("*").order("created_at", { ascending: false });
    if (data) setExperiments(data as Experiment[]);
    setLoadingExperiments(false);
  };

  const createProject = async () => {
    if (!newProjectName.trim()) return;
    const { error } = await supabase.from("research_projects").insert({ user_id: user!.id, name: newProjectName.trim() });
    if (error) { toast({ title: "Error", description: error.message, variant: "destructive" }); return; }
    setNewProjectName(""); setShowNewProject(false); fetchProjects();
    toast({ title: "Project created" });
  };

  const deleteProject = async (id: string) => {
    await supabase.from("research_projects").delete().eq("id", id);
    if (activeProject?.id === id) { setActiveProject(null); setStage("query"); setAnalysis(null); }
    fetchProjects();
  };

  const openProject = (p: Project) => {
    setActiveProject(p);
    setQuery(p.research_question || "");
    setStage("query");
    setAnalysis(null);
    setSelectedHypothesis(null);
    setNotebookJson("");
    setExpandedPaper(null);
  };

  const callCopilot = async (action: string, extraBody: any = {}) => {
    const endpoint = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/research-copilot`;
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}` },
      body: JSON.stringify({ query: query.trim(), action, ...extraBody }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `Request failed (${resp.status})`);
    }
    return resp.json();
  };

  const analyzeResearch = async () => {
    if (!query.trim() || !activeProject) return;
    setAnalyzing(true); setStage("analyzing"); setAnalysis(null);
    try {
      const { result } = await callCopilot("analyze");
      const parsed: AnalysisData = JSON.parse(result);
      setAnalysis(parsed);
      setStage("results");
      await supabase.from("research_projects").update({
        research_question: query.trim(),
        papers: JSON.parse(JSON.stringify(parsed.papers)),
        architecture: JSON.parse(JSON.stringify(parsed.proposal)),
        updated_at: new Date().toISOString(),
      }).eq("id", activeProject.id);
    } catch (err: any) {
      toast({ title: "Analysis failed", description: err.message, variant: "destructive" });
      setStage("query");
    } finally { setAnalyzing(false); }
  };

  const generateNotebook = async () => {
    if (!selectedHypothesis || !activeProject) return;
    setGeneratingNotebook(true); setStage("notebook-gen");
    try {
      const { result } = await callCopilot("notebook", { hypothesis: selectedHypothesis });
      setNotebookJson(result);
      setStage("notebook-ready");

      // Save experiment
      await supabase.from("experiments").insert({
        user_id: user!.id,
        project_id: activeProject.id,
        title: `${selectedHypothesis.name} Experiment`,
        hypothesis: selectedHypothesis.name,
        architecture: { ...selectedHypothesis },
        notebook: result,
        status: "notebook_generated",
      });
      fetchExperiments();

      await supabase.from("research_projects").update({
        notebooks: [{ hypothesis: selectedHypothesis.name, generated_at: new Date().toISOString() }],
        updated_at: new Date().toISOString(),
      }).eq("id", activeProject.id);
    } catch (err: any) {
      toast({ title: "Notebook generation failed", description: err.message, variant: "destructive" });
      setStage("results");
    } finally { setGeneratingNotebook(false); }
  };

  const downloadNotebook = () => {
    if (!notebookJson) return;
    const filename = `${selectedHypothesis?.name?.replace(/\s+/g, "_") || "notebook"}.ipynb`;
    let nbContent: string;
    try {
      const parsed = JSON.parse(notebookJson);
      // Validate it looks like a notebook; if not, wrap raw code into one
      if (parsed.nbformat && parsed.cells) {
        nbContent = JSON.stringify(parsed, null, 2);
      } else {
        nbContent = JSON.stringify(wrapAsNotebook(notebookJson), null, 2);
      }
    } catch {
      // LLM returned raw Python or malformed JSON — wrap it into a valid .ipynb
      nbContent = JSON.stringify(wrapAsNotebook(notebookJson), null, 2);
    }
    const blob = new Blob([nbContent], { type: "application/x-ipynb+json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = filename;
    a.click(); URL.revokeObjectURL(url);
  };

  const wrapAsNotebook = (code: string) => ({
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
      language_info: { name: "python", version: "3.10.0" },
      colab: { provenance: [] },
    },
    cells: [
      { cell_type: "markdown", metadata: {}, source: [`# ${selectedHypothesis?.name || "Experiment"}\n`, `Generated by Cloudbee Research Copilot\n`] },
      { cell_type: "code", metadata: {}, source: code.split("\n").map((l, i, a) => i < a.length - 1 ? l + "\n" : l), execution_count: null, outputs: [] },
    ],
  });

  const downloadReport = () => {
    if (!analysis || !activeProject) return;
    const md = generateReportMarkdown();
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url;
    a.download = `${activeProject.name.replace(/\s+/g, "_")}_report.md`;
    a.click(); URL.revokeObjectURL(url);
  };

  const generateReportMarkdown = () => {
    if (!analysis || !activeProject) return "";
    let md = `# Research Report: ${activeProject.name}\n\n`;
    md += `**Research Question:** ${query}\n\n`;
    md += `**Generated:** ${new Date().toLocaleDateString()}\n\n---\n\n`;

    md += `## Recommended Proposal\n\n`;
    md += `### ${analysis.proposal.title}\n\n`;
    md += `${analysis.proposal.summary}\n\n`;
    md += `**Pipeline:** ${analysis.proposal.pipeline.join(" → ")}\n\n`;
    md += `**Key Insight:** ${analysis.proposal.key_insight}\n\n---\n\n`;

    md += `## Papers Analyzed (${analysis.papers.length})\n\n`;
    analysis.papers.forEach((p, i) => {
      md += `### ${i + 1}. ${p.title}\n`;
      md += `- **Authors:** ${p.authors} (${p.year})\n`;
      md += `- **Problem:** ${p.problem}\n`;
      md += `- **Method:** ${p.method}\n`;
      md += `- **Results:** ${p.results}\n`;
      if (p.github) md += `- **GitHub:** ${p.github}\n`;
      if (p.dataset) md += `- **Dataset:** ${p.dataset}\n`;
      md += `\n`;
    });

    md += `---\n\n## Hypotheses\n\n`;
    analysis.hypotheses.forEach(h => {
      md += `### Hypothesis ${h.id}: ${h.name}\n`;
      md += `- **Architecture:** ${h.architecture}\n`;
      md += `- **Expected Accuracy:** ${h.expected_accuracy}\n`;
      md += `- **Compute:** ${h.compute}\n`;
      md += `- **Dataset:** ${h.dataset}\n`;
      md += `- **Reasoning:** ${h.reasoning}\n\n`;
    });

    if (analysis.datasets?.length) {
      md += `---\n\n## Recommended Datasets\n\n`;
      analysis.datasets.forEach(d => {
        md += `- **${d.name}** (${d.size}) — ${d.source}${d.url ? ` [Link](${d.url})` : ""}\n`;
      });
    }

    if (selectedHypothesis) {
      md += `\n---\n\n## Selected Hypothesis: ${selectedHypothesis.name}\n\n`;
      md += `Notebook generated for this hypothesis.\n`;
    }

    return md;
  };

  const saveModel = async () => {
    if (!newModel.name.trim() || !newModel.huggingface_link.trim()) return;
    const { error } = await supabase.from("saved_models").insert({
      user_id: user!.id,
      name: newModel.name.trim(),
      huggingface_link: newModel.huggingface_link.trim(),
      dataset_used: newModel.dataset_used.trim(),
      project_id: newModel.project_id || activeProject?.id || null,
    });
    if (error) { toast({ title: "Error", description: error.message, variant: "destructive" }); return; }
    setNewModel({ name: "", huggingface_link: "", dataset_used: "", project_id: "" }); setShowNewModel(false);
    fetchModels(); toast({ title: "Model registered" });
  };

  const deleteModel = async (id: string) => { await supabase.from("saved_models").delete().eq("id", id); fetchModels(); };
  const deleteExperiment = async (id: string) => { await supabase.from("experiments").delete().eq("id", id); fetchExperiments(); };

  if (authLoading) return <div className="flex items-center justify-center min-h-screen"><Loader2 className="h-6 w-6 animate-spin text-primary" /></div>;
  if (!user) return null;

  const tabs: { id: Tab; label: string; icon: any }[] = [
    { id: "projects", label: "Projects", icon: FolderOpen },
    { id: "experiments", label: "Experiments", icon: Beaker },
    { id: "models", label: "Models", icon: Box },
    { id: "timeline", label: "Timeline", icon: Clock },
    { id: "graph", label: "Research Graph", icon: Network },
  ];

  const projectModels = models.filter(m => activeProject && m.project_id === activeProject.id);
  const projectExperiments = experiments.filter(e => activeProject && e.project_id === activeProject.id);

  return (
    <div className="p-4 md:p-8 max-w-7xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-4">
        <ArrowLeft className="h-3 w-3" /> Dashboard
      </Link>

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
            <Sparkles className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground tracking-tight">Cloudbee Research Copilot</h1>
            <p className="text-[11px] text-muted-foreground">by Cloudbee Robotics — Structured AI Research Workflow</p>
          </div>
        </div>
        {/* Stats */}
        <div className="hidden md:flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1"><FolderOpen className="h-3 w-3" /> {projects.length} projects</span>
          <span className="flex items-center gap-1"><Beaker className="h-3 w-3" /> {experiments.length} experiments</span>
          <span className="flex items-center gap-1"><Box className="h-3 w-3" /> {models.length} models</span>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-6 p-1 rounded-xl bg-muted/50 border border-border w-fit overflow-x-auto">
        {tabs.map((t) => (
          <button key={t.id} onClick={() => { setActiveTab(t.id); if (t.id !== "projects") setActiveProject(null); setStage("query"); setAnalysis(null); }}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${activeTab === t.id ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground hover:bg-muted"}`}>
            <t.icon className="h-3.5 w-3.5" />{t.label}
          </button>
        ))}
      </div>

      {/* ===================== PROJECTS TAB ===================== */}
      {activeTab === "projects" && !activeProject && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-foreground">My Projects</h3>
            <button onClick={() => setShowNewProject(!showNewProject)} className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90">
              <Plus className="h-3 w-3" /> New Project
            </button>
          </div>

          {showNewProject && (
            <div className="rounded-xl border border-primary/30 bg-card p-4 flex gap-3">
              <input value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)} placeholder="Project name (e.g. Optical Flow for Autonomous Driving)"
                className="flex-1 rounded-lg border border-border bg-muted/30 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                onKeyDown={(e) => e.key === "Enter" && createProject()} />
              <button onClick={createProject} className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-medium">Create</button>
            </div>
          )}

          {loadingProjects ? (
            <div className="flex justify-center py-12"><Loader2 className="h-5 w-5 animate-spin text-primary" /></div>
          ) : projects.length === 0 ? (
            <div className="text-center py-16 text-muted-foreground text-sm">No projects yet. Create your first research project!</div>
          ) : (
            <div className="grid gap-3">
              {projects.map((p) => {
                const pModels = models.filter(m => m.project_id === p.id);
                const pExps = experiments.filter(e => e.project_id === p.id);
                return (
                  <button key={p.id} onClick={() => openProject(p)}
                    className="rounded-xl border border-border bg-card p-4 flex items-center justify-between group hover:border-primary/30 transition-colors text-left w-full">
                    <div className="min-w-0 flex-1">
                      <h4 className="text-sm font-semibold text-foreground">{p.name}</h4>
                      {p.research_question && <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">{p.research_question}</p>}
                      <div className="flex items-center gap-3 mt-1.5">
                        <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${p.status === "active" ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"}`}>{p.status}</span>
                        {pExps.length > 0 && <span className="text-[10px] text-muted-foreground flex items-center gap-1"><Beaker className="h-2.5 w-2.5" /> {pExps.length} exp</span>}
                        {pModels.length > 0 && <span className="text-[10px] text-muted-foreground flex items-center gap-1"><Box className="h-2.5 w-2.5" /> {pModels.length} models</span>}
                        <span className="text-[10px] text-muted-foreground/60 font-mono">{new Date(p.updated_at).toLocaleDateString()}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span onClick={(e) => { e.stopPropagation(); deleteProject(p.id); }}
                        className="p-1.5 rounded-lg hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors opacity-0 group-hover:opacity-100">
                        <Trash2 className="h-3.5 w-3.5" />
                      </span>
                      <ChevronRight className="h-4 w-4 text-muted-foreground" />
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* ===================== ACTIVE PROJECT WORKSPACE ===================== */}
      {activeTab === "projects" && activeProject && (
        <div className="space-y-6">
          {/* Breadcrumb */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <button onClick={() => { setActiveProject(null); setStage("query"); setAnalysis(null); }} className="hover:text-foreground transition-colors">Projects</button>
              <ChevronRight className="h-3 w-3" />
              <span className="text-foreground font-medium">{activeProject.name}</span>
            </div>
            <div className="flex items-center gap-2">
              {analysis && (
                <>
                  <button onClick={downloadReport} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-xs text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors">
                    <FileDown className="h-3 w-3" /> Export Report
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Project summary bar */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground bg-muted/30 rounded-lg px-4 py-2 border border-border">
            <span className="flex items-center gap-1"><Beaker className="h-3 w-3" /> {projectExperiments.length} experiments</span>
            <span className="flex items-center gap-1"><Box className="h-3 w-3" /> {projectModels.length} models</span>
            {analysis && <span className="flex items-center gap-1"><FileText className="h-3 w-3" /> {analysis.papers.length} papers found</span>}
          </div>

          {/* Stage: Query */}
          {(stage === "query" || stage === "analyzing") && (
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
                <Search className="h-4 w-4 text-primary" /> Research Question
              </h3>
              <textarea value={query} onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g. Build a robust optical flow model for autonomous driving"
                rows={3} className="w-full rounded-lg border border-border bg-muted/30 px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 resize-none" />
              <button onClick={analyzeResearch} disabled={analyzing || !query.trim()}
                className="mt-4 flex items-center gap-2 px-5 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50">
                {analyzing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                {analyzing ? "Analyzing papers & generating hypotheses..." : "Analyze Research"}
              </button>
            </div>
          )}

          {/* Stage: Results */}
          {stage === "results" && analysis && (
            <div className="space-y-6">
              {/* Proposal Box - Highlighted */}
              <div className="rounded-xl border-2 border-primary/40 bg-gradient-to-br from-primary/5 to-accent/5 p-6 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-full -translate-y-1/2 translate-x-1/2" />
                <div className="relative">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="h-7 w-7 rounded-lg bg-primary/15 flex items-center justify-center">
                      <Sparkles className="h-4 w-4 text-primary" />
                    </div>
                    <div>
                      <span className="text-[10px] uppercase tracking-wider text-primary font-semibold">Core Proposal</span>
                    </div>
                  </div>
                  <h4 className="text-lg font-bold text-foreground">{analysis.proposal.title}</h4>
                  <p className="text-sm text-muted-foreground mt-2 leading-relaxed">{analysis.proposal.summary}</p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {analysis.proposal.pipeline.map((step, i) => (
                      <span key={i} className="flex items-center gap-1 text-xs bg-primary/10 text-primary px-3 py-1.5 rounded-full font-medium">
                        {i > 0 && <span className="text-primary/40 mr-1">→</span>}
                        {step}
                      </span>
                    ))}
                  </div>
                  <div className="mt-4 rounded-lg bg-primary/10 px-4 py-3 border border-primary/20">
                    <p className="text-xs text-primary font-medium">💡 Key Insight</p>
                    <p className="text-sm text-foreground mt-1">{analysis.proposal.key_insight}</p>
                  </div>
                </div>
              </div>

              {/* Papers - Expandable cards */}
              <div className="rounded-xl border border-border bg-card p-6">
                <h3 className="text-sm font-semibold text-foreground mb-4 flex items-center gap-2">
                  <FileText className="h-4 w-4 text-primary" /> Research Papers ({analysis.papers.length})
                </h3>
                <div className="space-y-2">
                  {analysis.papers.map((p, i) => (
                    <div key={i} className="rounded-lg border border-border bg-muted/10 overflow-hidden transition-all">
                      <button onClick={() => setExpandedPaper(expandedPaper === i ? null : i)}
                        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-muted/30 transition-colors">
                        <div className="flex items-center gap-3 min-w-0 flex-1">
                          <span className="shrink-0 h-6 w-6 rounded-full bg-muted flex items-center justify-center text-[10px] font-bold text-muted-foreground">{i + 1}</span>
                          <div className="min-w-0">
                            <h4 className="text-sm font-semibold text-foreground line-clamp-1">{p.title}</h4>
                            <p className="text-[11px] text-muted-foreground">{p.authors} · {p.year}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2 shrink-0 ml-3">
                          {p.github ? (
                            <span className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-green-500/10 text-green-600 dark:text-green-400 font-medium">
                              <GitBranch className="h-2.5 w-2.5" /> GitHub
                            </span>
                          ) : (
                            <span className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">
                              <AlertCircle className="h-2.5 w-2.5" /> No Repo
                            </span>
                          )}
                          <ChevronRight className={`h-3.5 w-3.5 text-muted-foreground transition-transform ${expandedPaper === i ? "rotate-90" : ""}`} />
                        </div>
                      </button>
                      {expandedPaper === i && (
                        <div className="px-4 pb-4 border-t border-border bg-muted/5">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                            <div className="space-y-2">
                              <div>
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">Problem</span>
                                <p className="text-xs text-foreground mt-0.5">{p.problem}</p>
                              </div>
                              <div>
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">Method</span>
                                <p className="text-xs text-foreground mt-0.5">{p.method}</p>
                              </div>
                            </div>
                            <div className="space-y-2">
                              <div>
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">Results</span>
                                <p className="text-xs text-foreground mt-0.5">{p.results}</p>
                              </div>
                              {p.dataset && (
                                <div>
                                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">Dataset</span>
                                  <p className="text-xs text-foreground mt-0.5">{p.dataset}</p>
                                </div>
                              )}
                            </div>
                          </div>
                          {p.github && (
                            <a href={p.github} target="_blank" rel="noopener noreferrer"
                              className="mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-xs text-muted-foreground hover:text-foreground hover:border-primary/30 transition-colors">
                              <GitBranch className="h-3 w-3" /> View Repository
                              <ExternalLink className="h-2.5 w-2.5 ml-1" />
                            </a>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Hypotheses - Selection */}
              <div className="rounded-xl border border-border bg-card p-6">
                <h3 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                  <FlaskConical className="h-4 w-4 text-accent" /> Choose a Hypothesis
                </h3>
                <p className="text-xs text-muted-foreground mb-4">Select one hypothesis to generate a Jupyter training notebook.</p>
                <div className="grid gap-3">
                  {analysis.hypotheses.map((h) => (
                    <button key={h.id} onClick={() => setSelectedHypothesis(h)}
                      className={`rounded-xl border p-4 text-left transition-all ${selectedHypothesis?.id === h.id ? "border-primary bg-primary/5 ring-2 ring-primary/20" : "border-border bg-muted/20 hover:border-primary/30"}`}>
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`h-7 w-7 rounded-full flex items-center justify-center text-xs font-bold ${selectedHypothesis?.id === h.id ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"}`}>{h.id}</span>
                        <h4 className="text-sm font-semibold text-foreground">{h.name}</h4>
                      </div>
                      <p className="text-xs text-muted-foreground ml-9">{h.architecture}</p>
                      <div className="grid grid-cols-3 gap-2 mt-3 ml-9">
                        <div className="rounded-lg bg-muted/30 px-3 py-2">
                          <span className="text-[10px] text-muted-foreground block">Accuracy</span>
                          <span className="text-xs font-semibold text-foreground">{h.expected_accuracy}</span>
                        </div>
                        <div className="rounded-lg bg-muted/30 px-3 py-2">
                          <span className="text-[10px] text-muted-foreground block">Compute</span>
                          <span className="text-xs font-semibold text-foreground">{h.compute}</span>
                        </div>
                        <div className="rounded-lg bg-muted/30 px-3 py-2">
                          <span className="text-[10px] text-muted-foreground block">Dataset</span>
                          <span className="text-xs font-semibold text-foreground">{h.dataset}</span>
                        </div>
                      </div>
                      <p className="text-[11px] text-muted-foreground/80 mt-2 ml-9 italic">{h.reasoning}</p>
                    </button>
                  ))}
                </div>

                {selectedHypothesis && (
                  <button onClick={generateNotebook}
                    className="mt-4 flex items-center gap-2 px-5 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors">
                    <BookOpen className="h-4 w-4" /> Generate Jupyter Notebook for Hypothesis {selectedHypothesis.id}
                  </button>
                )}
              </div>

              {/* Datasets */}
              {analysis.datasets && analysis.datasets.length > 0 && (
                <div className="rounded-xl border border-border bg-card p-6">
                  <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
                    <Database className="h-4 w-4 text-primary" /> Recommended Datasets
                  </h3>
                  <div className="grid gap-2">
                    {analysis.datasets.map((d, i) => (
                      <div key={i} className="flex items-center justify-between rounded-lg border border-border bg-muted/20 px-4 py-2.5">
                        <div>
                          <span className="text-sm font-medium text-foreground">{d.name}</span>
                          <span className="text-xs text-muted-foreground ml-2">({d.size})</span>
                          {d.source && <span className="text-[10px] text-primary ml-2 bg-primary/10 px-2 py-0.5 rounded-full">{d.source}</span>}
                        </div>
                        {d.url && <a href={d.url} target="_blank" rel="noopener noreferrer" className="text-xs text-primary hover:underline flex items-center gap-1"><Globe className="h-3 w-3" />Link</a>}
                      </div>
                    ))}
                  </div>
                  <p className="text-[11px] text-muted-foreground mt-3">💡 Connect custom datasets later for training. Ultralytics datasets preferred.</p>
                </div>
              )}

              <div className="flex gap-3">
                <button onClick={() => { setStage("query"); setAnalysis(null); setSelectedHypothesis(null); }}
                  className="px-4 py-2 rounded-lg border border-border text-sm text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors">
                  ← New Query
                </button>
                <button onClick={downloadReport}
                  className="px-4 py-2 rounded-lg border border-border text-sm text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors flex items-center gap-2">
                  <FileDown className="h-3.5 w-3.5" /> Download Report
                </button>
              </div>
            </div>
          )}

          {/* Stage: Generating notebook */}
          {stage === "notebook-gen" && (
            <div className="rounded-xl border border-border bg-card p-12 text-center">
              <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto mb-4" />
              <h3 className="text-sm font-semibold text-foreground">Generating Jupyter Notebook</h3>
              <p className="text-xs text-muted-foreground mt-1">Building training pipeline for: {selectedHypothesis?.name}</p>
              <p className="text-[11px] text-muted-foreground/60 mt-2">Includes HuggingFace model upload, W&B logging, and dataset loaders...</p>
            </div>
          )}

          {/* Stage: Notebook ready */}
          {stage === "notebook-ready" && notebookJson && (
            <div className="space-y-4">
              <div className="rounded-xl border-2 border-primary/30 bg-card p-6">
                <h3 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                  <Check className="h-4 w-4 text-primary" /> Notebook Ready
                </h3>
                <p className="text-xs text-muted-foreground mb-4">
                  Generated for <span className="font-medium text-foreground">{selectedHypothesis?.name}</span>. Download and run in Jupyter or Google Colab.
                </p>
                <div className="flex gap-3 flex-wrap">
                  <button onClick={downloadNotebook}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors">
                    <Download className="h-4 w-4" /> Download .ipynb
                  </button>
                  <a href="https://colab.research.google.com/" target="_blank" rel="noopener noreferrer"
                    className="flex items-center gap-2 px-4 py-2.5 rounded-lg border border-border text-sm text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors">
                    <ExternalLink className="h-4 w-4" /> Open Google Colab
                  </a>
                  <a href="https://www.kaggle.com/code" target="_blank" rel="noopener noreferrer"
                    className="flex items-center gap-2 px-4 py-2.5 rounded-lg border border-border text-sm text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors">
                    <ExternalLink className="h-4 w-4" /> Open Kaggle
                  </a>
                </div>
              </div>

              <div className="rounded-xl border border-border bg-card p-4">
                <h4 className="text-xs font-semibold text-muted-foreground mb-2 flex items-center gap-2">
                  <Copy className="h-3 w-3" /> Notebook Preview
                </h4>
                <pre className="bg-muted/50 rounded-lg p-4 text-xs text-foreground overflow-x-auto max-h-96 overflow-y-auto font-mono">
                  {(() => { try { return JSON.stringify(JSON.parse(notebookJson), null, 2).slice(0, 3000) + "\n..."; } catch { return notebookJson.slice(0, 3000) + "\n..."; } })()}
                </pre>
              </div>

              <div className="rounded-xl border border-border bg-muted/20 p-4">
                <h4 className="text-xs font-semibold text-foreground mb-1">Next Steps</h4>
                <ul className="text-xs text-muted-foreground space-y-1">
                  <li>1. Run training in Google Colab, Kaggle, or local machine</li>
                  <li>2. The notebook includes HuggingFace <code className="bg-muted px-1 rounded">push_to_hub</code> to save weights</li>
                  <li>3. Register your trained model in the <button onClick={() => { setActiveTab("models"); setActiveProject(null); }} className="text-primary hover:underline">Models</button> tab with the HuggingFace link</li>
                  <li>4. Registered models can be loaded for inference experiments</li>
                </ul>
              </div>

              <div className="flex gap-3">
                <button onClick={() => { setStage("results"); setNotebookJson(""); }}
                  className="px-4 py-2 rounded-lg border border-border text-sm text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors">
                  ← Back to Results
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ===================== EXPERIMENTS TAB ===================== */}
      {activeTab === "experiments" && (
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-foreground">All Experiments</h3>
          {loadingExperiments ? (
            <div className="flex justify-center py-12"><Loader2 className="h-5 w-5 animate-spin text-primary" /></div>
          ) : experiments.length === 0 ? (
            <div className="text-center py-16">
              <Beaker className="h-10 w-10 text-muted-foreground/30 mx-auto mb-4" />
              <h3 className="text-sm font-semibold text-foreground mb-1">No Experiments Yet</h3>
              <p className="text-xs text-muted-foreground max-w-md mx-auto">Analyze a research question and generate notebooks to create experiments.</p>
            </div>
          ) : (
            <div className="grid gap-3">
              {experiments.map((exp) => {
                const proj = projects.find(p => p.id === exp.project_id);
                return (
                  <div key={exp.id} className="rounded-xl border border-border bg-card p-4 group hover:border-primary/20 transition-colors">
                    <div className="flex items-start justify-between">
                      <div className="min-w-0 flex-1">
                        <h4 className="text-sm font-semibold text-foreground">{exp.title}</h4>
                        {exp.hypothesis && <p className="text-xs text-muted-foreground mt-0.5">Hypothesis: {exp.hypothesis}</p>}
                        <div className="flex items-center gap-3 mt-1.5">
                          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${exp.status === "notebook_generated" ? "bg-primary/10 text-primary" : exp.status === "training" ? "bg-yellow-500/10 text-yellow-600" : exp.status === "completed" ? "bg-green-500/10 text-green-600" : "bg-muted text-muted-foreground"}`}>{exp.status.replace(/_/g, " ")}</span>
                          {proj && <span className="text-[10px] text-muted-foreground">Project: {proj.name}</span>}
                          <span className="text-[10px] text-muted-foreground/60 font-mono">{new Date(exp.created_at).toLocaleDateString()}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {exp.notebook && (
                          <button onClick={() => {
                            try {
                              const parsed = JSON.parse(exp.notebook);
                              const blob = new Blob([JSON.stringify(parsed, null, 2)], { type: "application/json" });
                              const url = URL.createObjectURL(blob);
                              const a = document.createElement("a"); a.href = url; a.download = `${exp.title.replace(/\s+/g, "_")}.ipynb`; a.click(); URL.revokeObjectURL(url);
                            } catch { /* ignore */ }
                          }} className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground hover:text-foreground transition-colors">
                            <Download className="h-3.5 w-3.5" />
                          </button>
                        )}
                        <button onClick={() => deleteExperiment(exp.id)} className="p-1.5 rounded-lg hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors opacity-0 group-hover:opacity-100">
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* ===================== MODELS TAB ===================== */}
      {activeTab === "models" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-foreground">Registered Models</h3>
            <button onClick={() => setShowNewModel(!showNewModel)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90">
              <Plus className="h-3 w-3" /> Register Model
            </button>
          </div>

          {showNewModel && (
            <div className="rounded-xl border border-primary/30 bg-card p-4 space-y-3">
              <input value={newModel.name} onChange={(e) => setNewModel(m => ({ ...m, name: e.target.value }))} placeholder="Model name"
                className="w-full rounded-lg border border-border bg-muted/30 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30" />
              <input value={newModel.huggingface_link} onChange={(e) => setNewModel(m => ({ ...m, huggingface_link: e.target.value }))} placeholder="HuggingFace model link (e.g. https://huggingface.co/user/model)"
                className="w-full rounded-lg border border-border bg-muted/30 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30" />
              <input value={newModel.dataset_used} onChange={(e) => setNewModel(m => ({ ...m, dataset_used: e.target.value }))} placeholder="Dataset used (e.g. COCO, KITTI)"
                className="w-full rounded-lg border border-border bg-muted/30 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30" />
              <select value={newModel.project_id} onChange={(e) => setNewModel(m => ({ ...m, project_id: e.target.value }))}
                className="w-full rounded-lg border border-border bg-muted/30 px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30">
                <option value="">No project (standalone)</option>
                {projects.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
              <button onClick={saveModel} className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-medium">Register</button>
            </div>
          )}

          {loadingModels ? (
            <div className="flex justify-center py-12"><Loader2 className="h-5 w-5 animate-spin text-primary" /></div>
          ) : models.length === 0 ? (
            <div className="text-center py-16">
              <Box className="h-10 w-10 text-muted-foreground/30 mx-auto mb-4" />
              <h3 className="text-sm font-semibold text-foreground mb-1">No Models Yet</h3>
              <p className="text-xs text-muted-foreground max-w-md mx-auto">
                Train a model using a generated notebook, push weights to HuggingFace, then register it here.
              </p>
            </div>
          ) : (
            <div className="grid gap-3">
              {models.map((m) => {
                const proj = projects.find(p => p.id === m.project_id);
                return (
                  <div key={m.id} className="rounded-xl border border-border bg-card p-4 flex items-center justify-between group hover:border-primary/20 transition-colors">
                    <div className="min-w-0 flex-1">
                      <h4 className="text-sm font-semibold text-foreground">{m.name}</h4>
                      <div className="flex items-center gap-3 mt-1">
                        {m.huggingface_link && (
                          <a href={m.huggingface_link} target="_blank" rel="noopener noreferrer" className="text-xs text-primary hover:underline flex items-center gap-1">
                            <ExternalLink className="h-3 w-3" /> HuggingFace
                          </a>
                        )}
                        {m.dataset_used && <span className="text-[11px] text-muted-foreground">Dataset: {m.dataset_used}</span>}
                        {proj && <span className="text-[10px] text-primary/70 bg-primary/10 px-2 py-0.5 rounded-full">{proj.name}</span>}
                      </div>
                      <p className="text-[10px] text-muted-foreground/60 font-mono mt-1">{new Date(m.created_at).toLocaleDateString()}</p>
                    </div>
                    <button onClick={() => deleteModel(m.id)} className="p-1.5 rounded-lg hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors opacity-0 group-hover:opacity-100">
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* ===================== TIMELINE TAB ===================== */}
      {activeTab === "timeline" && (
        <ResearchTimeline projects={projects} experiments={experiments} models={models} />
      )}

      {/* ===================== GRAPH TAB ===================== */}
      {activeTab === "graph" && (
        <ResearchGraph projects={projects} experiments={experiments} models={models} />
      )}
    </div>
  );
}
