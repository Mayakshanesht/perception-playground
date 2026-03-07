import { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  Search, FileText, GitBranch, Download, Plus, ArrowLeft, Loader2,
  Sparkles, FlaskConical, Box, Trash2, FolderOpen, ChevronRight,
  ExternalLink, BookOpen, Database, Check, Copy, Globe
} from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";

type Tab = "projects" | "models";

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

// Workflow stages within a project
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

  // Active project state
  const [activeProject, setActiveProject] = useState<Project | null>(null);
  const [stage, setStage] = useState<Stage>("query");
  const [query, setQuery] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [selectedHypothesis, setSelectedHypothesis] = useState<Hypothesis | null>(null);
  const [notebookJson, setNotebookJson] = useState<string>("");
  const [generatingNotebook, setGeneratingNotebook] = useState(false);

  // Models
  const [models, setModels] = useState<SavedModel[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [showNewModel, setShowNewModel] = useState(false);
  const [newModel, setNewModel] = useState({ name: "", huggingface_link: "", dataset_used: "" });

  useEffect(() => {
    if (!authLoading && !user) navigate("/sign-in");
  }, [user, authLoading, navigate]);

  useEffect(() => {
    if (user) {
      fetchProjects();
      fetchModels();
    }
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

  const createProject = async () => {
    if (!newProjectName.trim()) return;
    const { error } = await supabase.from("research_projects").insert({
      user_id: user!.id,
      name: newProjectName.trim(),
    });
    if (error) {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    } else {
      setNewProjectName("");
      setShowNewProject(false);
      fetchProjects();
      toast({ title: "Project created" });
    }
  };

  const deleteProject = async (id: string) => {
    await supabase.from("research_projects").delete().eq("id", id);
    if (activeProject?.id === id) {
      setActiveProject(null);
      setStage("query");
      setAnalysis(null);
    }
    fetchProjects();
  };

  const openProject = (p: Project) => {
    setActiveProject(p);
    setQuery(p.research_question || "");
    setStage("query");
    setAnalysis(null);
    setSelectedHypothesis(null);
    setNotebookJson("");
  };

  const callCopilot = async (action: string, extraBody: any = {}) => {
    const endpoint = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/research-copilot`;
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
      },
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
    setAnalyzing(true);
    setStage("analyzing");
    setAnalysis(null);

    try {
      const { result } = await callCopilot("analyze");
      const parsed: AnalysisData = JSON.parse(result);
      setAnalysis(parsed);
      setStage("results");

      // Save to project
      await supabase.from("research_projects").update({
        research_question: query.trim(),
        papers: parsed.papers,
        architecture: parsed.proposal,
        updated_at: new Date().toISOString(),
      }).eq("id", activeProject.id);
    } catch (err: any) {
      toast({ title: "Analysis failed", description: err.message, variant: "destructive" });
      setStage("query");
    } finally {
      setAnalyzing(false);
    }
  };

  const generateNotebook = async () => {
    if (!selectedHypothesis || !activeProject) return;
    setGeneratingNotebook(true);
    setStage("notebook-gen");

    try {
      const { result } = await callCopilot("notebook", { hypothesis: selectedHypothesis });
      setNotebookJson(result);
      setStage("notebook-ready");

      // Save notebook ref to project
      await supabase.from("research_projects").update({
        notebooks: [{ hypothesis: selectedHypothesis.name, generated_at: new Date().toISOString() }],
        updated_at: new Date().toISOString(),
      }).eq("id", activeProject.id);
    } catch (err: any) {
      toast({ title: "Notebook generation failed", description: err.message, variant: "destructive" });
      setStage("results");
    } finally {
      setGeneratingNotebook(false);
    }
  };

  const downloadNotebook = () => {
    if (!notebookJson) return;
    try {
      // Validate it's JSON
      const parsed = JSON.parse(notebookJson);
      const blob = new Blob([JSON.stringify(parsed, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${selectedHypothesis?.name?.replace(/\s+/g, "_") || "notebook"}.ipynb`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // Fallback: download as .py
      const blob = new Blob([notebookJson], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "notebook.py";
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  const saveModel = async () => {
    if (!newModel.name.trim() || !newModel.huggingface_link.trim()) return;
    const { error } = await supabase.from("saved_models").insert({
      user_id: user!.id,
      name: newModel.name.trim(),
      huggingface_link: newModel.huggingface_link.trim(),
      dataset_used: newModel.dataset_used.trim(),
      project_id: activeProject?.id || null,
    });
    if (error) {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    } else {
      setNewModel({ name: "", huggingface_link: "", dataset_used: "" });
      setShowNewModel(false);
      fetchModels();
      toast({ title: "Model registered" });
    }
  };

  const deleteModel = async (id: string) => {
    await supabase.from("saved_models").delete().eq("id", id);
    fetchModels();
  };

  if (authLoading) return <div className="flex items-center justify-center min-h-screen"><Loader2 className="h-6 w-6 animate-spin text-primary" /></div>;
  if (!user) return null;

  const tabs: { id: Tab; label: string; icon: any }[] = [
    { id: "projects", label: "Projects", icon: FolderOpen },
    { id: "models", label: "Models", icon: Box },
  ];

  return (
    <div className="p-4 md:p-8 max-w-7xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors mb-4">
        <ArrowLeft className="h-3 w-3" /> Dashboard
      </Link>

      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
          <Sparkles className="h-5 w-5 text-primary" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-foreground tracking-tight">Cloudbee Research Copilot</h1>
          <p className="text-[11px] text-muted-foreground">by Cloudbee Robotics</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-6 p-1 rounded-xl bg-muted/50 border border-border w-fit">
        {tabs.map((t) => (
          <button key={t.id} onClick={() => { setActiveTab(t.id); setActiveProject(null); setStage("query"); setAnalysis(null); }}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all ${activeTab === t.id ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground hover:bg-muted"}`}>
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
              {projects.map((p) => (
                <button key={p.id} onClick={() => openProject(p)}
                  className="rounded-xl border border-border bg-card p-4 flex items-center justify-between group hover:border-primary/30 transition-colors text-left w-full">
                  <div className="min-w-0 flex-1">
                    <h4 className="text-sm font-semibold text-foreground">{p.name}</h4>
                    {p.research_question && <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">{p.research_question}</p>}
                    <p className="text-[10px] text-muted-foreground/60 font-mono mt-1">{new Date(p.updated_at).toLocaleDateString()}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span onClick={(e) => { e.stopPropagation(); deleteProject(p.id); }}
                      className="p-1.5 rounded-lg hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors opacity-0 group-hover:opacity-100">
                      <Trash2 className="h-3.5 w-3.5" />
                    </span>
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ===================== ACTIVE PROJECT WORKSPACE ===================== */}
      {activeTab === "projects" && activeProject && (
        <div className="space-y-6">
          {/* Breadcrumb */}
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <button onClick={() => { setActiveProject(null); setStage("query"); setAnalysis(null); }} className="hover:text-foreground transition-colors">Projects</button>
            <ChevronRight className="h-3 w-3" />
            <span className="text-foreground font-medium">{activeProject.name}</span>
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
              {/* Papers */}
              <div className="rounded-xl border border-border bg-card p-6">
                <h3 className="text-sm font-semibold text-foreground mb-4 flex items-center gap-2">
                  <FileText className="h-4 w-4 text-primary" /> Found Papers ({analysis.papers.length})
                </h3>
                <div className="space-y-3">
                  {analysis.papers.map((p, i) => (
                    <div key={i} className="rounded-lg border border-border bg-muted/20 p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0 flex-1">
                          <h4 className="text-sm font-semibold text-foreground">{p.title}</h4>
                          <p className="text-[11px] text-muted-foreground mt-0.5">{p.authors} · {p.year}</p>
                          <p className="text-xs text-muted-foreground mt-2"><span className="text-foreground font-medium">Problem:</span> {p.problem}</p>
                          <p className="text-xs text-muted-foreground mt-1"><span className="text-foreground font-medium">Method:</span> {p.method}</p>
                          <p className="text-xs text-muted-foreground mt-1"><span className="text-foreground font-medium">Results:</span> {p.results}</p>
                          {p.dataset && <p className="text-xs text-muted-foreground mt-1"><span className="text-foreground font-medium">Dataset:</span> {p.dataset}</p>}
                        </div>
                        {p.github && (
                          <a href={p.github} target="_blank" rel="noopener noreferrer"
                            className="shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-xs text-muted-foreground hover:text-foreground hover:border-primary/30 transition-colors">
                            <GitBranch className="h-3 w-3" /> GitHub
                          </a>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Proposal Box */}
              <div className="rounded-xl border-2 border-primary/30 bg-primary/5 p-6">
                <h3 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-primary" /> Recommended Proposal
                </h3>
                <h4 className="text-base font-bold text-foreground">{analysis.proposal.title}</h4>
                <p className="text-sm text-muted-foreground mt-2">{analysis.proposal.summary}</p>
                <div className="mt-4 flex flex-wrap gap-2">
                  {analysis.proposal.pipeline.map((step, i) => (
                    <span key={i} className="flex items-center gap-1 text-xs bg-primary/10 text-primary px-3 py-1.5 rounded-full font-medium">
                      {i > 0 && <span className="text-primary/40 mr-1">→</span>}
                      {step}
                    </span>
                  ))}
                </div>
                <p className="text-xs text-primary/80 mt-3 italic">💡 {analysis.proposal.key_insight}</p>
              </div>

              {/* Hypotheses */}
              <div className="rounded-xl border border-border bg-card p-6">
                <h3 className="text-sm font-semibold text-foreground mb-4 flex items-center gap-2">
                  <FlaskConical className="h-4 w-4 text-accent" /> Choose a Hypothesis
                </h3>
                <p className="text-xs text-muted-foreground mb-4">Select one hypothesis to generate a training notebook for.</p>
                <div className="grid gap-3">
                  {analysis.hypotheses.map((h) => (
                    <button key={h.id} onClick={() => setSelectedHypothesis(h)}
                      className={`rounded-xl border p-4 text-left transition-all ${selectedHypothesis?.id === h.id ? "border-primary bg-primary/5 ring-2 ring-primary/20" : "border-border bg-muted/20 hover:border-primary/30"}`}>
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`h-6 w-6 rounded-full flex items-center justify-center text-xs font-bold ${selectedHypothesis?.id === h.id ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"}`}>
                          {h.id}
                        </span>
                        <h4 className="text-sm font-semibold text-foreground">{h.name}</h4>
                      </div>
                      <p className="text-xs text-muted-foreground ml-8">{h.architecture}</p>
                      <div className="flex gap-4 mt-2 ml-8">
                        <span className="text-[11px] text-muted-foreground"><span className="text-foreground font-medium">Accuracy:</span> {h.expected_accuracy}</span>
                        <span className="text-[11px] text-muted-foreground"><span className="text-foreground font-medium">Compute:</span> {h.compute}</span>
                        <span className="text-[11px] text-muted-foreground"><span className="text-foreground font-medium">Dataset:</span> {h.dataset}</span>
                      </div>
                      <p className="text-[11px] text-muted-foreground/80 mt-2 ml-8 italic">{h.reasoning}</p>
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
                  <p className="text-[11px] text-muted-foreground mt-3">💡 Users can connect custom datasets later for training.</p>
                </div>
              )}

              {/* Export analysis */}
              <div className="flex gap-3">
                <button onClick={() => { setStage("query"); setAnalysis(null); setSelectedHypothesis(null); }}
                  className="px-4 py-2 rounded-lg border border-border text-sm text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors">
                  ← New Query
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
                </div>
              </div>

              {/* Notebook preview */}
              <div className="rounded-xl border border-border bg-card p-4">
                <h4 className="text-xs font-semibold text-muted-foreground mb-2 flex items-center gap-2">
                  <Copy className="h-3 w-3" /> Notebook Preview
                </h4>
                <pre className="bg-muted/50 rounded-lg p-4 text-xs text-foreground overflow-x-auto max-h-96 overflow-y-auto font-mono">
                  {(() => {
                    try {
                      return JSON.stringify(JSON.parse(notebookJson), null, 2).slice(0, 3000) + "\n...";
                    } catch {
                      return notebookJson.slice(0, 3000) + "\n...";
                    }
                  })()}
                </pre>
              </div>

              <div className="rounded-xl border border-border bg-muted/20 p-4">
                <h4 className="text-xs font-semibold text-foreground mb-1">Next Steps</h4>
                <ul className="text-xs text-muted-foreground space-y-1">
                  <li>1. Run training in Google Colab or local machine</li>
                  <li>2. The notebook includes HuggingFace <code className="bg-muted px-1 rounded">push_to_hub</code> to save weights</li>
                  <li>3. Register your trained model in the <button onClick={() => setActiveTab("models")} className="text-primary hover:underline">Models</button> tab with the HuggingFace link</li>
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
                Train a model using a generated notebook, push weights to HuggingFace, then register it here to track your experiments.
              </p>
            </div>
          ) : (
            <div className="grid gap-3">
              {models.map((m) => (
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
                    </div>
                    <p className="text-[10px] text-muted-foreground/60 font-mono mt-1">{new Date(m.created_at).toLocaleDateString()}</p>
                  </div>
                  <button onClick={() => deleteModel(m.id)} className="p-1.5 rounded-lg hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors opacity-0 group-hover:opacity-100">
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
