import { ReactNode, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Camera, Layers, Mountain, Activity, Box, MessageSquare,
  LayoutDashboard, ChevronRight, BookOpen, Eye, Menu, X,
  Network, FlaskConical
} from "lucide-react";
import AIAssistant from "@/components/AIAssistant";

const modules = [
  { name: "Dashboard", path: "/", icon: LayoutDashboard },
  { name: "Camera Image Formation", path: "/module/camera", icon: Camera },
  { name: "Semantic Information", path: "/module/semantic", icon: Layers },
  { name: "Geometric Information", path: "/module/geometric", icon: Mountain },
  { name: "Motion Estimation", path: "/module/motion", icon: Activity },
  { name: "3D Reconstruction", path: "/module/reconstruction", icon: Box },
  { name: "Scene Reasoning", path: "/module/scene-reasoning", icon: MessageSquare },
  { name: "Tutorials", path: "/tutorials", icon: BookOpen },
  { name: "Perception Studios", path: "/studios", icon: FlaskConical },
  { name: "Knowledge Graph", path: "/knowledge-graph", icon: Network },
];

export default function AppLayout({ children }: { children: ReactNode }) {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex min-h-screen w-full bg-background">
      {/* Mobile header */}
      <div className="fixed top-0 left-0 right-0 z-40 md:hidden flex items-center gap-3 px-4 py-3 border-b border-border bg-sidebar/95 backdrop-blur-sm">
        <button onClick={() => setMobileOpen(!mobileOpen)} className="text-foreground">
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
        <div className="flex items-center gap-2">
          <div className="h-7 w-7 rounded-lg bg-primary/20 flex items-center justify-center">
            <Eye className="h-3.5 w-3.5 text-primary" />
          </div>
          <span className="text-sm font-bold text-foreground">KnowGraph's Perception Lab</span>
        </div>
      </div>

      {/* Sidebar */}
      <aside className={`
        fixed md:sticky top-0 left-0 z-30 h-screen w-64 shrink-0 border-r border-border bg-sidebar flex flex-col
        transition-transform duration-200 md:translate-x-0
        ${mobileOpen ? "translate-x-0" : "-translate-x-full"}
      `}>
        <Link to="/" className="flex items-center gap-3 px-5 py-5 border-b border-border" onClick={() => setMobileOpen(false)}>
          <div className="h-9 w-9 rounded-xl bg-primary/15 flex items-center justify-center glow-primary">
            <Eye className="h-4.5 w-4.5 text-primary" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-foreground tracking-tight leading-tight">KnowGraph's</h1>
            <p className="text-[10px] font-mono text-primary tracking-wider uppercase">Perception Lab</p>
          </div>
        </Link>

        <nav className="flex-1 overflow-y-auto scrollbar-thin py-3 px-3 space-y-0.5 mt-0 md:mt-0">
          {modules.map((mod) => {
            const isActive = location.pathname === mod.path ||
              (mod.path !== "/" && location.pathname.startsWith(mod.path));
            return (
              <Link
                key={mod.path}
                to={mod.path}
                onClick={() => setMobileOpen(false)}
                className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all duration-150 group ${
                  isActive
                    ? "bg-primary/10 text-primary font-medium"
                    : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-foreground"
                }`}
              >
                <mod.icon className={`h-4 w-4 shrink-0 ${isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground"}`} />
                <span className="truncate">{mod.name}</span>
                {isActive && <ChevronRight className="h-3 w-3 ml-auto text-primary" />}
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-border">
          <div className="rounded-lg bg-muted/50 p-3">
            <p className="text-[10px] text-muted-foreground font-mono">v3.0 — KnowGraph Perception Lab</p>
            <p className="text-[9px] text-muted-foreground/60 mt-0.5">Interactive CV Learning Lab</p>
          </div>
        </div>
      </aside>

      {/* Overlay for mobile */}
      {mobileOpen && (
        <div className="fixed inset-0 z-20 bg-background/60 backdrop-blur-sm md:hidden" onClick={() => setMobileOpen(false)} />
      )}

      {/* Main content */}
      <main className="flex-1 overflow-y-auto pt-14 md:pt-0">
        <motion.div
          key={location.pathname}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          className="min-h-screen"
        >
          {children}
        </motion.div>
      </main>

      {/* AI Assistant */}
      <AIAssistant />
    </div>
  );
}
