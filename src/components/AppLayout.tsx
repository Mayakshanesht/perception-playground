import { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Eye, Box, Layers, Mountain, Video, GitBranch, Brain,
  LayoutDashboard, ChevronRight, Scan, Move3D, Users, Waves
} from "lucide-react";

const modules = [
  { name: "Dashboard", path: "/", icon: LayoutDashboard },
  { name: "Classification", path: "/module/classification", icon: Brain },
  { name: "Object Detection", path: "/module/detection", icon: Scan },
  { name: "Segmentation", path: "/module/segmentation", icon: Layers },
  { name: "Depth Estimation", path: "/module/depth", icon: Mountain },
  { name: "Structure from Motion", path: "/module/sfm", icon: Move3D },
  { name: "Neural Rendering", path: "/module/nerf", icon: Box },
  { name: "Pose Estimation", path: "/module/pose", icon: Users },
  { name: "Multi-Object Tracking", path: "/module/tracking", icon: Eye },
  { name: "Action Recognition", path: "/module/action", icon: Video },
  { name: "Optical Flow", path: "/module/opticalflow", icon: Waves },
  { name: "Knowledge Graph", path: "/knowledge-graph", icon: GitBranch },
];

export default function AppLayout({ children }: { children: ReactNode }) {
  const location = useLocation();

  return (
    <div className="flex min-h-screen w-full bg-background">
      {/* Sidebar */}
      <aside className="w-64 shrink-0 border-r border-border bg-sidebar flex flex-col">
        <Link to="/" className="flex items-center gap-3 px-5 py-5 border-b border-border">
          <div className="h-8 w-8 rounded-lg bg-primary/20 flex items-center justify-center glow-primary">
            <Eye className="h-4 w-4 text-primary" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-foreground tracking-tight">Perception Lab</h1>
            <p className="text-[10px] font-mono text-muted-foreground tracking-wider uppercase">Vision Studio</p>
          </div>
        </Link>

        <nav className="flex-1 overflow-y-auto scrollbar-thin py-3 px-3 space-y-0.5">
          {modules.map((mod) => {
            const isActive = location.pathname === mod.path || 
              (mod.path !== "/" && location.pathname.startsWith(mod.path));
            return (
              <Link
                key={mod.path}
                to={mod.path}
                className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-all duration-150 group ${
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
          <div className="rounded-lg bg-muted p-3">
            <p className="text-xs text-muted-foreground font-mono">v1.0 — Educational Platform</p>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
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
    </div>
  );
}
