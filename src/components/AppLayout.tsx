import { ReactNode, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Camera, Layers, Mountain, Activity, Box, MessageSquare,
  LayoutDashboard, ChevronRight, BookOpen, Eye, Menu, X,
  Network, FlaskConical, Sparkles, LogIn, LogOut, User,
  Crown, Shield, CreditCard
} from "lucide-react";
import AIAssistant from "@/components/AIAssistant";
import { useAuth } from "@/hooks/useAuth";
import { useSubscription } from "@/hooks/useSubscription";

const modules = [
  { name: "Dashboard", path: "/", icon: LayoutDashboard, premium: false },
  { name: "Camera Image Formation", path: "/module/camera", icon: Camera, premium: false },
  { name: "Semantic Information", path: "/module/semantic", icon: Layers, premium: false },
  { name: "Geometric Information", path: "/module/geometric", icon: Mountain, premium: true },
  { name: "Motion Estimation", path: "/module/motion", icon: Activity, premium: true },
  { name: "3D Reconstruction", path: "/module/reconstruction", icon: Box, premium: true },
  { name: "NLP & Large Language Models", path: "/module/nlp-llm", icon: MessageSquare, premium: true },
  { name: "Scene Reasoning & VLMs", path: "/module/scene-reasoning", icon: Eye, premium: true },
  { name: "Generative Vision", path: "/module/generative-vision", icon: Sparkles, premium: true },
  { name: "Tutorials", path: "/tutorials", icon: BookOpen, premium: true },
  { name: "Knowledge Graph", path: "/knowledge-graph", icon: Network, premium: false },
  { name: "Perception Studios", path: "/studios", icon: FlaskConical, premium: false },
  { name: "Research Copilot", path: "/research-copilot", icon: Sparkles, premium: true },
];

export default function AppLayout({ children }: { children: ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();
  const [mobileOpen, setMobileOpen] = useState(false);
  const { user, signOut } = useAuth();
  const { isSubscribed, isAdmin } = useSubscription();

  const handleSignOut = async () => {
    await signOut();
    navigate("/sign-in");
  };

  // Hide sidebar on auth pages
  const authPages = ["/sign-in", "/sign-up"];
  const isAuthPage = authPages.includes(location.pathname);

  if (isAuthPage) {
    return <div className="min-h-screen w-full bg-background">{children}</div>;
  }

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
            const locked = mod.premium && !isSubscribed;
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
                <span className="truncate flex-1">{mod.name}</span>
                {locked && <Crown className="h-3 w-3 text-accent shrink-0" />}
                {isActive && <ChevronRight className="h-3 w-3 ml-auto text-primary" />}
              </Link>
            );
          })}

          {/* Pricing link */}
          <Link
            to="/pricing"
            onClick={() => setMobileOpen(false)}
            className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all duration-150 group ${
              location.pathname === "/pricing"
                ? "bg-primary/10 text-primary font-medium"
                : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-foreground"
            }`}
          >
            <CreditCard className={`h-4 w-4 shrink-0 ${location.pathname === "/pricing" ? "text-primary" : "text-muted-foreground group-hover:text-foreground"}`} />
            <span className="truncate">{isSubscribed ? "My Plan" : "Upgrade to Pro"}</span>
            {!isSubscribed && <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-accent/15 text-accent font-semibold">PRO</span>}
          </Link>

          {/* Admin link */}
          {isAdmin && (
            <Link
              to="/admin"
              onClick={() => setMobileOpen(false)}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all duration-150 group ${
                location.pathname === "/admin"
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-foreground"
              }`}
            >
              <Shield className={`h-4 w-4 shrink-0 ${location.pathname === "/admin" ? "text-primary" : "text-muted-foreground group-hover:text-foreground"}`} />
              <span className="truncate">Admin Panel</span>
            </Link>
          )}
        </nav>

        {/* User section */}
        <div className="p-3 border-t border-border">
          {user ? (
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                <User className="h-4 w-4 text-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-foreground truncate font-medium">{user.email}</p>
                {isSubscribed && (
                  <p className="text-[9px] text-primary font-semibold">PRO</p>
                )}
              </div>
              <button onClick={handleSignOut} className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground hover:text-foreground transition-colors" title="Sign out">
                <LogOut className="h-3.5 w-3.5" />
              </button>
            </div>
          ) : (
            <Link
              to="/sign-in"
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            >
              <LogIn className="h-4 w-4" />
              <span>Sign In</span>
            </Link>
          )}
        </div>

        <div className="px-4 pb-4">
          <div className="rounded-lg bg-muted/50 p-3">
            <p className="text-[10px] text-muted-foreground font-mono">v3.1 — KnowGraph Perception Lab</p>
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
