import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/hooks/useAuth";
import AppLayout from "@/components/AppLayout";
import Dashboard from "@/pages/Dashboard";
import CameraModule from "@/pages/CameraModule";
import SemanticModule from "@/pages/SemanticModule";
import GeometricModule from "@/pages/GeometricModule";
import MotionModule from "@/pages/MotionModule";
import ReconstructionModule from "@/pages/ReconstructionModule";
import SceneReasoningModule from "@/pages/SceneReasoningModule";
import Tutorials from "@/pages/Tutorials";
import KnowledgeGraph from "@/pages/KnowledgeGraph";
import PerceptionStudios from "@/pages/PerceptionStudios";
import GenericModule from "@/pages/GenericModule";
import SignIn from "@/pages/SignIn";
import SignUp from "@/pages/SignUp";
import ResearchCopilot from "@/pages/ResearchCopilot";
import NotFound from "@/pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AuthProvider>
          <AppLayout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/module/camera" element={<CameraModule />} />
              <Route path="/module/semantic" element={<SemanticModule />} />
              <Route path="/module/geometric" element={<GeometricModule />} />
              <Route path="/module/motion" element={<MotionModule />} />
              <Route path="/module/reconstruction" element={<ReconstructionModule />} />
              <Route path="/module/scene-reasoning" element={<SceneReasoningModule />} />
              <Route path="/tutorials" element={<Tutorials />} />
              <Route path="/knowledge-graph" element={<KnowledgeGraph />} />
              <Route path="/studios" element={<PerceptionStudios />} />
              <Route path="/research-copilot" element={<ResearchCopilot />} />
              <Route path="/sign-in" element={<SignIn />} />
              <Route path="/sign-up" element={<SignUp />} />
              {/* Legacy routes for individual modules */}
              <Route path="/module/:moduleId" element={<GenericModule />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </AppLayout>
        </AuthProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
