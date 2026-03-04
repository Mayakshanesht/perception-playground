import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import AppLayout from "@/components/AppLayout";
import Dashboard from "@/pages/Dashboard";
import CameraModule from "@/pages/CameraModule";
import SemanticModule from "@/pages/SemanticModule";
import GeometricModule from "@/pages/GeometricModule";
import MotionModule from "@/pages/MotionModule";
import ReconstructionModule from "@/pages/ReconstructionModule";
import SceneReasoningModule from "@/pages/SceneReasoningModule";
import Tutorials from "@/pages/Tutorials";
import GenericModule from "@/pages/GenericModule";
import NotFound from "@/pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
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
            {/* Legacy routes for individual modules */}
            <Route path="/module/:moduleId" element={<GenericModule />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </AppLayout>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
