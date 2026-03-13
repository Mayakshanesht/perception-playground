import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/hooks/useAuth";
import ProtectedRoute from "@/components/ProtectedRoute";
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
import Pricing from "@/pages/Pricing";
import PaymentSuccess from "@/pages/PaymentSuccess";
import AdminPanel from "@/pages/AdminPanel";
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
              {/* Public routes */}
              <Route path="/sign-in" element={<SignIn />} />
              <Route path="/sign-up" element={<SignUp />} />
              <Route path="/pricing" element={<Pricing />} />
              <Route path="/payment-success" element={<PaymentSuccess />} />

              {/* Free - just needs auth */}
              <Route path="/" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />

              {/* Free with auth - learning modules & tutorials */}
              <Route path="/module/camera" element={<ProtectedRoute><CameraModule /></ProtectedRoute>} />
              <Route path="/module/semantic" element={<ProtectedRoute><SemanticModule /></ProtectedRoute>} />
              <Route path="/module/geometric" element={<ProtectedRoute><GeometricModule /></ProtectedRoute>} />
              <Route path="/module/motion" element={<ProtectedRoute><MotionModule /></ProtectedRoute>} />
              <Route path="/module/reconstruction" element={<ProtectedRoute><ReconstructionModule /></ProtectedRoute>} />
              <Route path="/module/scene-reasoning" element={<ProtectedRoute><SceneReasoningModule /></ProtectedRoute>} />
              <Route path="/tutorials" element={<ProtectedRoute requireSubscription><Tutorials /></ProtectedRoute>} />
              <Route path="/knowledge-graph" element={<ProtectedRoute><KnowledgeGraph /></ProtectedRoute>} />
              <Route path="/studios" element={<ProtectedRoute><PerceptionStudios /></ProtectedRoute>} />
              <Route path="/module/:moduleId" element={<ProtectedRoute><GenericModule /></ProtectedRoute>} />

              {/* Pro - needs subscription */}
              <Route path="/research-copilot" element={<ProtectedRoute requireSubscription><ResearchCopilot /></ProtectedRoute>} />

              {/* Admin */}
              <Route path="/admin" element={<ProtectedRoute><AdminPanel /></ProtectedRoute>} />

              <Route path="*" element={<NotFound />} />
            </Routes>
          </AppLayout>
        </AuthProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
