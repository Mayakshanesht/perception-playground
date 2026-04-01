import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";

export default function AuthCallback() {
  const navigate = useNavigate();

  useEffect(() => {
    const handleCallback = async () => {
      try {
        // Try to exchange code if present in URL
        const url = new URL(window.location.href);
        const code = url.searchParams.get("code");
        if (code) {
          await supabase.auth.exchangeCodeForSession(window.location.href);
        }
        // Also check hash fragment for implicit flow
        if (window.location.hash) {
          const { data } = await supabase.auth.getSession();
          if (!data.session) {
            // Wait a moment for auth state to settle
            await new Promise(r => setTimeout(r, 1000));
          }
        }
      } catch (e) {
        console.error("Auth callback error:", e);
      }
      navigate("/", { replace: true });
    };
    handleCallback();
  }, [navigate]);

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <div className="h-8 w-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-sm text-muted-foreground">Signing you in…</p>
      </div>
    </div>
  );
}
