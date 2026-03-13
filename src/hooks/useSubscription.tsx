import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/useAuth";

export function useSubscription() {
  const { user } = useAuth();
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [isPending, setIsPending] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user) {
      setIsSubscribed(false);
      setIsPending(false);
      setIsAdmin(false);
      setLoading(false);
      return;
    }

    const fetchStatus = async () => {
      setLoading(true);
      const [subResult, roleResult] = await Promise.all([
        supabase.from("subscriptions").select("status").eq("user_id", user.id).maybeSingle(),
        supabase.from("user_roles").select("role").eq("user_id", user.id),
      ]);

      setIsSubscribed(subResult.data?.status === "active");
      setIsPending(subResult.data?.status === "pending");
      setIsAdmin(roleResult.data?.some((r: any) => r.role === "admin") ?? false);
      setLoading(false);
    };

    fetchStatus();
  }, [user]);

  const activateSubscription = async () => {
    if (!user) return { error: "Not authenticated" };
    const { error } = await supabase.from("subscriptions").upsert({
      user_id: user.id,
      status: "active",
    }, { onConflict: "user_id" });
    if (!error) setIsSubscribed(true);
    return { error };
  };

  return { isSubscribed, isPending, isAdmin, loading, activateSubscription };
}
