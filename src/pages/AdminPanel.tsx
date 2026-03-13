import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useSubscription } from "@/hooks/useSubscription";
import { useAuth } from "@/hooks/useAuth";
import { Shield, UserCheck, UserX, RefreshCw, Crown } from "lucide-react";

interface SubUser {
  id: string;
  user_id: string;
  status: string;
  created_at: string;
  approved_at: string | null;
  email?: string;
  name?: string;
}

export default function AdminPanel() {
  const { isAdmin, loading: subLoading } = useSubscription();
  const { user } = useAuth();
  const [users, setUsers] = useState<SubUser[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchUsers = async () => {
    setLoading(true);
    const { data: subs } = await supabase.from("subscriptions").select("*");
    if (!subs) { setLoading(false); return; }

    // Fetch profiles for these users
    const userIds = subs.map((s: any) => s.user_id);
    const { data: profiles } = await supabase.from("profiles").select("id, name, email").in("id", userIds);

    const merged = subs.map((s: any) => {
      const profile = profiles?.find((p: any) => p.id === s.user_id);
      return { ...s, email: profile?.email || "Unknown", name: profile?.name || "" };
    });

    setUsers(merged);
    setLoading(false);
  };

  useEffect(() => {
    if (isAdmin) fetchUsers();
  }, [isAdmin]);

  const toggleStatus = async (sub: SubUser) => {
    const newStatus = sub.status === "active" ? "revoked" : "active";
    await supabase.from("subscriptions").update({
      status: newStatus,
      approved_at: newStatus === "active" ? new Date().toISOString() : null,
      updated_at: new Date().toISOString(),
    }).eq("id", sub.id);
    fetchUsers();
  };

  if (subLoading) return <div className="p-8 text-muted-foreground">Loading...</div>;

  if (!isAdmin) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="text-center">
          <Shield className="h-12 w-12 text-destructive mx-auto mb-4" />
          <h1 className="text-xl font-bold text-foreground mb-2">Access Denied</h1>
          <p className="text-muted-foreground text-sm">You need admin privileges to view this page.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 md:p-10 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
            <Crown className="h-6 w-6 text-primary" />
            Admin Panel
          </h1>
          <p className="text-sm text-muted-foreground mt-1">Manage subscriber access</p>
        </div>
        <button
          onClick={fetchUsers}
          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary text-secondary-foreground text-sm hover:bg-secondary/80 transition-colors"
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {loading ? (
        <p className="text-muted-foreground">Loading users...</p>
      ) : users.length === 0 ? (
        <div className="rounded-xl border border-border bg-card p-8 text-center">
          <p className="text-muted-foreground">No subscribers yet.</p>
        </div>
      ) : (
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                <th className="text-left px-4 py-3 text-muted-foreground font-medium">User</th>
                <th className="text-left px-4 py-3 text-muted-foreground font-medium">Status</th>
                <th className="text-left px-4 py-3 text-muted-foreground font-medium">Joined</th>
                <th className="text-right px-4 py-3 text-muted-foreground font-medium">Action</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id} className="border-b border-border last:border-0">
                  <td className="px-4 py-3">
                    <p className="text-foreground font-medium">{u.name || "—"}</p>
                    <p className="text-xs text-muted-foreground">{u.email}</p>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                      u.status === "active"
                        ? "bg-primary/10 text-primary"
                        : "bg-destructive/10 text-destructive"
                    }`}>
                      {u.status === "active" ? <UserCheck className="h-3 w-3" /> : <UserX className="h-3 w-3" />}
                      {u.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-muted-foreground text-xs">
                    {new Date(u.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <button
                      onClick={() => toggleStatus(u)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                        u.status === "active"
                          ? "bg-destructive/10 text-destructive hover:bg-destructive/20"
                          : "bg-primary/10 text-primary hover:bg-primary/20"
                      }`}
                    >
                      {u.status === "active" ? "Revoke" : "Activate"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
