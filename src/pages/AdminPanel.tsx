import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useSubscription } from "@/hooks/useSubscription";
import { useAuth } from "@/hooks/useAuth";
import { Shield, UserCheck, UserX, RefreshCw, Crown, Users, CreditCard, Mail, Copy } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface SubUser {
  id: string;
  user_id: string;
  status: string;
  created_at: string;
  approved_at: string | null;
  email?: string;
  name?: string;
}

interface AppUser {
  id: string;
  name: string;
  email: string;
  created_at: string;
}

export default function AdminPanel() {
  const { isAdmin, loading: subLoading } = useSubscription();
  const { user } = useAuth();
  const { toast } = useToast();
  const [subscribers, setSubscribers] = useState<SubUser[]>([]);
  const [allUsers, setAllUsers] = useState<AppUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"users" | "subscribers">("users");

  const fetchData = async () => {
    setLoading(true);

    // Fetch all profiles (admin RLS policy allows this)
    const { data: profiles } = await supabase.from("profiles").select("id, name, email, created_at");
    if (profiles) {
      setAllUsers(profiles.map((p: any) => ({ id: p.id, name: p.name || "", email: p.email || "Unknown", created_at: p.created_at })));
    }

    // Fetch subscriptions
    const { data: subs } = await supabase.from("subscriptions").select("*");
    if (subs && profiles) {
      const merged = subs.map((s: any) => {
        const profile = profiles.find((p: any) => p.id === s.user_id);
        return { ...s, email: profile?.email || "Unknown", name: profile?.name || "" };
      });
      setSubscribers(merged);
    }

    setLoading(false);
  };

  useEffect(() => {
    if (isAdmin) fetchData();
  }, [isAdmin]);

  const toggleStatus = async (sub: SubUser) => {
    const newStatus = sub.status === "active" ? "revoked" : "active";
    await supabase.from("subscriptions").update({
      status: newStatus,
      approved_at: newStatus === "active" ? new Date().toISOString() : null,
      updated_at: new Date().toISOString(),
    }).eq("id", sub.id);
    fetchData();
  };

  const copyEmails = (list: { email?: string }[]) => {
    const emails = list.map(u => u.email).filter((e): e is string => !!e && e !== "Unknown").join(", ");
    navigator.clipboard.writeText(emails);
    toast({ title: "Copied!", description: `${list.filter(u => u.email && u.email !== "Unknown").length} email(s) copied to clipboard.` });
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
    <div className="p-6 md:p-10 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
            <Crown className="h-6 w-6 text-primary" />
            Admin Panel
          </h1>
          <p className="text-sm text-muted-foreground mt-1">Manage users &amp; subscriptions</p>
        </div>
        <button
          onClick={fetchData}
          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary text-secondary-foreground text-sm hover:bg-secondary/80 transition-colors"
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="rounded-xl border border-border bg-card p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1"><Users className="h-3.5 w-3.5" /> Total Users</div>
          <p className="text-2xl font-bold text-foreground">{allUsers.length}</p>
        </div>
        <div className="rounded-xl border border-border bg-card p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1"><CreditCard className="h-3.5 w-3.5" /> Active Subs</div>
          <p className="text-2xl font-bold text-primary">{subscribers.filter(s => s.status === "active").length}</p>
        </div>
        <div className="rounded-xl border border-border bg-card p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1"><Mail className="h-3.5 w-3.5" /> Pending</div>
          <p className="text-2xl font-bold text-accent">{subscribers.filter(s => s.status === "pending").length}</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-4">
        {(["users", "subscribers"] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              tab === t ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
            }`}
          >
            {t === "users" ? <span className="flex items-center gap-2"><Users className="h-4 w-4" /> All Users ({allUsers.length})</span> :
              <span className="flex items-center gap-2"><CreditCard className="h-4 w-4" /> Subscribers ({subscribers.length})</span>}
          </button>
        ))}
      </div>

      {loading ? (
        <p className="text-muted-foreground">Loading...</p>
      ) : tab === "users" ? (
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/30">
            <span className="text-xs text-muted-foreground font-medium">All registered users</span>
            <button onClick={() => copyEmails(allUsers)} className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-primary/10 text-primary text-xs font-medium hover:bg-primary/20 transition-colors">
              <Copy className="h-3 w-3" /> Copy all emails
            </button>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left px-4 py-3 text-muted-foreground font-medium">Name</th>
                <th className="text-left px-4 py-3 text-muted-foreground font-medium">Email</th>
                <th className="text-left px-4 py-3 text-muted-foreground font-medium">Joined</th>
              </tr>
            </thead>
            <tbody>
              {allUsers.map(u => (
                <tr key={u.id} className="border-b border-border last:border-0 hover:bg-muted/20">
                  <td className="px-4 py-3 text-foreground font-medium">{u.name || "—"}</td>
                  <td className="px-4 py-3 text-muted-foreground text-xs">{u.email}</td>
                  <td className="px-4 py-3 text-muted-foreground text-xs">{new Date(u.created_at).toLocaleDateString()}</td>
                </tr>
              ))}
              {allUsers.length === 0 && (
                <tr><td colSpan={3} className="px-4 py-8 text-center text-muted-foreground">No users yet.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/30">
            <span className="text-xs text-muted-foreground font-medium">Subscribers</span>
            <button onClick={() => copyEmails(subscribers)} className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-primary/10 text-primary text-xs font-medium hover:bg-primary/20 transition-colors">
              <Copy className="h-3 w-3" /> Copy subscriber emails
            </button>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left px-4 py-3 text-muted-foreground font-medium">User</th>
                <th className="text-left px-4 py-3 text-muted-foreground font-medium">Status</th>
                <th className="text-left px-4 py-3 text-muted-foreground font-medium">Joined</th>
                <th className="text-right px-4 py-3 text-muted-foreground font-medium">Action</th>
              </tr>
            </thead>
            <tbody>
              {subscribers.map(u => (
                <tr key={u.id} className="border-b border-border last:border-0 hover:bg-muted/20">
                  <td className="px-4 py-3">
                    <p className="text-foreground font-medium">{u.name || "—"}</p>
                    <p className="text-xs text-muted-foreground">{u.email}</p>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                      u.status === "active" ? "bg-primary/10 text-primary" :
                      u.status === "pending" ? "bg-accent/10 text-accent" :
                      "bg-destructive/10 text-destructive"
                    }`}>
                      {u.status === "active" ? <UserCheck className="h-3 w-3" /> : <UserX className="h-3 w-3" />}
                      {u.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-muted-foreground text-xs">{new Date(u.created_at).toLocaleDateString()}</td>
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
              {subscribers.length === 0 && (
                <tr><td colSpan={4} className="px-4 py-8 text-center text-muted-foreground">No subscribers yet.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
