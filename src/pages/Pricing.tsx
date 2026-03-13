import { Check, Crown, Zap, ArrowRight, Shield, Clock, ExternalLink } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { useSubscription } from "@/hooks/useSubscription";
import { Link } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";

const STRIPE_LINK = "https://buy.stripe.com/eVqdR93RD9664BGbs12kw09";
const freeFeatures = [
  "Dashboard overview",
  "Pipeline visualization",
  "Community access",
];

const proFeatures = [
  "All Dashboard features",
  "Camera, Semantic, Geometric modules",
  "Motion, Reconstruction, Scene Reasoning",
  "Full Tutorials library",
  "Perception Studios",
  "Knowledge Graph",
  "Research Copilot with AI",
  "Notebook generation & export",
  "Priority support",
];

export default function Pricing() {
  const { user } = useAuth();
  const { isSubscribed, isPending, cancelSubscription } = useSubscription();
  const { toast } = useToast();
  const [cancelling, setCancelling] = useState(false);

  const handleSubscribe = async () => {
    if (!user) return;

    // Prevent double-charge: don't allow if already pending or active
    const { data: existing } = await supabase
      .from("subscriptions")
      .select("status")
      .eq("user_id", user.id)
      .maybeSingle();

    if (existing?.status === "active") {
      toast({ title: "Already subscribed", description: "You already have an active subscription." });
      return;
    }
    if (existing?.status === "pending") {
      toast({ title: "Payment pending", description: "Your previous payment is awaiting admin approval." });
      // Still open the link in case they didn't complete it
      const url = `${STRIPE_LINK}?prefilled_email=${encodeURIComponent(user.email || "")}`;
      window.open(url, "_blank");
      return;
    }

    // Open Stripe payment link in new tab
    const url = `${STRIPE_LINK}?prefilled_email=${encodeURIComponent(user.email || "")}`;
    window.open(url, "_blank");

    // Create a pending subscription record
    await supabase.from("subscriptions").upsert({
      user_id: user.id,
      status: "pending",
    }, { onConflict: "user_id" });
    toast({
      title: "Payment started",
      description: "Complete payment in the new tab. An admin will approve your access shortly after.",
    });
  };

  const handleCancel = async () => {
    setCancelling(true);
    const { error } = await cancelSubscription();
    setCancelling(false);
    if (error) {
      toast({ title: "Error", description: "Failed to cancel subscription.", variant: "destructive" });
    } else {
      toast({ title: "Subscription cancelled", description: "You've been moved to the Free plan." });
    }
  };

  return (
    <div className="min-h-screen p-6 md:p-10">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-medium mb-4">
            <Crown className="h-3.5 w-3.5" />
            Perception Lab Pro
          </div>
          <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-3">
            Unlock the Full Research Lab
          </h1>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Get unlimited access to all modules, tutorials, research copilot, and perception studios.
          </p>
        </div>

        {/* Cards */}
        <div className="grid md:grid-cols-2 gap-6 max-w-3xl mx-auto">
          {/* Free */}
          <div className="rounded-xl border border-border bg-card p-6 flex flex-col">
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-foreground mb-1">Free</h3>
              <p className="text-muted-foreground text-sm">Get started with basics</p>
              <div className="mt-4">
                <span className="text-3xl font-bold text-foreground">€0</span>
                <span className="text-muted-foreground text-sm">/month</span>
              </div>
            </div>
            <ul className="space-y-3 flex-1">
              {freeFeatures.map((f) => (
                <li key={f} className="flex items-start gap-2 text-sm text-muted-foreground">
                  <Check className="h-4 w-4 text-muted-foreground/60 mt-0.5 shrink-0" />
                  {f}
                </li>
              ))}
            </ul>
            <div className="mt-6 pt-4 border-t border-border">
              {user ? (
                <span className="text-xs text-muted-foreground">Current plan</span>
              ) : (
                <Link to="/sign-up" className="inline-flex items-center gap-2 text-sm text-primary hover:underline font-medium">
                  Sign up free <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              )}
            </div>
          </div>

          {/* Pro */}
          <div className="rounded-xl border-2 border-primary/40 bg-card p-6 flex flex-col relative overflow-hidden">
            <div className="absolute top-0 right-0 px-3 py-1 bg-primary text-primary-foreground text-xs font-semibold rounded-bl-lg">
              Recommended
            </div>
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-foreground mb-1 flex items-center gap-2">
                <Zap className="h-4 w-4 text-primary" />
                Pro
              </h3>
              <p className="text-muted-foreground text-sm">Full research lab access</p>
              <div className="mt-4">
                <span className="text-3xl font-bold text-foreground">€9.99</span>
                <span className="text-muted-foreground text-sm">/month</span>
              </div>
            </div>
            <ul className="space-y-3 flex-1">
              {proFeatures.map((f) => (
                <li key={f} className="flex items-start gap-2 text-sm text-foreground">
                  <Check className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                  {f}
                </li>
              ))}
            </ul>
            <div className="mt-6 pt-4 border-t border-border">
              {isSubscribed ? (
                <div className="flex items-center gap-2 text-sm text-primary">
                  <Shield className="h-4 w-4" />
                  <span className="font-medium">Active subscription</span>
                </div>
              ) : isPending ? (
                <div className="flex items-center gap-2 text-sm text-accent">
                  <Clock className="h-4 w-4" />
                  <span className="font-medium">Pending admin approval</span>
                </div>
              ) : (
                <button
                  onClick={handleSubscribe}
                  className="w-full rounded-lg bg-primary text-primary-foreground py-2.5 text-sm font-medium hover:bg-primary/90 transition-colors flex items-center justify-center gap-2"
                >
                  Subscribe Now <ExternalLink className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Info */}
        <p className="text-center text-xs text-muted-foreground mt-8">
          After payment, an admin will review and approve your access. You'll get full access once approved.
        </p>
      </div>
    </div>
  );
}
