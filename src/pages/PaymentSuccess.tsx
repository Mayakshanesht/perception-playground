import { Link } from "react-router-dom";
import { Clock, ArrowRight } from "lucide-react";

export default function PaymentSuccess() {
  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="text-center max-w-md">
        <div className="h-16 w-16 rounded-2xl bg-accent/10 flex items-center justify-center mx-auto mb-6">
          <Clock className="h-8 w-8 text-accent" />
        </div>
        <h1 className="text-2xl font-bold text-foreground mb-2">Payment Received!</h1>
        <p className="text-muted-foreground text-sm mb-6">
          Thank you for subscribing! An admin will review and approve your access shortly. You'll get full access to all modules once approved.
        </p>
        <Link
          to="/"
          className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
        >
          Go to Dashboard <ArrowRight className="h-4 w-4" />
        </Link>
      </div>
    </div>
  );
}
