
-- Remove auto-approve trigger so admin must manually approve
DROP TRIGGER IF EXISTS on_subscription_created ON public.subscriptions;
DROP FUNCTION IF EXISTS public.auto_approve_subscription();
