
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $function$
BEGIN
  INSERT INTO public.profiles (id, name, email)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.raw_user_meta_data->>'name', ''),
    COALESCE(NEW.email, '')
  )
  ON CONFLICT (id) DO UPDATE SET
    name = CASE WHEN profiles.name = '' OR profiles.name IS NULL 
           THEN COALESCE(EXCLUDED.name, profiles.name) 
           ELSE profiles.name END,
    email = COALESCE(EXCLUDED.email, profiles.email),
    updated_at = now();
  RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
