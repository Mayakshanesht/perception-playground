import { useParams } from "react-router-dom";
import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import ModulePage from "@/components/ModulePage";
import { moduleContents } from "@/data/moduleContent";

export default function GenericModule() {
  const { moduleId } = useParams<{ moduleId: string }>();
  const content = moduleContents[moduleId || ""];

  if (!content) {
    return (
      <div className="p-8">
        <p className="text-muted-foreground">Module not found.</p>
        <Link to="/" className="text-primary text-sm mt-2 inline-block">
          <ArrowLeft className="h-3 w-3 inline mr-1" />Back to Dashboard
        </Link>
      </div>
    );
  }

  return <ModulePage content={content} />;
}
