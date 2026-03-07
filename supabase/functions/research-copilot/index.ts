import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

const SYSTEM_PROMPTS: Record<string, string> = {
  analyze: `You are the Cloudbee Research Copilot, an expert AI research assistant for computer vision and perception systems developed by Cloudbee Robotics.

When a user asks a research question, provide a comprehensive analysis:

1. **Paper Analysis**: List 5-8 relevant research papers with:
   - Paper title, authors, year
   - Problem solved
   - Core methodology
   - Key results
   - GitHub repo link (if known)

2. **Research Hypotheses**: Generate 3 concrete hypotheses/approaches:
   - Hypothesis name
   - Architecture description
   - Expected performance
   - Compute requirements
   - Dataset compatibility

3. **Architecture Proposal**: For the most promising approach:
   - Model architecture diagram (text-based)
   - Key components explained
   - Training strategy
   - Loss functions

4. **Datasets**: Recommend relevant datasets with sizes and benchmarks.

5. **Implementation Roadmap**: Step-by-step plan.

Use markdown formatting with headers, lists, and code blocks.`,

  notebook: `You are the Cloudbee Research Copilot notebook generator. Generate a complete Python training notebook in markdown format.

Structure:
1. **Setup & Dependencies** - pip install commands
2. **Dataset Loading** - DataLoader with transforms
3. **Model Architecture** - Full PyTorch model definition
4. **Training Loop** - With loss, optimizer, scheduler
5. **Evaluation** - Metrics computation
6. **Visualization** - Plot results

Include complete, runnable Python code. Use PyTorch. Add comments explaining each section.
The notebook should be ready to copy into Google Colab or Jupyter.`,
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { query, action } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    const systemPrompt = SYSTEM_PROMPTS[action] || SYSTEM_PROMPTS.analyze;

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: query },
        ],
        stream: true,
      }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(JSON.stringify({ error: "Rate limit exceeded. Please try again later." }), {
          status: 429,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      if (response.status === 402) {
        return new Response(JSON.stringify({ error: "Usage credits exhausted. Please add credits." }), {
          status: 402,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      const t = await response.text();
      console.error("AI gateway error:", response.status, t);
      return new Response(JSON.stringify({ error: "AI gateway error" }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    return new Response(response.body, {
      headers: { ...corsHeaders, "Content-Type": "text/event-stream" },
    });
  } catch (e) {
    console.error("research-copilot error:", e);
    return new Response(
      JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
