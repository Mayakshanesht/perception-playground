import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

const SYSTEM_PROMPTS: Record<string, string> = {
  analyze: `You are the Cloudbee Research Copilot, an expert AI research assistant for computer vision developed by Cloudbee Robotics.

When a user asks a research question, respond with EXACTLY this JSON structure (no markdown, pure JSON):

{
  "papers": [
    {
      "title": "Paper Title",
      "authors": "Author1, Author2",
      "year": 2024,
      "problem": "What problem it solves",
      "method": "Core methodology",
      "results": "Key results",
      "github": "https://github.com/...",
      "dataset": "Dataset used"
    }
  ],
  "hypotheses": [
    {
      "id": "A",
      "name": "Hypothesis name",
      "architecture": "Architecture description",
      "expected_accuracy": "Expected performance",
      "compute": "Compute requirements",
      "dataset": "Recommended dataset",
      "reasoning": "Why this approach"
    }
  ],
  "proposal": {
    "title": "Recommended approach title",
    "summary": "2-3 sentence summary of the recommended direction",
    "pipeline": ["Step 1", "Step 2", "Step 3"],
    "key_insight": "The main insight driving this proposal"
  },
  "datasets": [
    {
      "name": "Dataset name",
      "size": "Size info",
      "source": "ultralytics or other source",
      "url": "URL to dataset"
    }
  ]
}

IMPORTANT RULES:
- Return ONLY papers that have REAL, VERIFIED GitHub repositories. Do not invent GitHub links.
- If you are not sure a GitHub link exists, set github to null.
- Include 5-10 relevant papers.
- Generate exactly 3 hypotheses labeled A, B, C.
- Datasets should prefer Ultralytics-compatible datasets when applicable.
- Return ONLY valid JSON, no markdown wrapping, no code fences.`,

  notebook: `You are the Cloudbee Research Copilot notebook generator. Generate a complete Jupyter notebook in JSON format (.ipynb).

The user will provide a hypothesis description. Generate a notebook that implements that hypothesis.

Return ONLY a valid Jupyter notebook JSON (.ipynb format) with this structure:
{
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {
    "kernelspec": { "display_name": "Python 3", "language": "python", "name": "python3" },
    "language_info": { "name": "python", "version": "3.10.0" }
  },
  "cells": [
    { "cell_type": "markdown", "metadata": {}, "source": ["# Title\\n", "Description"] },
    { "cell_type": "code", "metadata": {}, "source": ["!pip install ..."], "execution_count": null, "outputs": [] }
  ]
}

Structure the notebook:
1. Title and hypothesis description (markdown)
2. Install Dependencies (code) - use pip install
3. Import Libraries (code)
4. Dataset Loading (code) - use Ultralytics datasets when possible, with a note about custom dataset connection
5. Model Architecture (code) - full PyTorch model
6. Training Loop (code) - with loss, optimizer, scheduler
7. Evaluation (code) - metrics computation
8. Save to HuggingFace (code) - include huggingface_hub push_to_hub code
9. Visualization (code) - plot results
10. Export and Next Steps (markdown) - mention saving weights to HuggingFace for inference

Include complete, runnable Python code. Use PyTorch.
Add a cell for Weights & Biases logging (optional).
Return ONLY valid JSON, no markdown, no code fences.`,
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { query, action, hypothesis } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    let systemPrompt = SYSTEM_PROMPTS[action] || SYSTEM_PROMPTS.analyze;
    let userMessage = query;

    if (action === "notebook" && hypothesis) {
      userMessage = `Generate a training notebook for this hypothesis:\n\nHypothesis: ${hypothesis.name}\nArchitecture: ${hypothesis.architecture}\nDataset: ${hypothesis.dataset}\nExpected Accuracy: ${hypothesis.expected_accuracy}\n\nOriginal research question: ${query}`;
    }

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
          { role: "user", content: userMessage },
        ],
        stream: action === "analyze" ? false : false,
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

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content || "";

    // Try to parse JSON from the content
    let cleaned = content.trim();
    // Remove markdown code fences if present
    if (cleaned.startsWith("```")) {
      cleaned = cleaned.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "");
    }

    return new Response(JSON.stringify({ result: cleaned }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("research-copilot error:", e);
    return new Response(
      JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
