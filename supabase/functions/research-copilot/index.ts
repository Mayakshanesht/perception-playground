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

  notebook: `You are the Cloudbee Research Copilot notebook generator. You MUST generate a COMPLETE, RUNNABLE Jupyter/Colab notebook in valid .ipynb JSON format.

CRITICAL RULES:
- Output MUST be a valid JSON object with keys: nbformat, nbformat_minor, metadata, cells
- Every code cell MUST have: cell_type, metadata, source (array of strings), execution_count (null), outputs ([])
- Every markdown cell MUST have: cell_type, metadata, source (array of strings)
- Each line in "source" array must end with "\\n" except the last line
- Do NOT output Python code directly — wrap ALL code inside notebook cells
- Do NOT use markdown code fences — return raw JSON only
- The notebook must be COMPLETE with real, working, production-quality code

Required .ipynb structure:
{
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {
    "kernelspec": { "display_name": "Python 3", "language": "python", "name": "python3" },
    "language_info": { "name": "python", "version": "3.10.0" },
    "colab": { "provenance": [] }
  },
  "cells": [ ... ]
}

Generate these cells IN ORDER (each as a separate cell):

1. **Title Cell** (markdown): Hypothesis name, description, expected outcomes
2. **Environment Setup** (code): !pip install torch torchvision ultralytics huggingface_hub wandb matplotlib numpy tqdm
3. **Imports** (code): Import ALL required libraries (torch, torch.nn, torchvision, ultralytics, huggingface_hub, wandb, matplotlib, numpy, tqdm, os, etc.)
4. **Configuration** (code): Hyperparameters dict (lr, batch_size, epochs, device, model_name, dataset_name, hf_repo_id) with clear comments
5. **Dataset Setup** (code): Complete dataset loading with Ultralytics or torchvision. Include train/val split, DataLoader creation, data augmentation transforms. Add comment about connecting custom datasets.
6. **Model Architecture** (code): COMPLETE PyTorch nn.Module class definition. Include __init__ and forward methods with full layer definitions. Not pseudocode — real tensors, real layers.
7. **Training Utilities** (code): Loss function, optimizer (AdamW), learning rate scheduler (CosineAnnealingLR), early stopping logic
8. **W&B Logging Setup** (code): wandb.init with project name, config logging (wrap in try/except for optional use)
9. **Training Loop** (code): COMPLETE training loop with: epoch iteration, batch iteration, forward pass, loss computation, backward pass, optimizer step, scheduler step, validation after each epoch, metric logging, best model checkpoint saving, progress bars with tqdm
10. **Evaluation** (code): Load best checkpoint, run full evaluation on test/val set, compute metrics (accuracy, precision, recall, F1, confusion matrix), print results table
11. **Visualization** (code): Plot training/validation loss curves, accuracy curves, sample predictions using matplotlib. At least 3 plots.
12. **Save to HuggingFace** (code): Complete huggingface_hub integration — login, create_repo, push model weights, push model config, push training metrics JSON
13. **Inference Example** (code): Load saved model, run inference on a sample input, display result
14. **Next Steps** (markdown): Summary of results, suggestions for improvement, mention that weights are on HuggingFace for deployment

IMPORTANT: Every code cell must contain COMPLETE, RUNNABLE Python code. No placeholders like "# TODO" or "pass" or "...". Write the actual implementation. This notebook should run end-to-end in Google Colab without modification.

Return ONLY the valid .ipynb JSON object. No markdown wrapping. No code fences. No explanation outside the notebook.`,
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
