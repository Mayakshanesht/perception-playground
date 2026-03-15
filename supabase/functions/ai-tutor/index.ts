import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

const SYSTEM_PROMPT = `You are an AI Tutor for the Perception Playground — a computer vision learning platform. You provide SHORT spoken explanations (20-45 seconds when read aloud, roughly 80-150 words) for specific CV concepts.

Your explanation MUST follow this exact pedagogical structure:

1. Demo (1 sentence) — Briefly describe what the user sees in the visual demo/animation for this concept.
2. Analogy (1-2 sentences) — Explain using a real-world analogy anyone can understand.
3. Concept (1-2 sentences) — Clearly define the computer vision concept.
4. Theory (2-3 sentences) — Provide deeper explanation with relevant math or model logic. Reference key equations or architectures.
5. Bridge (1 sentence) — Summarize the key takeaway and optionally connect to the next concept.

Rules:
- Speak naturally as if narrating to a student watching the demo.
- Reference the visual when possible ("As you can see in the animation...", "Notice how the visualization shows...").
- Keep it conversational but precise. No markdown formatting — this will be read aloud via text-to-speech.
- Do NOT use bullet points, headers, or special characters. Write flowing paragraphs.
- Avoid overly long sentences. Use short, clear statements.
- When mentioning equations, describe them verbally (e.g., "the loss function minimizes the difference between predicted and actual depth").`;

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const { conceptTitle, conceptContent, moduleName } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    const userPrompt = `Generate a spoken explanation for this computer vision concept:

Module: ${moduleName || "Computer Vision"}
Concept: ${conceptTitle}
Content: ${conceptContent || conceptTitle}

Remember: Keep it 80-150 words, natural speech, follow the 5-part structure (Demo → Analogy → Concept → Theory → Bridge). No markdown or special formatting.`;

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-3-flash-preview",
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: userPrompt },
        ],
      }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(JSON.stringify({ error: "Rate limit exceeded. Please try again in a moment." }), {
          status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      if (response.status === 402) {
        return new Response(JSON.stringify({ error: "Usage limit reached. Please try again later." }), {
          status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      const t = await response.text();
      console.error("AI gateway error:", response.status, t);
      return new Response(JSON.stringify({ error: "AI service error" }), {
        status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const data = await response.json();
    const explanation = data.choices?.[0]?.message?.content || "";

    return new Response(JSON.stringify({ explanation }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("ai-tutor error:", e);
    return new Response(JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
