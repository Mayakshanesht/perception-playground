import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

const SYSTEM_PROMPT = `You are the AI Learning Assistant for KnowGraph's Perception Lab — an interactive computer vision learning platform.

You help students:
- Navigate the curriculum (Camera → Semantics → Geometry → Motion → 3D Reconstruction → Scene Reasoning)
- Explain computer vision concepts clearly with intuition, math, and examples
- Recommend learning paths based on their goals (robotics, autonomous driving, medical imaging, etc.)
- Explain research papers referenced in the modules (AlexNet, ResNet, DETR, RAFT, NeRF, CLIP, etc.)
- Answer questions about equations, architectures, and algorithms
- Suggest relevant tutorials and playgrounds

Key modules in the platform:
1. Camera Image Formation - pinhole model, calibration, lens distortion
2. Semantic Information - classification, object detection (YOLO, Faster R-CNN), segmentation (U-Net, Mask R-CNN)
3. Geometric Information - depth estimation (MiDaS, DPT), pose estimation (HRNet, OpenPose)
4. Motion Estimation - optical flow (Lucas-Kanade, RAFT), tracking (SORT, DeepSORT), action recognition
5. 3D Reconstruction & Rendering - SfM (COLMAP), NeRF, 3D Gaussian Splatting
6. Scene Reasoning - CLIP, Florence-2, GPT-4V, multimodal LLMs

Available playgrounds: Object Detection, Segmentation, Depth Estimation, Pose Estimation, Speed Estimation, SAM2

When explaining papers, structure your response with:
- Problem it solves
- Core idea
- Key equation(s)
- Impact

Be concise, educational, and encouraging. Use LaTeX notation for equations when helpful (wrap in $...$ for inline or $$...$$ for display).`;

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const { messages, mode } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    const systemPrompt = mode === "paper" 
      ? SYSTEM_PROMPT + `\n\nThe student is asking about a specific research paper. Provide a detailed structured explanation with sections: Problem, Core Idea, Architecture/Method, Key Equations, Results, Limitations, and Impact on later research. Be thorough but accessible.`
      : SYSTEM_PROMPT;

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-3-flash-preview",
        messages: [
          { role: "system", content: systemPrompt },
          ...messages,
        ],
        stream: true,
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

    return new Response(response.body, {
      headers: { ...corsHeaders, "Content-Type": "text/event-stream" },
    });
  } catch (e) {
    console.error("cv-assistant error:", e);
    return new Response(JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
