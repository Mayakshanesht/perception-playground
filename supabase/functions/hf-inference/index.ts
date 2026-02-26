import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Map task types to HuggingFace model IDs
const TASK_MODELS: Record<string, string> = {
  "image-classification": "google/vit-base-patch16-224",
  "object-detection": "facebook/detr-resnet-50",
  "image-segmentation": "facebook/detr-resnet-50-panoptic",
  "depth-estimation": "Intel/dpt-large",
  "image-to-text": "Salesforce/blip-image-captioning-base",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const HF_API_KEY = Deno.env.get("HUGGINGFACE_API_KEY");
    if (!HF_API_KEY) {
      throw new Error("HUGGINGFACE_API_KEY not configured");
    }

    const { image, task, mimeType } = await req.json();

    if (!image || !task) {
      throw new Error("Missing required fields: image, task");
    }

    const model = TASK_MODELS[task];
    if (!model) {
      throw new Error(`Unsupported task: ${task}. Supported: ${Object.keys(TASK_MODELS).join(", ")}`);
    }

    // Convert base64 to binary
    const binaryData = Uint8Array.from(atob(image), (c) => c.charCodeAt(0));

    const response = await fetch(`https://api-inference.huggingface.co/models/${model}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_API_KEY}`,
        "Content-Type": mimeType || "image/jpeg",
      },
      body: binaryData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HuggingFace API error (${response.status}): ${errorText}`);
    }

    // For depth estimation, response is an image
    if (task === "depth-estimation") {
      const arrayBuffer = await response.arrayBuffer();
      const base64Result = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
      return new Response(JSON.stringify({ depth_image: base64Result }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const result = await response.json();
    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
