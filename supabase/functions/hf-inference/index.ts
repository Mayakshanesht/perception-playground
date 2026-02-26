import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};
const HF_INFERENCE_BASE_URL = "https://api-inference.huggingface.co/models";

function uint8ArrayToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

// Map task types to HuggingFace model IDs
const TASK_MODELS: Record<string, string> = {
  "image-classification": "google/vit-base-patch16-224",
  "object-detection": "facebook/detr-resnet-50",
  "image-segmentation": "facebook/detr-resnet-50-panoptic",
  "depth-estimation": "Intel/dpt-large",
  "pose-estimation": "usyd-community/vitpose-base-simple",
  "video-action-recognition": "MCG-NJU/videomae-base-finetuned-kinetics",
  "sam-segmentation": "facebook/sam-vit-base",
  "image-to-text": "Salesforce/blip-image-captioning-base",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { image, video, task, mimeType, options } = await req.json();
    const backendUrl = Deno.env.get("INFERENCE_BACKEND_URL");

    if (!task) {
      throw new Error("Missing required field: task");
    }

    const model = TASK_MODELS[task];
    if (!model) {
      throw new Error(`Unsupported task: ${task}. Supported: ${Object.keys(TASK_MODELS).join(", ")}`);
    }

    const payloadBase64 = video || image;
    if (!payloadBase64) {
      throw new Error("Missing payload. Provide image (base64) for image tasks or video (base64) for video tasks.");
    }

    // Convert base64 to binary
    const binaryData = Uint8Array.from(atob(payloadBase64), (c) => c.charCodeAt(0));

    if (backendUrl) {
      const backendResponse = await fetch(backendUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task,
          model,
          payloadBase64,
          mimeType: mimeType || "application/octet-stream",
          options: options ?? {},
        }),
      });

      const backendType = backendResponse.headers.get("content-type") || "";
      if (!backendResponse.ok) {
        const backendErr = backendType.includes("application/json")
          ? JSON.stringify(await backendResponse.json())
          : await backendResponse.text();
        throw new Error(`Backend inference error (${backendResponse.status}): ${backendErr}`);
      }

      if (backendType.includes("application/json")) {
        const backendJson = await backendResponse.json();
        return new Response(JSON.stringify(backendJson), {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }

      const backendBuffer = await backendResponse.arrayBuffer();
      const backendBase64 = uint8ArrayToBase64(new Uint8Array(backendBuffer));
      return new Response(JSON.stringify({ output_base64: backendBase64, content_type: backendType }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const HF_API_KEY = Deno.env.get("HUGGINGFACE_API_KEY");
    if (!HF_API_KEY) {
      throw new Error("HUGGINGFACE_API_KEY not configured");
    }

    const endpoint = `${HF_INFERENCE_BASE_URL}/${model}`;
    const isSamTask = task === "sam-segmentation";

    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_API_KEY}`,
        "Content-Type": isSamTask ? "application/json" : (mimeType || "image/jpeg"),
      },
      body: isSamTask
        ? JSON.stringify({
            inputs: {
              image: `data:${mimeType || "image/jpeg"};base64,${payloadBase64}`,
            },
            parameters: options ?? {},
          })
        : binaryData,
    });

    const responseType = response.headers.get("content-type") || "";
    if (responseType.includes("text/html")) {
      const html = await response.text();
      throw new Error(
        `Hugging Face returned HTML instead of inference output. Check endpoint/token/model support. URL=${endpoint} Snippet=${html.slice(0, 120)}`
      );
    }

    if (!response.ok) {
      const errorText = responseType.includes("application/json")
        ? JSON.stringify(await response.json())
        : await response.text();
      throw new Error(`HuggingFace API error (${response.status}): ${errorText}`);
    }

    // For depth estimation, response is expected as an image payload.
    if (task === "depth-estimation") {
      if (!responseType.startsWith("image/")) {
        const nonImagePayload = responseType.includes("application/json")
          ? JSON.stringify(await response.json())
          : await response.text();
        throw new Error(`Depth estimation expected image output, got ${responseType || "unknown"}: ${nonImagePayload}`);
      }
      const arrayBuffer = await response.arrayBuffer();
      const base64Result = uint8ArrayToBase64(new Uint8Array(arrayBuffer));
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
