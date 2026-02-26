import type { VercelRequest, VercelResponse } from "@vercel/node";

const HF_INFERENCE_BASE_URL = "https://api-inference.huggingface.co/models";

const TASK_MODELS: Record<string, string> = {
  "image-classification": "google/vit-base-patch16-224",
  "object-detection": "facebook/detr-resnet-50",
  "image-segmentation": "facebook/detr-resnet-50-panoptic",
  "depth-estimation": "Intel/dpt-large",
  "pose-estimation": "usyd-community/vitpose-base-simple",
  "video-action-recognition": "MCG-NJU/videomae-base-finetuned-kinetics",
  "sam-segmentation": "facebook/sam-vit-base",
  "velocity-estimation": "facebook/detr-resnet-50",
  "perception-pipeline": "facebook/detr-resnet-50",
  "image-to-text": "Salesforce/blip-image-captioning-base",
};

function json(res: VercelResponse, status: number, payload: unknown) {
  res.status(status).setHeader("Content-Type", "application/json").send(JSON.stringify(payload));
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }

  if (req.method !== "POST") {
    json(res, 405, { error: "Method not allowed" });
    return;
  }

  try {
    const { image, video, task, mimeType, options } = req.body ?? {};
    const backendUrl = process.env.INFERENCE_BACKEND_URL;

    if (!task) {
      json(res, 400, { error: "Missing required field: task" });
      return;
    }

    const model = TASK_MODELS[task];
    if (!model) {
      json(res, 400, { error: `Unsupported task: ${task}. Supported: ${Object.keys(TASK_MODELS).join(", ")}` });
      return;
    }

    const payloadBase64 = video || image;
    if (!payloadBase64) {
      json(res, 400, { error: "Missing payload. Provide image (base64) or video (base64)." });
      return;
    }

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
        const errPayload = backendType.includes("application/json")
          ? JSON.stringify(await backendResponse.json())
          : await backendResponse.text();
        throw new Error(`Backend inference error (${backendResponse.status}): ${errPayload}`);
      }

      if (backendType.includes("application/json")) {
        json(res, 200, await backendResponse.json());
        return;
      }

      const buffer = Buffer.from(await backendResponse.arrayBuffer());
      json(res, 200, { output_base64: buffer.toString("base64"), content_type: backendType });
      return;
    }

    // For action recognition, a GPU backend is strongly recommended due to payload size and latency.
    if (["video-action-recognition", "velocity-estimation", "perception-pipeline"].includes(task)) {
      throw new Error(`INFERENCE_BACKEND_URL is required for ${task}. Configure RunPod backend.`);
    }

    const hfApiKey = process.env.HUGGINGFACE_API_KEY;
    if (!hfApiKey) {
      throw new Error("HUGGINGFACE_API_KEY not configured");
    }

    const endpoint = `${HF_INFERENCE_BASE_URL}/${model}`;
    const isSamTask = task === "sam-segmentation";
    const binaryData = Buffer.from(payloadBase64, "base64");

    const hfResponse = await fetch(endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${hfApiKey}`,
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

    const responseType = hfResponse.headers.get("content-type") || "";
    if (responseType.includes("text/html")) {
      const html = await hfResponse.text();
      throw new Error(
        `Hugging Face returned HTML instead of inference output. Check endpoint/token/model support. URL=${endpoint} Snippet=${html.slice(0, 120)}`
      );
    }

    if (!hfResponse.ok) {
      const errorText = responseType.includes("application/json")
        ? JSON.stringify(await hfResponse.json())
        : await hfResponse.text();
      throw new Error(`HuggingFace API error (${hfResponse.status}): ${errorText}`);
    }

    if (task === "depth-estimation") {
      if (!responseType.startsWith("image/")) {
        const nonImagePayload = responseType.includes("application/json")
          ? JSON.stringify(await hfResponse.json())
          : await hfResponse.text();
        throw new Error(`Depth estimation expected image output, got ${responseType || "unknown"}: ${nonImagePayload}`);
      }
      const outputBuffer = Buffer.from(await hfResponse.arrayBuffer());
      json(res, 200, { depth_image: outputBuffer.toString("base64") });
      return;
    }

    json(res, 200, await hfResponse.json());
  } catch (error: any) {
    json(res, 500, { error: error?.message || "Inference request failed" });
  }
}
