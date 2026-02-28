import type { VercelRequest, VercelResponse } from "@vercel/node";

const TASK_MODELS: Record<string, string> = {
  "object-detection": "yolo26n.pt",
  "image-segmentation": "yolo26n-seg.pt",
  "pose-estimation": "yolo26n-pose.pt",
  "depth-estimation": "LiheYoung/depth-anything-small-hf",
  "velocity-estimation": "raft-large",
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

    if (!backendUrl) {
      throw new Error("INFERENCE_BACKEND_URL is required. This project uses notebook-aligned GPU backend inference.");
    }

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
  } catch (error: any) {
    json(res, 500, { error: error?.message || "Inference request failed" });
  }
}
