import { listDatasets, saveUploadedDataset } from "../../../lib/dataset";

export async function GET() {
  const datasets = await listDatasets();
  return Response.json({ datasets });
}

export async function POST(request) {
  const formData = await request.formData();
  const file = formData.get("file");

  if (!file || typeof file === "string") {
    return Response.json({ error: "A .jsonl file is required." }, { status: 400 });
  }

  if (!file.name.endsWith(".jsonl")) {
    return Response.json({ error: "Only .jsonl files are supported." }, { status: 400 });
  }

  const buffer = Buffer.from(await file.arrayBuffer());
  const dataset = await saveUploadedDataset(file.name, buffer);
  return Response.json({ dataset });
}
