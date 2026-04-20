import { addSuggestion, getDatasetByKey, getSuggestions } from "../../../lib/dataset";

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const datasetKey = searchParams.get("dataset");
  const rowNumber = Number(searchParams.get("row"));

  if (!datasetKey || !rowNumber) {
    return Response.json({ error: "dataset and row are required" }, { status: 400 });
  }

  const suggestions = await getSuggestions(datasetKey, rowNumber);
  return Response.json({ suggestions });
}

export async function POST(request) {
  const body = await request.json();
  const datasetKey = body.dataset;
  const rowNumber = Number(body.row);
  const suggestion = body.suggestion;

  if (!datasetKey || !rowNumber || !suggestion) {
    return Response.json({ error: "dataset, row, and suggestion are required" }, { status: 400 });
  }

  const datasetMeta = await getDatasetByKey(datasetKey);
  const suggestions = await addSuggestion(datasetKey, datasetMeta, rowNumber, suggestion);
  return Response.json({ suggestions });
}
