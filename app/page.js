import { loadDataset, listDatasets } from "../lib/dataset";
import ReviewWorkspace from "../components/review-workspace";

export default async function Page({ searchParams }) {
  const datasetKey = searchParams?.dataset || "train";
  const sampleIndex = Number(searchParams?.sample || 0);
  const { dataset, samples, totalLoaded, totalRows } = await loadDataset(datasetKey);
  const datasets = await listDatasets();
  const selectedIndex = Number.isNaN(sampleIndex)
    ? 0
    : Math.min(Math.max(sampleIndex, 0), Math.max(samples.length - 1, 0));

  return (
    <main className="min-h-screen bg-shell-bg bg-dots [background-size:18px_18px]">
      <ReviewWorkspace
        datasets={datasets}
        dataset={dataset}
        samples={samples}
        selectedIndex={selectedIndex}
        totalLoaded={totalLoaded}
        totalRows={totalRows}
      />
    </main>
  );
}
