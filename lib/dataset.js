import fs from "fs/promises";
import path from "path";

const OUTPUT_DIR = "output";
const UPLOADS_DIR = "data/uploads";
const REVIEWS_DIR = "data/reviews";

const BUILTIN_DATASETS = {
  train: {
    key: "train",
    label: "Merged Train",
    file: "output/merged_train.jsonl",
    kind: "builtin"
  },
  test: {
    key: "test",
    label: "Merged Test",
    file: "output/merged_test.jsonl",
    kind: "builtin"
  }
};

const ISSUE_RULES = [
  {
    id: "assistant-id",
    title: "Assistant should not emit call ids",
    test(sample) {
      return sample.messages.some((message) => {
        if (message.role !== "assistant") return false;
        try {
          const parsed = JSON.parse(message.content);
          return parsed.type === "tool_call" && ("id" in parsed || "tool_call_id" in parsed);
        } catch {
          return false;
        }
      });
    }
  },
  {
    id: "absolute-path",
    title: "Ungrounded absolute path guess",
    test(sample) {
      const userText = sample.messages
        .filter((message) => message.role === "user")
        .map((message) => message.content.toLowerCase())
        .join(" ");

      return sample.messages.some((message) => {
        if (message.role !== "assistant") return false;
        try {
          const parsed = JSON.parse(message.content);
          if (parsed.type !== "tool_call") return false;
          return Object.values(parsed.arguments || {}).some((value) => {
            return (
              typeof value === "string" &&
              value.startsWith("/Users/sridhar/project") &&
              !userText.includes("/users/") &&
              !userText.includes("absolute path") &&
              !userText.includes("working directory")
            );
          });
        } catch {
          return false;
        }
      });
    }
  },
  {
    id: "config-folder",
    title: "Config folder should not be hardcoded",
    test(sample) {
      const userText = sample.messages
        .filter((message) => message.role === "user")
        .map((message) => message.content.toLowerCase())
        .join(" ");
      if (!userText.includes("config folder")) return false;
      return sample.messages.some((message) => {
        if (message.role !== "assistant") return false;
        try {
          const parsed = JSON.parse(message.content);
          return (
            parsed.type === "tool_call" &&
            parsed.tool_name === "File_List" &&
            parsed.arguments?.directory?.startsWith?.("/Users/")
          );
        } catch {
          return false;
        }
      });
    }
  }
];

function safeJsonParse(value, fallback = null) {
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

async function ensureDir(relativeDir) {
  await fs.mkdir(path.join(process.cwd(), relativeDir), { recursive: true });
}

async function ensureReviewStorage() {
  await ensureDir(UPLOADS_DIR);
  await ensureDir(REVIEWS_DIR);
}

function reviewFileForDataset(datasetKey) {
  return path.join(process.cwd(), REVIEWS_DIR, `${datasetKey}.json`);
}

async function readJsonIfExists(filePath, fallback) {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

async function createReviewShell(dataset, rowCount = 0) {
  const review = {
    datasetKey: dataset.key,
    datasetLabel: dataset.label,
    datasetFile: dataset.file,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    rowCount,
    rows: {}
  };
  await fs.writeFile(reviewFileForDataset(dataset.key), JSON.stringify(review, null, 2));
  return review;
}

export async function ensureReviewFile(dataset, rowCount = 0) {
  await ensureReviewStorage();
  const filePath = reviewFileForDataset(dataset.key);
  const existing = await readJsonIfExists(filePath, null);
  if (existing) {
    if (typeof rowCount === "number" && rowCount > (existing.rowCount || 0)) {
      existing.rowCount = rowCount;
      existing.updatedAt = new Date().toISOString();
      await fs.writeFile(filePath, JSON.stringify(existing, null, 2));
    }
    return existing;
  }
  return createReviewShell(dataset, rowCount);
}

export async function getSuggestions(datasetKey, rowNumber) {
  const review = await readJsonIfExists(reviewFileForDataset(datasetKey), null);
  if (!review) return [];
  const row = review.rows?.[String(rowNumber)];
  return Array.isArray(row?.suggestions) ? row.suggestions : [];
}

export async function addSuggestion(datasetKey, datasetMeta, rowNumber, suggestion) {
  const clean = suggestion.trim();
  if (!clean) {
    throw new Error("Suggestion cannot be empty.");
  }

  const dataset = datasetMeta || { key: datasetKey, label: datasetKey, file: "" };
  const review = await ensureReviewFile(dataset);
  const rowKey = String(rowNumber);
  if (!review.rows[rowKey]) {
    review.rows[rowKey] = {
      rowNumber,
      suggestions: []
    };
  }
  review.rows[rowKey].suggestions.push(clean);
  review.updatedAt = new Date().toISOString();
  await fs.writeFile(reviewFileForDataset(datasetKey), JSON.stringify(review, null, 2));
  return review.rows[rowKey].suggestions;
}

export async function listDatasets() {
  await ensureReviewStorage();
  const items = [...Object.values(BUILTIN_DATASETS)];

  const uploadDir = path.join(process.cwd(), UPLOADS_DIR);
  const files = await fs.readdir(uploadDir).catch(() => []);
  for (const file of files) {
    if (!file.endsWith(".jsonl")) continue;
    const key = slugify(file.replace(/\.jsonl$/i, ""));
    items.push({
      key,
      label: file.replace(/\.jsonl$/i, ""),
      file: path.join(UPLOADS_DIR, file),
      kind: "uploaded"
    });
  }

  return items;
}

export async function getDatasetByKey(datasetKey = "train") {
  const datasets = await listDatasets();
  return datasets.find((dataset) => dataset.key === datasetKey) || datasets[0] || BUILTIN_DATASETS.train;
}

export async function loadDataset(datasetKey = "train", limit = 120) {
  const selected = await getDatasetByKey(datasetKey);
  const filePath = path.join(process.cwd(), selected.file);
  const content = await fs.readFile(filePath, "utf8");
  const allLines = content.trim().split("\n");
  const lines = allLines.slice(0, limit);
  const review = await ensureReviewFile(selected, allLines.length);

  const samples = lines.map((line, index) => {
    const sample = JSON.parse(line);
    const issueFlags = ISSUE_RULES.filter((rule) => rule.test(sample)).map((rule) => rule.title);
    const parsedMessages = sample.messages.map((message, messageIndex) => {
      const parsedContent =
        typeof message.content === "string" && message.content.startsWith("{")
          ? safeJsonParse(message.content)
          : null;
      return {
        id: `${index}-${messageIndex}`,
        ...message,
        parsedContent
      };
    });

    return {
      id: `${datasetKey}-${index}`,
      index,
      rowNumber: index + 1,
      datasetKey,
      metadata: sample.metadata,
      localization: sample.localization,
      tools: sample.tools,
      raw: sample,
      issueFlags,
      suggestions: review.rows?.[String(index + 1)]?.suggestions || [],
      messages: parsedMessages,
      title:
        sample.messages.find((message) => message.role === "user")?.content ||
        `Sample ${index + 1}`
    };
  });

  return {
    dataset: {
      ...selected,
      reviewFile: path.relative(process.cwd(), reviewFileForDataset(selected.key))
    },
    totalLoaded: samples.length,
    totalRows: allLines.length,
    samples
  };
}

export async function saveUploadedDataset(fileName, fileBuffer) {
  await ensureReviewStorage();
  const safeName = fileName.replace(/[^a-zA-Z0-9._-]/g, "_");
  const stampedName = `${Date.now()}-${safeName}`;
  const relativeFile = path.join(UPLOADS_DIR, stampedName);
  const absoluteFile = path.join(process.cwd(), relativeFile);
  await fs.writeFile(absoluteFile, fileBuffer);

  const dataset = {
    key: slugify(stampedName.replace(/\.jsonl$/i, "")),
    label: stampedName.replace(/\.jsonl$/i, ""),
    file: relativeFile,
    kind: "uploaded"
  };

  const raw = await fs.readFile(absoluteFile, "utf8");
  const rowCount = raw.trim() ? raw.trim().split("\n").length : 0;
  await createReviewShell(dataset, rowCount);
  return dataset;
}
