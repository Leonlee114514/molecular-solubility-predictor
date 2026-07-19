import type {
  AnalysisResponse,
  BatchStartResponse,
  BatchStatusResponse,
  ExplainRequestBody,
  ExplainResponse,
  GnnExplainResponse,
  HealthResponse,
  Mol3dResponse,
  ModelMode,
  MolInfo,
  MoleculeEntry,
  ParseFileResponse,
  PredictionResult,
  SearchResponse,
} from "@/types/api";

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(path, init);
  } catch (err) {
    // Network-level failure (backend down, CORS, etc.)
    throw new ApiError(0, err instanceof Error ? err.message : "network error");
  }
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = (await resp.json()) as { detail?: string };
      if (body && typeof body.detail === "string") {
        detail = body.detail;
      }
    } catch {
      // non-JSON error body; keep default detail
    }
    throw new ApiError(resp.status, detail);
  }
  return (await resp.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),

  predict: (smiles: string, mode: ModelMode, lang: string) =>
    request<PredictionResult>("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles, mode, lang }),
    }),

  listMolecules: async () => {
    const resp = await request<{ molecules: MoleculeEntry[] }>("/api/molecules");
    return resp.molecules;
  },

  searchMolecules: (q: string, lang: string, pubchem: boolean) =>
    request<SearchResponse>(
      `/api/molecules/search?q=${encodeURIComponent(q)}&lang=${encodeURIComponent(lang)}&pubchem=${pubchem}`,
    ),

  parseFile: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<ParseFileResponse>("/api/mol/parse-file", {
      method: "POST",
      body: form,
    });
  },

  mol2dUrl: (smiles: string, bonds?: string) =>
    `/api/mol/2d?smiles=${encodeURIComponent(smiles)}${bonds ? `&bonds=${encodeURIComponent(bonds)}` : ""}`,

  molInfo: (smiles: string) =>
    request<MolInfo>(`/api/mol/info?smiles=${encodeURIComponent(smiles)}`),

  mol3d: (smiles: string) =>
    request<Mol3dResponse>(`/api/mol/3d?smiles=${encodeURIComponent(smiles)}`),

  analysis: (smiles: string, pka: number | null, logs: number | null, lang: string) =>
    request<AnalysisResponse>("/api/analysis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles, pka, logs, lang }),
    }),

  explain: (body: ExplainRequestBody) =>
    request<ExplainResponse>("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  gnnExplain: (smiles: string) =>
    request<GnnExplainResponse>("/api/gnn-explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles }),
    }),

  predictBatchFile: (file: File, mode: ModelMode) => {
    const form = new FormData();
    form.append("file", file);
    form.append("mode", mode);
    return request<BatchStartResponse>("/api/predict/batch-file", {
      method: "POST",
      body: form,
    });
  },

  predictBatchList: (smilesList: string[], mode: ModelMode) =>
    request<BatchStartResponse>("/api/predict/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles_list: smilesList, mode }),
    }),

  batchStatus: (taskId: string) =>
    request<BatchStatusResponse>(`/api/predict/batch/${taskId}`),
};
