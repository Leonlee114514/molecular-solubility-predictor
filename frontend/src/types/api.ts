// API types mirroring backend/schemas.py (camelCase on the wire).

export type ModelMode = "auto" | "rf" | "gnn" | "ensemble";

export interface HealthModel {
  available: boolean;
  loaded: boolean;
}

export interface HealthResponse {
  status: string;
  models: {
    rf: HealthModel;
    pka: HealthModel;
    gnn: HealthModel;
    ood: HealthModel;
  };
}

export interface PredictionResult {
  smiles: string;
  features: Record<string, number>;
  logS_rf: number | null;
  logS_gnn: number | null;
  logS_final: number;
  model_selected: string;
  model_used: string;
  model_disagreement: boolean;
  pka: number | null;
  pka_kind: "acid" | "base" | "amphoteric" | null;
  ood_risk: "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN" | null;
  ood_score: number | null;
  ood_max_tanimoto: number | null;
  ood_out_of_range: string[];
  ood_extreme: string[];
  shap_values: number[] | null;
  shap_names: string[] | null;
  shap_base_value: number | null;
}

export interface MoleculeEntry {
  name: string;
  smiles: string;
}

export interface PubchemResult {
  smiles: string | null;
  status: string | null;
}

export interface SearchResponse {
  exact: MoleculeEntry[];
  fuzzy: MoleculeEntry[];
  pubchem: PubchemResult | null;
}

export interface ParseFileResponse {
  smiles: string;
  formula: string;
  mw: number;
}

export interface MolInfo {
  formula: string;
  mw: number;
}

export interface LipinskiResult {
  rules: [string, string, boolean, string][];
  passed: number;
  violations: number;
  is_druglike: boolean;
  interpretation: string;
}

export interface DruglikenessResult {
  qed: number;
  qed_color: string;
  qed_level: string;
  sascore: number;
  sa_color: string;
  sa_level: string;
  fsp3: number;
  fsp3_color: string;
  fsp3_level: string;
  n_carbons: number;
  n_sp3: number;
}

export interface AdmetResult {
  absorption: string;
  distribution: { summary: string; vd_estimate: string; ppb: string };
  metabolism: { summary: string; cyp_enzymes: string };
  excretion: { summary: string; route: string };
  toxicity: [string, string][];
}

export type PharmaAnalysisKey =
  | "strong_acid"
  | "mid_acid"
  | "strong_base"
  | "weak_base"
  | "amphoteric";

export interface IonizationPoint {
  env: string;
  ph: number;
  pct: number;
}

export interface AnalysisResponse {
  pka_factors: Record<string, number> | null;
  lipinski: LipinskiResult;
  druglikeness: DruglikenessResult | null;
  admet: AdmetResult;
  ionization: IonizationPoint[] | null;
  pharma_analysis: PharmaAnalysisKey | null;
  linkage: string | null;
}

export interface ExplainRequestBody {
  smiles: string;
  prediction: number;
  features: Record<string, number>;
  shap_features?: string[] | null;
  shap_values?: number[] | null;
  pka_value?: number | null;
  pka_type?: string | null;
  lang: string;
}

export interface ExplainResponse {
  markdown: string;
  cached: boolean;
}

export interface GnnExplainResponse {
  bond_importance: [number, number, number][];
  bond_details: [number, number, number, string][];
  feature_importance: number[];
  elapsed: number;
}

export interface Mol3dResponse {
  molblock: string;
}

// ── Batch prediction ──

export interface BatchStartResponse {
  task_id: string;
  count: number;
  smiles_column?: string;
}

export interface BatchRowError {
  smiles: string;
  error: string;
}

export type BatchRow = PredictionResult | BatchRowError;

export function isBatchRowError(row: BatchRow): row is BatchRowError {
  return typeof (row as BatchRowError).error === "string";
}

export interface BatchStatusResponse {
  status: "running" | "done" | "error";
  progress: { done: number; total: number };
  results: BatchRow[] | null;
  error: string | null;
}
