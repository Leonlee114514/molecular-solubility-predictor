import type { TFunction } from "i18next";

/**
 * Translate raw descriptor keys (MolWt, LogP, ..., "MorganFP") arriving from
 * SHAP / OOD payloads into localized display labels.
 *
 * The 13 descriptor labels live in the dumped locale JSONs as ood.descriptor.*;
 * MorganFP has its own key. Unknown names pass through unchanged.
 */
export function descriptorLabel(t: TFunction, name: string): string {
  if (name === "MorganFP") return t("model.shap.morgan_fp");
  const key = `ood.descriptor.${name}`;
  const value = t(key);
  return value === key ? name : value;
}

/** The 37 atom-feature dimension names used by the GNN encoder (technical, not translated). */
export const GNN_FEATURE_NAMES: string[] = [
  "AtomicNum_H", "AtomicNum_C", "AtomicNum_N", "AtomicNum_O",
  "AtomicNum_F", "AtomicNum_P", "AtomicNum_S", "AtomicNum_Cl",
  "AtomicNum_Br", "AtomicNum_I", "AtomicNum_Other",
  "Degree_0", "Degree_1", "Degree_2", "Degree_3", "Degree_4", "Degree_5+",
  "Charge_n2", "Charge_n1", "Charge_0", "Charge_p1", "Charge_p2",
  "Hyb_SP", "Hyb_SP2", "Hyb_SP3", "Hyb_SP3D", "Hyb_Other",
  "IsAromatic",
  "Hs_0", "Hs_1", "Hs_2", "Hs_3", "Hs_4+",
  "Chiral_None", "Chiral_R", "Chiral_S",
  "IsInRing",
];
