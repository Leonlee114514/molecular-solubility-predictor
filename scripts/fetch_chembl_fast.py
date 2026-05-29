"""
Faster ChEMBL solubility data fetcher using batch molecule queries.
Reduces ~3600 HTTP requests to ~40.
"""

import requests, json, time, csv, os, sys, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 100
MAX_PAGES = 50
BATCH_SIZE = 100  # molecules per batch query

def api_get(url, timeout=30):
    r = requests.get(url, headers={"User-Agent": "DisSolve/2.0"}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_activities():
    """Fetch all solubility activities from ChEMBL."""
    activities = []
    for page in range(1, MAX_PAGES + 1):
        offset = (page - 1) * PAGE_SIZE
        url = f"{BASE}/activity.json?standard_type=Solubility&limit={PAGE_SIZE}&offset={offset}"
        try:
            data = api_get(url)
            items = data.get("activities", [])
            if not items:
                break
            activities.extend(items)
            if page % 10 == 0:
                print(f"  Fetched {len(activities)} activities (page {page})...")
        except Exception as e:
            print(f"  Page {page} error: {e}, retrying...")
            time.sleep(2)
            try:
                data = api_get(url)
                activities.extend(data.get("activities", []))
            except:
                print(f"  Page {page} failed, stopping")
                break
        time.sleep(0.15)  # be nice to API
    return activities

def fetch_molecules_batch(chembl_ids, batch_size=BATCH_SIZE):
    """Fetch molecule records in batches, returning {chembl_id: smiles}."""
    smiles_map = {}
    ids_list = list(chembl_ids)
    total = len(ids_list)

    for i in range(0, total, batch_size):
        batch = ids_list[i:i + batch_size]
        ids_str = ",".join(batch)
        url = f"{BASE}/molecule.json?molecule_chembl_id__in={ids_str}&limit={batch_size}"
        try:
            data = api_get(url)
            for mol in data.get("molecules", []):
                mid = mol.get("molecule_chembl_id")
                smi = None
                structures = mol.get("molecule_structures")
                if structures:
                    smi = structures.get("canonical_smiles") or structures.get("smiles")
                if smi:
                    smiles_map[mid] = smi
            if (i // batch_size) % 5 == 0:
                print(f"  SMILES: {len(smiles_map)}/{total} ({min(i+batch_size, total)}/{total})")
        except Exception as e:
            print(f"  Batch {i//batch_size}: error {e}, retrying smaller...")
            # Fall back to individual queries for this batch
            for mid in batch:
                try:
                    d = api_get(f"{BASE}/molecule/{mid}.json")
                    structures = d.get("molecule_structures", {})
                    smi = structures.get("canonical_smiles") or structures.get("smiles")
                    if smi:
                        smiles_map[mid] = smi
                except:
                    pass
                time.sleep(0.05)
        time.sleep(0.15)

    return smiles_map

def main():
    print("Fetching ChEMBL solubility data (batch-optimized)...")

    # Step 1: Fetch activities
    print(f"\nStep 1: Fetching activities (max {MAX_PAGES} pages)...")
    activities = fetch_activities()
    print(f"  Total activities: {len(activities)}")

    if not activities:
        print("  No activities found, exiting.")
        return

    # Step 2: Extract unique molecule IDs and convert units
    print("\nStep 2: Processing activities...")
    mol_data = {}  # chembl_id -> [(unit_type, value)]

    for a in activities:
        mid = a.get("molecule_chembl_id")
        if not mid:
            continue
        value = a.get("standard_value")
        units = a.get("standard_units")
        relation = a.get("standard_relation", "=")

        if value is None or value == "":
            continue
        try:
            val = float(value)
        except (ValueError, TypeError):
            continue
        if val <= 0:
            continue

        # Classify units
        if units in ("ug.mL-1", "mg.mL-1", "mg.L-1"):
            mol_data.setdefault(mid, []).append(("mass_conc", val, units))
        elif units in ("nM", "uM", "mM"):
            mol_data.setdefault(mid, []).append(("molar_conc", val, units))
        # Skip other units

    print(f"  Unique molecules with usable data: {len(mol_data)}")

    # Step 3: Batch fetch molecule SMILES
    print(f"\nStep 3: Batch fetching SMILES for {len(mol_data)} molecules...")
    smiles_map = fetch_molecules_batch(mol_data.keys())
    print(f"  Got SMILES for {len(smiles_map)} molecules")

    # Step 4: Convert to logS
    print("\nStep 4: Computing logS values...")
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    results = []
    for mid, measurements in mol_data.items():
        smiles = smiles_map.get(mid)
        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mw = Descriptors.MolWt(mol)

        logS_vals = []
        for mtype, val, unit in measurements:
            if mtype == "molar_conc":
                if unit == "nM":
                    mol_L = val * 1e-9
                elif unit == "uM":
                    mol_L = val * 1e-6
                elif unit == "mM":
                    mol_L = val * 1e-3
                else:
                    continue
                if mol_L > 0:
                    logS_vals.append(math.log10(mol_L))
            elif mtype == "mass_conc":
                if unit == "ug.mL-1":
                    ug_mL = val
                elif unit == "mg.mL-1":
                    ug_mL = val * 1000
                elif unit == "mg.L-1":
                    ug_mL = val / 1000
                else:
                    continue
                if ug_mL > 0:
                    mol_L = (ug_mL / 1000000.0) / mw
                    if mol_L > 0:
                        logS_vals.append(math.log10(mol_L))

        if not logS_vals:
            continue

        logS = sum(logS_vals) / len(logS_vals)
        if logS < -12 or logS > 3:
            continue

        results.append({"chembl_id": mid, "SMILES": smiles, "logS": round(logS, 4)})

    print(f"  Valid molecules with logS: {len(results)}")

    # Step 5: Save
    path = os.path.join(os.path.dirname(__file__), "..", "chembl_solubility.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chembl_id", "SMILES", "logS"])
        for r in results:
            w.writerow([r["chembl_id"], r["SMILES"], r["logS"]])

    print(f"\nSaved {len(results)} molecules to {path}")

    if results:
        logS_vals = [r["logS"] for r in results]
        print(f"  Mean logS: {np.mean(logS_vals):.3f}")
        print(f"  Min logS:  {np.min(logS_vals):.3f}")
        print(f"  Max logS:  {np.max(logS_vals):.3f}")

if __name__ == "__main__":
    import numpy as np
    main()
