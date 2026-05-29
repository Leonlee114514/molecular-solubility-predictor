"""
Download aqueous solubility data from ChEMBL.
Maps ug/mL values → logS (mol/L), filters for aqueous measurements only.
"""

import urllib.request, json, time, csv, os, sys, math, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 100
MAX_PAGES = 50  # safety limit (5000 records, enough for our needs)
SLEEP = 0.3     # be nice to the API

def api_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "DisSolve/1.0"})
    with urllib.request.urlopen(req, timeout=30) as f:
        return json.loads(f.read())

def get_molecule_smiles(chembl_id):
    """Get canonical SMILES for a ChEMBL molecule ID."""
    url = f"{BASE}/molecule/{chembl_id}.json"
    try:
        data = api_get(url)
        # Try standard SMILES first, then various computed fields
        if data.get("molecule_structures"):
            s = data["molecule_structures"]
            for key in ("canonical_smiles", "smiles", "standard_smiles"):
                if s.get(key):
                    return s[key]
        return None
    except:
        return None

def convert_to_logS(value, units, relation):
    """Convert solubility value to logS (mol/L)."""
    if value is None or value == "":
        return None
    val = float(value)
    if val <= 0:
        return None

    # Convert ug/mL to mol/L, then to log10
    if units == "ug.mL-1":
        # Need MW to convert - handled by caller
        return ("ugmL", val)
    elif units == "mg.mL-1":
        val_ug = val * 1000
        return ("ugmL", val_ug)
    elif units == "mg.L-1":
        val_ug = val / 1000  # mg/L = ug/mL
        return ("ugmL", val_ug)
    elif units == "nM":
        # direct molar concentration
        mol_L = val * 1e-9
        return ("logS", math.log10(mol_L) if mol_L > 0 else None)
    elif units == "uM":
        mol_L = val * 1e-6
        return ("logS", math.log10(mol_L) if mol_L > 0 else None)
    elif units == "mM":
        mol_L = val * 1e-3
        return ("logS", math.log10(mol_L) if mol_L > 0 else None)
    else:
        return ("unknown", val)

def main():
    print("Fetching ChEMBL solubility data...")
    print(f"Page size: {PAGE_SIZE}, max pages: {MAX_PAGES}")

    # Step 1: find all assays with aqueous solubility
    # Search for assays with descriptions mentioning aqueous/water solubility
    total_fetched = 0
    activities = []

    for page in range(1, MAX_PAGES + 1):
        offset = (page - 1) * PAGE_SIZE
        url = f"{BASE}/activity.json?standard_type=Solubility&limit={PAGE_SIZE}&offset={offset}"
        try:
            data = api_get(url)
        except Exception as e:
            print(f"  Page {page}: error - {e}, retrying...")
            time.sleep(2)
            try:
                data = api_get(url)
            except:
                print(f"  Page {page}: failed, stopping")
                break

        items = data.get("activities", [])
        if not items:
            break

        for a in items:
            activities.append({
                "mol_id": a.get("molecule_chembl_id"),
                "assay_id": a.get("assay_chembl_id"),
                "doc_id": a.get("document_chembl_id"),
                "value": a.get("standard_value"),
                "units": a.get("standard_units"),
                "relation": a.get("standard_relation", "="),
                "type": a.get("standard_type"),
            })

        total_fetched += len(items)
        if page % 5 == 0:
            print(f"  Fetched {total_fetched} activities (page {page})...")
        time.sleep(SLEEP)

    print(f"\nTotal activities fetched: {total_fetched}")

    # Step 2: Filter to ug.mL-1 (most common solubility unit in ChEMBL)
    # and get unique molecules
    print(f"\nProcessing: deduplicating and converting to logS...")

    # Deduplicate by molecule (keep mean of multiple measurements)
    mol_data = {}  # chembl_id -> list of (converted_val_or_mode, raw_val, units)
    for a in activities:
        mid = a["mol_id"]
        if not mid:
            continue

        conv = convert_to_logS(a["value"], a["units"], a["relation"])
        if conv is None:
            continue

        mode, val = conv
        if mode == "ugmL":
            # Store temporarily - we'll need MW to convert
            mol_data.setdefault(mid, []).append(val)
        elif mode == "logS":
            # Already converted - store as (chembl_id, smiles, logS)
            mol_data.setdefault(mid, []).append(val)

    print(f"Unique molecules (raw data): {len(mol_data)}")

    # Step 3: Get SMILES for each molecule and convert ug/mL to logS
    print(f"\nFetching molecule SMILES and computing logS...")
    results = []
    batch = 0
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    for i, (mid, vals) in enumerate(mol_data.items()):
        # Get SMILES
        smiles = get_molecule_smiles(mid)
        if not smiles:
            if (i+1) % 100 == 0:
                print(f"  ({i+1}/{len(mol_data)}) processed...")
            continue

        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Get MW
        mw = Descriptors.MolWt(mol)

        # Compute logS from ug/mL values
        logS_vals = []
        if isinstance(vals[0], (int, float)):
            # Already logS
            logS_vals = [v for v in vals if v is not None]
        else:
            # It's ug/mL values that need MW conversion
            ugmL_vals = [v for v in vals if v is not None and v > 0]
            for ug in ugmL_vals:
                try:
                    mol_L = (ug / 1000000.0) / mw  # ug/mL -> g/mL -> mol/L
                    if mol_L > 0:
                        logS_vals.append(math.log10(mol_L))
                except:
                    pass

        if not logS_vals:
            continue

        # Take mean of multiple measurements
        logS = sum(logS_vals) / len(logS_vals)

        # Filter extreme values
        if logS < -12 or logS > 3:
            continue

        results.append({"chembl_id": mid, "SMILES": smiles, "logS": round(logS, 4)})

        if (i+1) % 200 == 0:
            print(f"  ({i+1}/{len(mol_data)}) → {len(results)} valid molecules")
            batch += 1
            if batch % 5 == 0:
                time.sleep(SLEEP)

    print(f"\nTotal valid molecules: {len(results)}")

    # Step 4: Save to CSV
    path = os.path.join(os.path.dirname(__file__), "..", "chembl_solubility.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chembl_id", "SMILES", "logS"])
        for r in results:
            w.writerow([r["chembl_id"], r["SMILES"], r["logS"]])

    print(f"Saved to {path}")

    # Stats
    logS_vals = [r["logS"] for r in results]
    print(f"\nStats:")
    print(f"  Count: {len(results)}")
    print(f"  Mean logS: {sum(logS_vals)/len(logS_vals):.3f}")
    print(f"  Min logS: {min(logS_vals):.3f}")
    print(f"  Max logS: {max(logS_vals):.3f}")

if __name__ == "__main__":
    main()
