"""
Build supplementary logS dataset to fill structural gaps in training data.
Targets: pure hydrocarbons, simple heteroatom compounds, steroids.
"""

import csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rdkit import Chem

CURATED = [
    # Alkanes
    ("Methane", "C", -2.19),
    ("Ethane", "CC", -2.53),
    ("Propane", "CCC", -2.76),
    ("Butane", "CCCC", -3.01),
    ("Pentane", "CCCCC", -3.30),
    ("Hexane", "CCCCCC", -3.71),
    ("Heptane", "CCCCCCC", -4.10),
    ("Octane", "CCCCCCCC", -4.47),
    ("Nonane", "CCCCCCCCC", -4.82),
    ("Decane", "CCCCCCCCCC", -5.16),
    ("Isobutane", "CC(C)C", -2.94),
    ("Isopentane", "CCC(C)C", -3.21),
    ("Neopentane", "CC(C)(C)C", -3.13),
    # Cycloalkanes
    ("Cyclopropane", "C1CC1", -1.66),
    ("Cyclobutane", "C1CCC1", -2.28),
    ("Cyclopentane", "C1CCCC1", -2.91),
    ("Cyclohexane", "C1CCCCC1", -3.10),
    ("Cycloheptane", "C1CCCCCC1", -3.62),
    ("Cyclooctane", "C1CCCCCCC1", -4.00),
    ("Methylcyclohexane", "CC1CCCCC1", -3.46),
    ("Decalin", "C1CCC2CCCCC2C1", -5.00),
    # Alkenes & Alkynes
    ("Ethylene", "C=C", -1.88),
    ("Propene", "CC=C", -2.22),
    ("1-Butene", "CCC=C", -2.57),
    ("1-Hexene", "CCCCC=C", -3.02),
    ("Cyclohexene", "C1CCC=CC1", -2.75),
    # Aromatic hydrocarbons
    ("Benzene", "c1ccccc1", -1.64),
    ("Toluene", "Cc1ccccc1", -2.30),
    ("Ethylbenzene", "CCc1ccccc1", -2.75),
    ("o-Xylene", "Cc1ccccc1C", -2.62),
    ("m-Xylene", "Cc1cccc(C)c1", -2.68),
    ("p-Xylene", "Cc1ccc(C)cc1", -2.73),
    ("Styrene", "c1ccccc1C=C", -2.82),
    ("Biphenyl", "c1ccc(cc1)c2ccccc2", -4.03),
    ("Naphthalene", "c1ccc2ccccc2c1", -3.66),
    ("Anthracene", "c1ccc2cc3ccccc3cc2c1", -5.50),
    ("Phenanthrene", "c1ccc2c(c1)ccc3ccccc32", -4.67),
    ("Pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34", -5.76),
    # Halogenated hydrocarbons
    ("Chloromethane", "CCl", -1.28),
    ("Dichloromethane", "C(Cl)Cl", -1.16),
    ("Chloroform", "C(Cl)(Cl)Cl", -1.46),
    ("Carbon tetrachloride", "C(Cl)(Cl)(Cl)Cl", -2.20),
    ("Chloroethane", "CCCl", -1.67),
    ("1,2-Dichloroethane", "C(CCl)Cl", -1.68),
    ("1,1,1-Trichloroethane", "CC(Cl)(Cl)Cl", -2.39),
    ("Bromomethane", "CBr", -1.47),
    ("Bromoethane", "CCBr", -1.90),
    ("1-Bromopropane", "CCCBr", -2.20),
    ("Iodomethane", "CI", -1.66),
    ("Iodoethane", "CCI", -2.10),
    ("Fluorobenzene", "c1ccc(cc1)F", -1.81),
    ("Chlorobenzene", "c1ccc(cc1)Cl", -2.54),
    ("Bromobenzene", "c1ccc(cc1)Br", -2.77),
    ("Iodobenzene", "c1ccc(cc1)I", -3.36),
    ("1,2-Dichlorobenzene", "Clc1ccccc1Cl", -3.10),
    ("1,4-Dichlorobenzene", "Clc1ccc(cc1)Cl", -3.38),
    ("Hexachlorobenzene", "Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl", -6.00),
    # Alcohols
    ("Methanol", "CO", 0.77),
    ("Ethanol", "CCO", 0.97),
    ("1-Propanol", "CCCO", 0.55),
    ("2-Propanol", "CC(C)O", 0.40),
    ("1-Butanol", "CCCCO", 0.07),
    ("2-Butanol", "CCC(C)O", 0.00),
    ("tert-Butanol", "CC(C)(C)O", 0.35),
    ("1-Pentanol", "CCCCCO", -0.35),
    ("1-Hexanol", "CCCCCCO", -0.73),
    ("1-Octanol", "CCCCCCCCO", -1.49),
    ("Phenol", "c1ccc(cc1)O", -0.13),
    ("Benzyl alcohol", "c1ccc(cc1)CO", -0.28),
    ("Ethylene glycol", "C(CO)O", 0.62),
    ("Glycerol", "C(C(CO)O)O", 0.85),
    # Ethers
    ("Dimethyl ether", "COC", 0.42),
    ("Diethyl ether", "CCOCC", -0.40),
    ("Methyl tert-butyl ether", "COC(C)(C)C", -0.06),
    ("THF", "C1CCOC1", 0.31),
    ("1,4-Dioxane", "C1COCCO1", 0.40),
    ("Anisole", "COc1ccccc1", -1.36),
    # Aldehydes & Ketones
    ("Formaldehyde", "C=O", 0.35),
    ("Acetaldehyde", "CC=O", 0.59),
    ("Propionaldehyde", "CCC=O", 0.17),
    ("Butyraldehyde", "CCCC=O", -0.20),
    ("Acetone", "CC(=O)C", 0.52),
    ("2-Butanone", "CCC(=O)C", 0.09),
    ("3-Pentanone", "CCC(=O)CC", -0.33),
    ("Cyclohexanone", "C1CCC(=O)CC1", -0.17),
    ("Acetophenone", "CC(=O)c1ccccc1", -1.94),
    ("Benzophenone", "O=C(c1ccccc1)c2ccccc2", -3.28),
    # Carboxylic acids
    ("Formic acid", "C(=O)O", 1.10),
    ("Acetic acid", "CC(=O)O", 0.80),
    ("Propionic acid", "CCC(=O)O", 0.33),
    ("Butyric acid", "CCCC(=O)O", -0.01),
    ("Valeric acid", "CCCCC(=O)O", -0.35),
    ("Caproic acid", "CCCCCC(=O)O", -0.80),
    ("Benzoic acid", "c1ccc(cc1)C(=O)O", -1.54),
    ("Salicylic acid", "c1ccc(c(c1)C(=O)O)O", -1.07),
    # Esters
    ("Methyl acetate", "CC(=O)OC", 0.49),
    ("Ethyl acetate", "CC(=O)OCC", 0.06),
    ("Butyl acetate", "CCCCOC(=O)C", -0.56),
    ("Methyl benzoate", "COC(=O)c1ccccc1", -1.83),
    # Amines
    ("Methylamine", "CN", 1.10),
    ("Ethylamine", "CCN", 0.62),
    ("Propylamine", "CCCN", 0.17),
    ("Butylamine", "CCCCN", -0.24),
    ("Dimethylamine", "CNC", 0.60),
    ("Trimethylamine", "CN(C)C", 0.10),
    ("Aniline", "c1ccc(cc1)N", -0.25),
    ("N-Methylaniline", "CNc1ccccc1", -0.67),
    ("N,N-Dimethylaniline", "CN(C)c1ccccc1", -1.07),
    ("Pyridine", "c1ccncc1", 0.65),
    ("Piperidine", "C1CCNCC1", 0.48),
    ("Triethylamine", "CCN(CC)CC", -0.53),
    # Simple heterocycles
    ("Furan", "c1ccco1", 0.40),
    ("Thiophene", "c1ccsc1", -0.51),
    ("Pyrrole", "c1cc[nH]c1", 0.30),
    ("Imidazole", "c1cnc[nH]1", 1.06),
    ("Thiazole", "c1cscn1", 0.18),
    ("Pyrimidine", "c1cncnc1", 0.43),
    ("Quinoline", "c1ccc2c(c1)cccn2", -1.86),
    ("Indole", "c1ccc2c(c1)cc[nH]2", -1.42),
    # Sulfur compounds
    ("Methanethiol", "CS", -0.60),
    ("Ethanethiol", "CCS", -0.95),
    ("Dimethyl sulfide", "CSC", -0.68),
    ("Dimethyl disulfide", "CSSC", -1.39),
    ("DMSO", "CS(=O)C", 1.11),
    # Common drugs / multifunctional (cross-check molecules)
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", -1.21),
    ("Paracetamol", "CC(=O)Nc1ccc(O)cc1", -1.44),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", -3.46),
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", -1.70),
    ("Resorcinol", "c1cc(cc(c1)O)O", -0.16),
    ("Hydroquinone", "c1cc(ccc1O)O", 0.63),
    ("Catechol", "c1ccc(c(c1)O)O", 0.20),
    ("Glucose", "C(C1C(C(C(C(O1)O)O)O)O)O", 0.42),
    ("Sucrose", "C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O", 0.61),
    ("Urea", "C(=O)(N)N", 0.42),
    ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1", -1.44),
    # Steroids
    ("Cholesterol", "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C", -6.83),
    ("Estradiol", "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc34)[C@@H]1CC[C@@H]2O", -4.65),
    ("Testosterone", "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O", -4.17),
    ("Progesterone", "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2C(=O)C", -4.35),
    ("Cortisone", "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2C(=O)CO", -3.50),
    ("Androsterone", "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O", -4.60),
    # Multi-ring natural product types
    ("Quercetin", "C1=CC(=C(C=C1O)O)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O", -3.05),
    ("Resveratrol", "C1=CC(=CC=C1C=C2C=C(C(=CC2=O)O)O)O", -3.54),
    ("Aflatoxin B1", "COC1=C2C3=C(C(=O)CC3)C(=O)OC2=C4C5C=COC5OC4=C1", -3.39),
]


def main():
    # Validate all SMILES
    valid_count = 0
    for name, smi, logS in CURATED:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"INVALID: {name} -> {smi}")
        else:
            valid_count += 1

    path = os.path.join(os.path.dirname(__file__), "..", "supplementary_logs.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Name", "SMILES", "logS"])
        for name, smi, logS in CURATED:
            w.writerow([name, smi, logS])

    print(f"Written {len(CURATED)} molecules ({valid_count} valid) to {path}")


if __name__ == "__main__":
    main()
