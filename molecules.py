"""
DisSolve - Local molecule database and PubChem search integration.
"""

import json
import os
import time
import urllib.parse

import requests

# ========== 本地分子库（100+ 分子）==========
MOLECULE_DB = {
    "(自定义输入)": "",

    # === 基础有机分子 ===
    "乙醇 Ethanol": "CCO",
    "甲醇 Methanol": "CO",
    "异丙醇 Isopropanol": "CC(C)O",
    "乙二醇 Ethylene glycol": "OCCO",
    "甘油 Glycerol": "OCC(O)CO",
    "苯 Benzene": "c1ccccc1",
    "甲苯 Toluene": "Cc1ccccc1",
    "苯酚 Phenol": "Oc1ccccc1",
    "苯甲酸 Benzoic acid": "O=C(O)c1ccccc1",
    "苯乙烯 Styrene": "C=Cc1ccccc1",
    "环己烷 Cyclohexane": "C1CCCCC1",
    "己烷 Hexane": "CCCCCC",
    "辛烷 Octane": "CCCCCCCC",

    # === 溶剂与工业 ===
    "乙酸乙酯 Ethyl acetate": "CCOC(=O)C",
    "丙酮 Acetone": "CC(=O)C",
    "乙醚 Diethyl ether": "CCOCC",
    "四氢呋喃 THF": "C1CCOC1",
    "氯仿 Chloroform": "C(Cl)(Cl)Cl",
    "四氯化碳 CCl4": "C(Cl)(Cl)(Cl)Cl",
    "甲醛 Formaldehyde": "C=O",
    "醋酸 Acetic acid": "CC(=O)O",
    "柠檬酸 Citric acid": "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",

    # === 生物化学基础 ===
    "尿素 Urea": "NC(=O)N",
    "甘氨酸 Glycine": "NCC(=O)O",
    "丙氨酸 Alanine": "CC(N)C(=O)O",
    "缬氨酸 Valine": "CC(C)C(N)C(=O)O",
    "亮氨酸 Leucine": "CC(C)CC(N)C(=O)O",
    "苯丙氨酸 Phenylalanine": "NC(Cc1ccccc1)C(=O)O",
    "色氨酸 Tryptophan": "NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "酪氨酸 Tyrosine": "NC(Cc1ccc(O)cc1)C(=O)O",
    "谷氨酸 Glutamic acid": "NC(CCC(=O)O)C(=O)O",

    # === 糖类 ===
    "葡萄糖 Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "果糖 Fructose": "C(C1C(C(CO1)(O)O)O)O",
    "蔗糖 Sucrose": "C(C1C(C(C(C(O1)O)O)O)O)OC2OC(C(C(C2O)O)O)CO",
    "乳糖 Lactose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)OC[C@@H]2[C@H]([C@@H]([C@H]([C@H](O2)O)O)O)O",

    # === 维生素 ===
    "维生素A Vitamin A": "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CCO)C)C",
    "维生素B2 Riboflavin": "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)CO)c2cc1C",
    "维生素B3 Niacin": "O=C(O)c1cccnc1",
    "维生素B6 Pyridoxine": "Cc1ncc(CO)c(CO)c1O",
    "维生素B9 Folic acid": "C1=CC(=CC2=C1C(=NC(=N2)N)N)CNC(=O)NC(CCC(=O)O)C(=O)O",
    "维生素C Ascorbic acid": "C([C@@H]([C@@H]1C(=C(C(=O)O1)O)O)O)O",
    "维生素E Tocopherol": "Cc1c(C)c2C(=C(C1C)CC[C@@](C)(CCCC(C)C)O)CCCC2(C)C",

    # === 激素 ===
    "睾酮 Testosterone": "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]43C)[C@@H]1CC[C@@H]2O",
    "雌二醇 Estradiol": "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O",
    "孕酮 Progesterone": "CC(=O)[C@H]1CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C",
    "皮质醇 Cortisol": "C[C@]12CCC(=O)C=C1CC[C@@H]3[C@@H]2[C@H](C[C@]4([C@H]3CC[C@@H]4C(=O)CO)C)O",
    "胆固醇 Cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",

    # === 常见药物 ===
    "阿司匹林 Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "布洛芬 Ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "萘普生 Naproxen": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "酮洛芬 Ketoprofen": "CC(C(=O)c1ccccc1)c2ccc(C(=O)O)cc2",
    "双氯芬酸 Diclofenac": "O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl",
    "对乙酰氨基酚 Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "可待因 Codeine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "吗啡 Morphine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "咖啡因 Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "茶碱 Theophylline": "Cn1c2c(c(=O)n(C)c1=O)NC=N2",
    "青霉素G Penicillin G": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "阿莫西林 Amoxicillin": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
    "四环素 Tetracycline": "C[C@]1(c2cccc(O)c2)C(=O)C(C(=O)NC(C(=O)O)C3CCCCC3)=C(O)C(=O)N1C",
    "多西环素 Doxycycline": "C[C@H]1c2cccc(O)c2C(=O)C3=C(O)[C@](C(=O)NC4C(=O)NC(C(=O)O)C5CCCCC54)(O)C(=O)[C@@H](O)[C@@H]3[C@@H]1C",
    "环丙沙星 Ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "甲硝唑 Metronidazole": "Cc1ncc([N+](=O)[O-])n1CCO",

    # === 心血管/代谢 ===
    "二甲双胍 Metformin": "CN(C)C(=N)N=C(N)N",
    "阿托伐他汀 Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "辛伐他汀 Simvastatin": "CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@H]21",
    "硝苯地平 Nifedipine": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]",
    "氨氯地平 Amlodipine": "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c2ccccc2Cl",
    "地高辛 Digoxin": "C[C@H]1O[C@@H](O[C@H]2CC[C@@]3(C)[C@@H](CC[C@@H]4[C@@H]3CC[C@]3(C)[C@@H](C5=CC(=O)OC5)CC[C@]43O)C2)C[C@H](O)[C@@H]1O",

    # === 精神神经 ===
    "地西泮 Diazepam": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc31",
    "劳拉西泮 Lorazepam": "O=C1CN=C(c2ccccc2)c3cc(Cl)ccc3N1",
    "阿普唑仑 Alprazolam": "Cc1nnc2n1-c3ccc(Cl)cc3C(c4ccccc4)=NC2",
    "氟西汀 Fluoxetine": "CNCCC(c1ccc(OC)cc1)c2ccccc2",
    "舍曲林 Sertraline": "CNC1CCC(c2ccc(Cl)cc2)c3cccnc31",
    "奥氮平 Olanzapine": "CN1CCN(C2=Nc3ccccc3Sc4ccc(Cl)cc24)CC1",

    # === 消化系统 ===
    "奥美拉唑 Omeprazole": "COc1ccc2nc(S(=O)Cc3ncc(C)c(OC)c3C)[nH]c2c1",
    "雷尼替丁 Ranitidine": "CN(C)CCNC(=O)CSc1ccc(CN/C=C/[N+](=O)[O-])cc1",
    "西咪替丁 Cimetidine": "CC1=C(NC=N1)CSCCN/C(=N/C#N)/NC",

    # === 抗过敏/呼吸 ===
    "氯雷他定 Loratadine": "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1",
    "西替利嗪 Cetirizine": "O=C(O)C(Cc1ccc(cc1)Cl)CN2CCC(CC2)C(c3ccccc3)c4ccc(Cl)cc4",
    "沙丁胺醇 Salbutamol": "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",

    # === 抗肿瘤 ===
    "甲氨蝶呤 Methotrexate": "CN(Cc1cnc2nc(N)nc(O)c2n1)c3ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc3",
    "5-氟尿嘧啶 5-FU": "O=c1[nH]cc(F)c(=O)[nH]1",
    "紫杉醇 Paclitaxel": "CC(=O)OC1=C2C(C)[C@@H](OC(=O)C(O)C(NC(=O)c3ccccc3)c3ccccc3)C[C@@](O)(C(=O)C(=O)C4C5COC(=O)C5C(OC(C)=O)C4C2(C)C)C1(C)C",
    "顺铂 Cisplatin": "N[P+](N)(Cl)Cl",

    # === 天然产物 ===
    "青蒿素 Artemisinin": "C[C@@H]1CC[C@@H]2[C@@H](C)C3OC(=O)O[C@@H]3C[C@]2(C)O1",
    "白藜芦醇 Resveratrol": "Oc1ccc(C=Cc2cc(O)cc(O)c2)cc1",
    "姜黄素 Curcumin": "COc1cc(C=CC(=O)C=Cc2ccc(O)c(OC)c2)ccc1O",
    "辣椒素 Capsaicin": "COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O",
    "薄荷醇 Menthol": "CC(C)C1CCC(C)CC1O",
    "樟脑 Camphor": "CC12CCC(CC1=O)C2(C)C",
    "香兰素 Vanillin": "COc1cc(C=O)ccc1O",
    "丁香酚 Eugenol": "C=CCc1cc(OC)c(O)cc1",

    # === 环境污染物 ===
    "萘 Naphthalene": "c1ccc2ccccc2c1",
    "蒽 Anthracene": "c1ccc2cc3ccccc3cc2c1",
    "菲 Phenanthrene": "c1ccc2c(c1)c3ccccc3cc2",
    "芘 Pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
    "苯并芘 Benzo[a]pyrene": "c1ccc2c(c1)cc3ccc4cccc5ccc2c3c45",
    "DDT": "Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1",
    "双酚A BPA": "CC(C)(c1ccc(O)cc1)c2ccc(O)cc2",
    "三聚氰胺 Melamine": "Nc1nc(N)nc(N)n1",

    # === 复杂分子 ===
    "三氯蔗糖 Sucralose": "C[C@@H]1O[C@@H](O[C@H]2O[C@H](CCl)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@H]1Cl",
    "胰岛素片段 Insulin (simplified)": "NCCCCC(N)C(=O)N",

    # === 抗生素扩展 ===
    "红霉素 Erythromycin": "CCC1(C(=O)O)C(OC2C(C(OC3CC(C(C(O3)C)O)(C)OC)C(OC4CC(C(C(O4)C)O)(C)O)C(C2=O)C)C)C(C)C(=O)C(C)CC(C)C1O",
    "阿奇霉素 Azithromycin": "CCC1(C(=O)O)C(OC2C(C(OC3CC(C(C(O3)C)O)(C)OC)C(OC4CC(C(C(O4)C)O)(C)O)C(C2=O)C)C)C(C)C(=O)C(C)CC(C)C1(C)O",
    "克拉霉素 Clarithromycin": "COC1C(OC2C(C(OC3CC(C(C(O3)C)O)(C)OC)C(OC4CC(C(C(O4)C)O)(C)O)C(C2=O)C)C)C(C)C(=O)C(C)CC(C)(OC)C1O",
    "链霉素 Streptomycin": "CC1C(C(=O)NC2C(C(=O)NC(CO)C(O)C3C(O)C(O)C(O3)C=O)C(O)C(O)C2O)OC(C4C(C(C(O4)CO)O)O)C(O)C1N",
    "庆大霉素 Gentamicin": "CC1(CO)OC2C(O)C(N)C(O)C2O1",
    "氯霉素 Chloramphenicol": "O=C(NC(CO)C(O)c1ccc([N+](=O)[O-])cc1)C(Cl)Cl",
    "头孢氨苄 Cephalexin": "CC1=C(C(=O)O)N2C(=O)C(NC(=O)C(N)c3ccccc3)C2SC1",
    "头孢曲松 Ceftriaxone": "CN1C(=O)C(=NO)C(=O)N1C2C(C(=O)O)N3C(C(=O)O)=C(CSc4nc(N)n[nH]4)CSC3C2",
    "万古霉素 Vancomycin": "CNC1C(O)C2C(OC3C(O)C(O)C(OC4CC(C)(N)C(O)C(C)O4)C(C)OC(=O)C(CC(N)C(=O)O)NC(=O)C5CC(=O)NC(Cc6ccc(O)cc6)C(=O)NC(Cc7ccc(O)c(O)c7)C(=O)NC8C(=O)NC9C(=O)NC(C(=O)NC(C(=O)O)Cc%10cc(Cl)ccc%10O)Cc%11cc(Cl)ccc%11O",
    "异烟肼 Isoniazid": "O=C(NN)c1ccncc1",
    "乙胺丁醇 Ethambutol": "CCN(CC)C(C)C(O)CC(O)C(C)N(CC)CC",
    "利福平 Rifampicin": "CC1C=CC=C(C)C(=O)Nc2c(O)c3c(O)c(C)c(OC(C)=O)c4c3c2C(=O)C1(C)OC=CC(C)C(O)C(C)C(O)C(C)C(OC(C)=O)C(C)C(OC)C=COC4(C)O",
    "磺胺甲恶唑 Sulfamethoxazole": "Cc1noc(NS(=O)(=O)c2ccc(N)cc2)n1",
    "甲氧苄啶 Trimethoprim": "COc1cc(Cc2cnc(N)nc2N)ccc1OC",

    # === 抗病毒药物 ===
    "阿昔洛韦 Acyclovir": "NC1=NC2=C(N1COCCO)N=CN2",
    "奥司他韦 Oseltamivir": "CCOC(=O)C1=C(C)OC(C(CC)OC(C)=O)C(N)C1",
    "利巴韦林 Ribavirin": "OC1OC(CO)C(O)C1n2cnc3c(=O)[nH]cnc23",
    "齐多夫定 Zidovudine": "CC1=CN(C2CC(N=[N+]=[N-])C(CO)O2)C(=O)NC1=O",
    "拉米夫定 Lamivudine": "NC1=NC(=O)N(C2CSC(CO)O2)C=C1",
    "替诺福韦 Tenofovir": "CC(N1C=NC2=C1N=CN=C2N)OCP(=O)(O)O",
    "瑞德西韦 Remdesivir": "CCC(CC)COC(=O)C(N)CC(=O)OC1C(OC2C(C(F)(F)C3=CN(N=C3)C4=CC=CC=C4)C(O)C2N)OC(CO)C1O",
    "法匹拉韦 Favipiravir": "NC1=NC(=O)C(F)=NN1C(=O)c2ccccc2F",
    "洛匹那韦 Lopinavir": "CC1=C(C(=O)NC(Cc2ccccc2)C(O)CN3CCN(Cc4ccccc4)CC3)C=CC=C1",
    "利托那韦 Ritonavir": "CC(C)C1=NC(CN(C)C(=O)NC(Cc2ccccc2)C(O)C(NC(=O)OCc3scnc3)Cc4ccccc4)=CS1",

    # === 抗真菌药物 ===
    "氟康唑 Fluconazole": "OC(Cn1cncn1)(Cn2cncn2)c3ccc(F)cc3F",
    "酮康唑 Ketoconazole": "CC(=O)N1CCN(C(=O)OC2C(C)OC(OC2)c3ccc(Oc4ccc(Cl)cc4)cc3)CC1",
    "伊曲康唑 Itraconazole": "CC1(C)OC(OC2CCC(=O)N(C(=O)N3CCN(c4ccc(Oc5ccc(Cl)cc5)cc4)CC3)c6ccc(Cl)cc6)CC1",
    "两性霉素B Amphotericin B": "CC1C=CC=CC=CC=CC=CC=CC=CC(C)C(O)C(C)C(O)C(C)C(OC2OC(C)C(O)C(N)C2O)C(C)C(O)C(C)C(=O)C=CC3(CO3)C(C)C(O)CC1=O",
    "制霉菌素 Nystatin": "CC1C=CC=CC=CC=CC=CC=CC=CC(C)C(O)C(C)C(O)C(C)C(OC2OC(C)C(O)C(N)C2O)C(C)C(O)C(C)C(=O)C=CC3(CO3)C(C)C(O)CC1=O",

    # === 抗肿瘤药物扩展 ===
    "伊马替尼 Imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)nc1Nc4ncc(C)cc4C",
    "吉非替尼 Gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN4CCOCC4",
    "厄洛替尼 Erlotinib": "COCCOc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOC",
    "索拉非尼 Sorafenib": "CNC(=O)c1ccc(Oc2ccc(Cl)c(C(F)(F)F)c2)nc1Nc3ccc(F)cc3",
    "舒尼替尼 Sunitinib": "CCN(CC)CCOc1cc2c(C=C(NC(=O)N3CCN(C)CC3)C2=O)cc1F",
    "达沙替尼 Dasatinib": "CC1=C(C(=O)Nc2ccc(Cl)cc2)SC(=N1)N3CCN(CCN4CCOCC4)CC3",
    "硼替佐米 Bortezomib": "CC(C)C(NC(=O)C(N)Cc1ccccc1)B(O)O",
    "环磷酰胺 Cyclophosphamide": "O=P1(OCCCN1CCCl)N(CCCl)CCCl",
    "卡铂 Carboplatin": "O=C1O[Pt]2(N1)OC(=O)C3(CCC3)OC2=O",
    "奥沙利铂 Oxaliplatin": "O=C1O[Pt]2(OC(=O)C12)C3CCCCC3N",
    "长春新碱 Vincristine": "CCC1(CC2CN(C1)C3C(C2)C(C4=C(C=C5C(=C4)C(=CN5)C6(C(C(C7=CC(=C(C=C7N6)C(=O)OC)C(=O)OC)C(O)C(=O)OC)O)O)O)CC)OC(=O)C",
    "依托泊苷 Etoposide": "COc1cc2c(cc1OC)C3C(OC4C(C5COC(=O)c6cc(O)c(O)c(Oc7OC(CO)C(O)C(O)C7O)c6)COC5OC4C3O)COC2=O",
    "多西他赛 Docetaxel": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1O)OC(=O)C(C(O)C(O)C5=CC=CC=C5)N(C)C(=O)OC(C)(C)C)O)(CO4)OC(=O)C6=CC=CC=C6)O)OC(=O)C",

    # === 麻醉药与镇痛药 ===
    "利多卡因 Lidocaine": "CCN(CC)C(=O)CN(c1c(C)cccc1C)C",
    "普鲁卡因 Procaine": "CCN(CC)CCOC(=O)c1ccc(N)cc1",
    "布比卡因 Bupivacaine": "CCCCC1=C(C(=O)NC2=C(C=CC=C2)C)C=CC=C1",
    "芬太尼 Fentanyl": "CCN(CC)C(=O)N(c1ccccc1)C2CCN(CCc3ccccc3)CC2",
    "曲马多 Tramadol": "CN(C)CCC1(O)C2CCCC(C2)C1c3cccc(OC)c3",
    "美沙酮 Methadone": "CCC(=O)C(CC(C)N(C)C)(c1ccccc1)c2ccccc2",
    "哌替啶 Pethidine": "CCOC(=O)C1(CCN(C)CC1)c2ccccc2",
    "纳洛酮 Naloxone": "C=CCN1CCC23C4=CC=CC(O)=C4OC2C(O)C=CC3C1C5CC5",
    "氯胺酮 Ketamine": "CNC1(CCCCC1=O)c2ccccc2Cl",

    # === 心血管药物扩展 ===
    "氯沙坦 Losartan": "CCCCC1=NC(Cl)=C(CO)n1Cc2ccc(-c3ccccc3C(=O)O)cc2",
    "缬沙坦 Valsartan": "CCCCC(=O)N(Cc1ccc(-c2ccccc2C(=O)O)cc1)C(C(C)C)C(=O)O",
    "卡托普利 Captopril": "CC(S)C(=O)N1C(C(=O)O)CCC1",
    "依那普利 Enalapril": "CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N2C(C(=O)O)CCC2",
    "维拉帕米 Verapamil": "COc1cc(C(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)c3ccc(OC)c(OC)c3)ccc1OC",
    "地尔硫卓 Diltiazem": "COc1ccc(CCN2CC(C)(C)OC(=O)C(OC(C)=O)c3ccccc3SC2)cc1",
    "普萘洛尔 Propranolol": "CC(C)NCC(O)COc1cccc2ccccc12",
    "美托洛尔 Metoprolol": "COCCCc1ccc(OCC(O)CNC(C)C)cc1",
    "氢氯噻嗪 Hydrochlorothiazide": "NS(=O)(=O)c1cc2c(NCNS2(=O)=O)cc1Cl",
    "呋塞米 Furosemide": "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl",
    "华法林 Warfarin": "CC(=O)CC(C1=CC=CC=C1)c2c(O)c3ccccc3oc2=O",
    "氯吡格雷 Clopidogrel": "COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=C(C2)C=CS3",
    "阿哌沙班 Apixaban": "COc1ccc(N2C(=O)C=C(N3CCN(C(=O)Nc4ccc(OC)cc4)CC3)NC2=O)cc1",

    # === 降糖药物 ===
    "格列本脲 Glibenclamide": "COc1ccc(Cl)cc1C(=O)NCCCc2ccc(S(=O)(=O)NC(=O)NC3CCCCC3)cc2",
    "格列美脲 Glimepiride": "CC1=C(CNC(=O)Nc2ccc(S(=O)(=O)NC(=O)NC3CCCCC3)cc2)C(=O)OC1",
    "吡格列酮 Pioglitazone": "CC1(COc2ccc(CC3SC(=O)NC3=O)cc2)COc4ccc(C)cc4O1",
    "罗格列酮 Rosiglitazone": "CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c3ncc(C)cc3C",
    "西格列汀 Sitagliptin": "Fc1cc(F)c(CC(N)C(=O)N2Cc3nn(C(F)(F)F)c4ccc(F)cc3C2)cc1F",
    "达格列净 Dapagliflozin": "CCOc1ccc(CC2C(O)C(O)C(O)C(CO)O2)cc1Cl",
    "恩格列净 Empagliflozin": "Clc1ccc(CC2C(O)C(O)C(O)C(CO)O2)cc1Oc3ccc(OCCO)cc3",

    # === 消化系统药物扩展 ===
    "泮托拉唑 Pantoprazole": "COc1ccnc(CS(=O)c2nc3cc(OC(F)F)ccc3n2C)c1OC",
    "兰索拉唑 Lansoprazole": "CC1=C(CS(=O)c2nc3ccccc3n2C)C=CN=C1COCC(F)(F)F",
    "昂丹司琼 Ondansetron": "CN1CC(C2=CC=CN=C2)CC3=C1C(=O)NC4=CC=CC=C34",
    "甲氧氯普胺 Metoclopramide": "CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC",
    "洛哌丁胺 Loperamide": "CN(C)C(=O)C(CCN1CCC(CC1)(c2ccccc2)c3ccccc3)(c4ccccc4)c5ccccc5",

    # === 精神神经药物扩展 ===
    "利培酮 Risperidone": "CC1=C(CCN2CCC(CC2)c3noc4cc(F)ccc34)C(=O)N5CCCCC5=N1",
    "喹硫平 Quetiapine": "OCCOCCN1CCN(C2=Nc3ccccc3Sc4ccccc24)CC1",
    "阿立哌唑 Aripiprazole": "Clc1ccc(OCCCCN2CCN(c3cccc(NC(=O)C4=C(Cl)CCCC4)c3)CC2)cc1",
    "帕罗西汀 Paroxetine": "FC1=CC=C(C2C(CNCC2)COC3=CC=C4OCOC4=C3)C=C1",
    "西酞普兰 Citalopram": "N#CC(C1=CC=C(F)C=C1)(c2ccc3OCCCOc3c2)CCCN(C)C",
    "文拉法辛 Venlafaxine": "COc1ccc(CC(N(C)C)C2(O)CCCCC2)cc1",
    "度洛西汀 Duloxetine": "CNCCc1cccs1OC2=CC=CC3=CC=CC=C32",
    "米氮平 Mirtazapine": "CN1CCN2C(C1)C3=CC=CC=C3CC4=CC=CN=C24",
    "艾司唑仑 Estazolam": "ClC1=CC2=C(NC(=O)CN3C=NC=N3)C=NC=C2C=C1",
    "氯硝西泮 Clonazepam": "O=C1CN=C(C2=CC=CC=C2Cl)C3=CC([N+](=O)[O-])=CC=C3N1",
    "苯巴比妥 Phenobarbital": "CCC1(C(=O)NC(=O)NC1=O)C2=CC=CC=C2",
    "卡马西平 Carbamazepine": "NC(=O)N1C2=CC=CC=C2C=CC3=CC=CC=C31",
    "丙戊酸 Valproic acid": "CCCC(C(=O)O)CCC",
    "加巴喷丁 Gabapentin": "NC1(CC(=O)O)CCCCC1",
    "普瑞巴林 Pregabalin": "CC(C)CC(CC(=O)O)CN",

    # === 抗组胺与呼吸系统 ===
    "苯海拉明 Diphenhydramine": "CN(C)CCOC(c1ccccc1)c2ccccc2",
    "非索非那定 Fexofenadine": "CC(C)(C(=O)O)c1ccc(CC2(O)CCN(CCCC(c3ccccc3)c4ccccc4)CC2)cc1",
    "孟鲁司特 Montelukast": "CC(C)(O)c1cc(Cl)c(C=Cc2ccc3c(c2)CCC3SCC4(CC4)CC(=O)O)cc1",
    "沙美特罗 Salmeterol": "CC(C)(C)NCC(O)c1cccc(O)c1COCCCCCCc2ccccc2",
    "福莫特罗 Formoterol": "COc1cc(C(O)CNC(C)CCc2ccc(OC)cc2)ccc1NC=O",
    "异丙托溴铵 Ipratropium": "C[N+]1(C)C2CC(OC(=O)C(CO)c3ccccc3)CC1C1OC12",
    "噻托溴铵 Tiotropium": "C[N+]1(C)C2CC(OC(=O)C(O)(c3cccs3)c4cccs4)CC1C1OC12",

    # === 农药与除草剂 ===
    "草甘膦 Glyphosate": "OC(=O)CNCP(=O)(O)O",
    "百草枯 Paraquat": "C[n+]1ccc(cc1)-c2cc[n+](C)cc2",
    "阿特拉津 Atrazine": "CCNC1=NC(=NC(=N1)Cl)NC(C)C",
    "毒死蜱 Chlorpyrifos": "CCOP(=S)(OCC)Oc1nc(Cl)c(Cl)cc1Cl",
    "吡虫啉 Imidacloprid": "O=[N+]([O-])/N=C1\\NCCN1Cc2cnc(Cl)cc2",
    "噻虫嗪 Thiamethoxam": "CN1C(=N[N+](=O)[O-])SCC1COc2cnc(Cl)cc2",
    "马拉硫磷 Malathion": "CCOC(=O)CC(SP(=S)(OC)OC)C(=O)OCC",
    "敌敌畏 Dichlorvos": "COP(=O)(OC)OC=C(Cl)Cl",
    "氯氰菊酯 Cypermethrin": "CC1(C)C(C=C(Cl)Cl)C1C(=O)OC(C#N)c2cccc(Oc3ccccc3)c2",
    "溴氰菊酯 Deltamethrin": "CC1(C)C(C=C(Br)Br)C1C(=O)OC(C#N)c2cccc(Oc3ccccc3)c2",
    "2,4-D 二氯苯氧乙酸": "OC(=O)COc1ccc(Cl)cc1Cl",
    "西维因 Carbaryl": "CNC(=O)Oc1cccc2ccccc12",
    "氟虫腈 Fipronil": "N#CC1=C(N)N(C(F)(F)F)C(=O)C1S(=O)C(F)(F)F",
    "草铵膦 Glufosinate": "CP(=O)(CCC(N)C(=O)O)O",

    # === 生物碱 ===
    "尼古丁 Nicotine": "CN1CCC[C@H]1c2cccnc2",
    "奎宁 Quinine": "COc1ccc2nccc(C(O)C3CC4CCN3CC4C=C)c2c1",
    "士的宁 Strychnine": "O=C1C[C@@H]2[C@H]3N(C4=CC=CC=C34)C[C@]15[C@H]6[C@@H]([C@@H]2N(CC7=CCO[C@]67[H])C5)O",
    "阿托品 Atropine": "CN1C2CCC1CC(C2)OC(=O)C(CO)c3ccccc3",
    "东莨菪碱 Scopolamine": "CN1C2CC(CC1C3OC32)OC(=O)C(CO)c4ccccc4",
    "可卡因 Cocaine": "CN1C2CCC1C(C2)C(=O)OC(C3=CC=CC=C3)C(=O)OC",
    "秋水仙碱 Colchicine": "COc1cc2c(c(OC)c1OC)CCC3=C(C(=O)C=CC3=O)c2N(C)C=O",
    "麻黄碱 Ephedrine": "CNC(C)C(O)c1ccccc1",
    "小檗碱 Berberine": "COc1cc2c(c(OC)c1OC)CCN3CCC4=CC5=C(C=C4C3=C2)OCO5",
    "长春碱 Vinblastine": "CCC1(CC2CN(C1)C3C(C2)C(C4=C(C=C5C(=C4)C(=CN5)C6(C(C(C7=CC=C(C=C7N6)C(=O)OC)C(=O)OC)O)O)C)CC)OC(=O)C",
    "喜树碱 Camptothecin": "CC[C@@]1(O)C(=O)OCc2c1cnc3ccc4c(c23)C=CC(=O)N4",
    "紫草素 Shikonin": "CC(C)=CCC1=C(O)C(=O)c2c(O)cccc2C1=O",

    # === 萜类化合物 ===
    "柠檬烯 Limonene": "CC1=CCC(CC1)C(=C)C",
    "蒎烯 Pinene": "CC1=CCC2C(C1)C2(C)C",
    "芳樟醇 Linalool": "CC(C)=CCCC(C)(O)C=C",
    "香叶醇 Geraniol": "CC(C)=CCCC(C)=CCO",
    "桉叶油素 Eucalyptol": "CC1(C2CCC3(OC3(C1)C2)C)C",
    "百里香酚 Thymol": "CC(C)c1ccc(C)cc1O",
    "香茅醛 Citronellal": "CC(C)=CCCC(C)CC=O",
    "冰片 Borneol": "CC1(C2CCC1C(C2)O)C",
    "甘草酸 Glycyrrhizic acid": "CC1(C2CCC3(C(C2(CCC1OC4C(C(C(C(O4)C(=O)O)O)O)OC5C(C(C(C(O5)C(=O)O)O)O)O)C)C(=O)C=C6C3(CCC7(C6(CCC(=O)O)C(C(=O)O)=CC7=O)C)C)C)C",
    "紫杉醇 Taxol": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1O)OC(=O)C(O)C(NC(=O)C5=CC=CC=C5)C6=CC=CC=C6)O)(CO4)OC(=O)C)C)O)OC(=O)C",

    # === 黄酮类化合物 ===
    "槲皮素 Quercetin": "Oc1cc(O)c2C(=O)C(O)=C(Oc2c1)c3ccc(O)c(O)c3",
    "山奈酚 Kaempferol": "Oc1cc(O)c2C(=O)C(O)=C(Oc2c1)c3ccc(O)cc3",
    "芹菜素 Apigenin": "Oc1ccc(C2=CC(=O)c3c(O)cc(O)cc3O2)cc1",
    "木犀草素 Luteolin": "Oc1ccc(C2=CC(=O)c3c(O)cc(O)cc3O2)c(O)c1",
    "大豆苷元 Daidzein": "Oc1ccc(C2=COc3cc(O)ccc3C2=O)cc1",
    "染料木素 Genistein": "Oc1ccc(C2=COc3cc(O)cc(O)c3C2=O)cc1",
    "橙皮素 Hesperetin": "COc1cc(O)c2C(=O)CC(Oc2c1)c3ccc(O)c(OC)c3",
    "儿茶素 Catechin": "Oc1cc(O)c2C(O)CC(Oc2c1)c3ccc(O)c(O)c3",
    "表没食子儿茶素没食子酸酯 EGCG": "O=C(OC1Cc2c(O)cc(O)cc2OC1c3cc(O)c(O)c(O)c3)c4cc(O)c(O)c(O)c4",
    "芦丁 Rutin": "CC1OC(OC2C(O)C(O)C(CO)OC2Oc3c(O)cc4C(=O)C(O)=C(Oc4c3)c5ccc(O)c(O)c5)C(O)C(O)C1O",

    # === 甾体化合物 ===
    "氢化可的松 Hydrocortisone": "CC12CCC(=O)C=C1CCC3C2C(O)CC4(C)C3CCC4(O)C(=O)CO",
    "泼尼松 Prednisone": "CC12CC(=O)C3C(C1CCC2(O)C(=O)CO)CCC4=CC(=O)C=CC34C",
    "地塞米松 Dexamethasone": "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
    "螺内酯 Spironolactone": "CC12CCC3C(C1CCC2(O)C(=O)CO)CCC4=CC(=O)C=CC34C",
    "非那雄胺 Finasteride": "CC12CCC3C(C1CCC2C(=O)NC(C)(C)C)CCC4=CC(=O)C=CC34C",

    # === 食品添加剂 ===
    "阿斯巴甜 Aspartame": "COC(=O)C(CC1=CC=CC=C1)NC(=O)C(N)CC(=O)O",
    "糖精 Saccharin": "O=C1NS(=O)(=O)C2=CC=CC=C12",
    "甜蜜素 Cyclamate": "OS(=O)(=O)NC1CCCCC1",
    "安赛蜜 Acesulfame K": "CC1=CC(=O)NS(=O)(=O)O1",
    "山梨酸钾 Potassium sorbate": "CC=CC=CC(=O)[O-]",
    "苯甲酸钠 Sodium benzoate": "O=C([O-])c1ccccc1",
    "谷氨酸钠 MSG": "N[C@@H](CCC(=O)[O-])C(=O)O",
    "柠檬黄 Tartrazine": "O=S(=O)(O)c1ccc(N=Nc2ccc(N=Nc3ccc(S(=O)(=O)O)cc3)c(O)c2)cc1",
    "日落黄 Sunset Yellow": "O=S(=O)(O)c1ccc(N=Nc2ccc(N=Nc3ccc(S(=O)(=O)O)cc3)c(O)c2)cc1",
    "赤藓红 Erythrosine": "O=C1OC(c2c(I)c(O)c(I)c(I)c2I)=C(I)C(=O)c3c1c(I)c(O)c(I)c3I",
    "诱惑红 Allura Red AC": "COc1cc(N=Nc2ccc(S(=O)(=O)O)cc2)c(O)c(C)c1",
    "亮蓝 Brilliant Blue": "CCN(CC1=CC=C(C=C1)S(=O)(=O)O)C2=CC=C(C=C2)C(C3=CC=C(C=C3)N(CC)CC4=CC=C(C=C4)S(=O)(=O)O)=C5C=CC(=O)C=C5",

    # === 染料与色素 ===
    "靛蓝 Indigo": "O=c1c(=O)c2ccccc2N1c3ccccc3",
    "苏丹红 Sudan I": "Oc1ccc(N=Nc2ccccc2)c3ccccc13",
    "刚果红 Congo Red": "NC1=CC=CC=C1N=NC2=CC=C(C=C2)C3=CC=C(C=C3)N=NC4=C(N)C=CC=C4S(=O)(=O)O",
    "孔雀石绿 Malachite Green": "CN(C)C1=CC=C(C=C1)C(C2=CC=C(C=C2)N(C)C)=C3C=CC(=N(C)C)C=C3",
    "亚甲蓝 Methylene Blue": "CN(C)C1=CC2=C(C=C1)N=C3C=CC(=N(C)C)C=C3S2",
    "荧光素 Fluorescein": "Oc1ccc2c(c1)OC3=CC(=O)C=CC3=C24C5=CC=CC=C5C(=O)O4",
    "罗丹明B Rhodamine B": "CCN(CC)C1=CC2=C(C=C1)C(=C3C=CC(=N(CC)CC)C=C3O2)C4=CC=CC=C4C(=O)O",
    "茜素 Alizarin": "O=C1C(=O)C2=CC=CC=C2C3=C1C=CC=C3O",

    # === 香料与香精 ===
    "香豆素 Coumarin": "O=C1OC2=CC=CC=C2C=C1",
    "肉桂醛 Cinnamaldehyde": "O=CC=Cc1ccccc1",
    "茴香脑 Anethole": "COc1ccc(C=CC)cc1",
    "麦芽酚 Maltol": "CC1OC=CC(=O)C1=O",
    "乙基麦芽酚 Ethyl Maltol": "CCC1OC=CC(=O)C1=O",
    "苯乙醇 Phenylethyl Alcohol": "OCCc1ccccc1",
    "乙酸苄酯 Benzyl acetate": "CC(=O)OCc1ccccc1",
    "乙酸芳樟酯 Linalyl acetate": "CC(C)=CCCC(C)(OC(C)=O)C=C",
    "柠檬醛 Citral": "CC(C)=CCCC(C)=CC=O",
    "紫罗兰酮 Ionone": "CC1=CC(=O)CC(C)(C)C1C=CC(=O)C",
    "苯甲醛 Benzaldehyde": "O=Cc1ccccc1",
    "水杨酸甲酯 Methyl salicylate": "COC(=O)c1ccccc1O",

    # === 工业化学品扩展 ===
    "乙腈 Acetonitrile": "CC#N",
    "二甲基甲酰胺 DMF": "CN(C)C=O",
    "二甲基亚砜 DMSO": "CS(C)=O",
    "六甲基磷酰胺 HMPA": "CN(C)P(=O)(N(C)C)N(C)C",
    "二氯甲烷 Dichloromethane": "C(Cl)Cl",
    "三氯乙烯 Trichloroethylene": "C(=C(Cl)Cl)Cl",
    "四氯乙烯 Tetrachloroethylene": "C(=C(Cl)Cl)(Cl)Cl",
    "二恶烷 Dioxane": "C1COCCO1",
    "吡啶 Pyridine": "c1ccncc1",
    "苯胺 Aniline": "Nc1ccccc1",
    "硝基苯 Nitrobenzene": "O=N(=O)c1ccccc1",
    "邻苯二甲酸二丁酯 DBP": "CCCCOC(=O)c1ccccc1C(=O)OCCCC",
    "邻苯二甲酸二(2-乙基己基)酯 DEHP": "CCCCC(CC)COC(=O)c1ccccc1C(=O)OCC(CC)CCCC",
    "多氯联苯 PCB": "Clc1ccc(Cl)c(Cl)c1Cl",
    "丙烯酰胺 Acrylamide": "NC(=O)C=C",
    "环氧乙烷 Ethylene oxide": "C1CO1",
    "异氰酸甲酯 Methyl isocyanate": "CN=C=O",
    "光气 Phosgene": "O=C(Cl)Cl",
    "氢氟酸 HF": "F",
    "过氧化氢 Hydrogen peroxide": "OO",
    "次氯酸钠 Sodium hypochlorite": "[O-]Cl",
    "氨水 Ammonia": "N",

    # === 更多天然产物 ===
    "番茄红素 Lycopene": "CC(C)=CCCC(C)=CC=CC(C)=CC=CC(C)=CC=CC=C(C)C=CC=C(C)CCC=C(C)C",
    "虾青素 Astaxanthin": "CC1=C(C(=O)C(O)CC1(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(C)C(=O)C(O)CC2(C)C)C)C",
    "叶黄素 Lutein": "CC1=C(C(O)CC(C)(C)C1)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2C(=CC(O)CC2(C)C)C)C)C",
    "玉米黄质 Zeaxanthin": "CC1=C(C(O)CC(C)(C)C1)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(C)C(O)CC(C)(C)C2)C)C",
    "辣椒红素 Capsanthin": "CC1=C(C(O)CC(C)(C)C1)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2C(=CC(=O)CC2(C)C)C)C)C",
    "大黄素 Emodin": "Cc1cc(O)c2C(=O)c3c(O)cc(O)cc3C(=O)c2c1",
    "雷公藤甲素 Triptolide": "CC1C2OC2C3(O)C4C(O)C5C(C)C6=C(C=CC6=O)C5(C)C4(CO3)C1",
    "紫草素 Alkanin": "CC(C)=CCC1=C(O)C(=O)c2c(O)cccc2C1=O",
    "银杏内酯B Ginkgolide B": "CC1C2C3C4(C)C(O)C(O)C5OC(=O)C6C(O)C(O)C3C(C)(C)C6OC(=O)C45C2C(=O)OC1(C)C",
    "丹酚酸B Salvianolic acid B": "COc1cc(C=CC(=O)OC2C(OC(=O)C=Cc3ccc(O)c(O)c3)C(O)C(O)C(CO)O2)ccc1O",
    "大黄酸 Rhein": "O=C(O)c1cc(O)c2C(=O)c3c(O)cc(O)cc3C(=O)c2c1",
    "五味子酯甲 Schisandrin A": "COc1cc2c(c(OC)c1OC)C3C(C)C(C)C(=O)C3C(C)(C)C2O",
    "穿心莲内酯 Andrographolide": "CC1(CO)CC(O)C2C(C)(C)C(O)C(O)C3=CC(=O)OC3C12",
    "青蒿琥酯 Artesunate": "CC1CC2C3C(C)OC(=O)OC3CC(C)(C)C2O1",
    "雷公藤红素 Celastrol": "CC1=C(O)C(=O)C=C2C1=CC=C3C2(C)CCC4(C)C3CCC5(C)C4CCC6(C)C5C=CC6=O",

    # === 更多农药与有毒物质 ===
    "滴滴涕 DDT (full name)": "C(c1ccc(Cl)cc1)(c2ccc(Cl)cc2)C(Cl)(Cl)Cl",
    "六六六 Lindane": "ClC1C(Cl)C(Cl)C(Cl)C(Cl)C1Cl",
    "五氯酚 Pentachlorophenol": "Oc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl",
    "黄曲霉毒素B1 Aflatoxin B1": "COc1cc2c(c3c4C=COC4=CC(=O)C3C2=O)c5c1OCC5=O",
    "河豚毒素 Tetrodotoxin": "O=C1NC2C3C4OC5(O)OC(CO)C(O)C5C4C6(O)C23C(O)C1(O)C6O",
    "肉毒杆菌毒素模拟 Botulinum (mock)": "NCCCCC(N)C(=O)NCC(=O)N",
    "芥子气 Mustard gas": "ClCCSCCCl",
    "沙林 Sarin": "CC(C)OP(=O)(C)F",
    "塔崩 Tabun": "CCN(C)P(=O)(C#N)OCC",
    "蓖麻毒素模拟 Ricin (mock)": "NCCCCC(N)C(=O)NCC(N)=O",

    # === 稀土与无机物（SMILES扩展） ===
    "氧气 Oxygen": "O=O",
    "臭氧 Ozone": "[O-][O+]=O",
    "二氧化碳 Carbon dioxide": "O=C=O",
    "一氧化碳 Carbon monoxide": "[C-]#[O+]",
    "二氧化硫 Sulfur dioxide": "O=S=O",
    "三氧化硫 Sulfur trioxide": "O=S(=O)=O",
    "硫酸 Sulfuric acid": "OS(=O)(=O)O",
    "硝酸 Nitric acid": "O[N+](=O)[O-]",
    "磷酸 Phosphoric acid": "OP(=O)(O)O",
    "高氯酸 Perchloric acid": "O[Cl+3]([O-])([O-])[O-]",

    # === 常见毒品与管制物质（教育用途） ===
    "麦角酸二乙酰胺 LSD": "CCN(CC)C(=O)C1CN(C2CC3=CNC4=CC=CC(=C34)C2=C1)C",
    "甲基苯丙胺 Methamphetamine": "CNC(C)Cc1ccccc1",
    "摇头丸 MDMA": "CC(NC)Cc1ccc2OCOc2c1",
    "氯胺酮 Ketamine (药用)": "CNC1(CCCCC1=O)c2ccccc2Cl",
    "大麻二酚 Cannabidiol": "CCCCCC1=CC(=C(C(=C1)O)C2C=C(CC[C@@H]2C(=C)C)C)O",
    "四氢大麻酚 THC": "CCCCCC1=CC(=C2C3C=C(CCC3C(OC2=C1)(C)C)C)O",
}


def build_search_index():
    """Build lowercase search index from MOLECULE_DB."""
    index = {}
    for display_name, smiles in MOLECULE_DB.items():
        index[display_name.lower()] = smiles
        parts = display_name.split()
        for part in parts:
            clean = part.strip().lower()
            if len(clean) > 1:
                index[clean] = smiles
    return index


SEARCH_INDEX = build_search_index()

# ========== PubChem 缓存与搜索 ==========
CACHE_FILE = "pubchem_cache.json"
pubchem_cache = {}

_session = None


def _get_session():
    """Lazy-init a requests Session with connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": "DisSolve/1.0"})
    return _session


def _pubchem_request(url, timeout=20):
    """Try SSL-verified request first, fall back to verify=False on SSLError."""
    session = _get_session()
    try:
        return session.get(url, timeout=timeout, verify=True)
    except requests.exceptions.SSLError:
        return session.get(url, timeout=timeout, verify=False)
    except requests.exceptions.ProxyError:
        # Proxy configured but unreachable — try direct connection
        return requests.get(url, timeout=timeout, verify=False, proxies={"http": None, "https": None})


def _configure_proxy():
    """Read HTTP_PROXY / HTTPS_PROXY from environment and configure the session."""
    import os as _os
    http_proxy = _os.environ.get("HTTP_PROXY") or _os.environ.get("http_proxy")
    https_proxy = _os.environ.get("HTTPS_PROXY") or _os.environ.get("https_proxy")
    if http_proxy or https_proxy:
        _get_session().proxies = {
            "http": http_proxy or https_proxy,
            "https": https_proxy or http_proxy,
        }


_configure_proxy()


def load_cache():
    global pubchem_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                pubchem_cache = json.load(f)
        except (json.JSONDecodeError, OSError, IOError):
            pubchem_cache = {}


def save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(pubchem_cache, f, ensure_ascii=False, indent=2)
    except (OSError, IOError):
        pass


load_cache()


PUBCHEM_BASE_URL = os.environ.get(
    "PUBCHEM_BASE_URL",
    "https://pubchem.ncbi.nlm.nih.gov",
).rstrip("/")

# Alternative domains to try as fallback (some may be more accessible from China)
_PUBCHEM_FALLBACK_HOSTS = [
    "pubchem.ncbi.nlm.nih.gov",
]


def _try_pubchem_urls(encoded_name):
    """Try PubChem lookup across base URL and fallback hosts."""
    path = f"/rest/pug/compound/name/{encoded_name}/property/CanonicalSMILES/JSON"
    urls = [f"{PUBCHEM_BASE_URL}{path}"]
    for host in _PUBCHEM_FALLBACK_HOSTS:
        alt = f"https://{host}{path}"
        if alt not in urls:
            urls.append(alt)

    for url in urls:
        try:
            r = _pubchem_request(url, timeout=10)
            if r.status_code == 200:
                return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            continue
    return None


def search_pubchem(name, max_retries=3):
    """Search PubChem PUG REST API for a compound SMILES by name."""
    if not name or not name.strip():
        return None, "名称不能为空"

    name_clean = name.strip()
    name_lower = name_clean.lower()

    if name_lower in pubchem_cache:
        return pubchem_cache[name_lower], "success (cached)"

    time.sleep(0.5)
    encoded = urllib.parse.quote(name_clean)

    for attempt in range(max_retries):
        try:
            r = _try_pubchem_urls(encoded)
            if r is None:
                time.sleep(1.0 * (attempt + 1))
                continue
            if r.status_code == 200:
                data = r.json()
                if 'Fault' in data:
                    fault = data.get('Fault', {}).get('Message', '')
                    if 'NotFound' in fault or 'not found' in fault.lower():
                        return None, "PubChem 未找到该化合物"
                    time.sleep(1.0 * (attempt + 1))
                    continue
                props = data.get('PropertyTable', {}).get('Properties', [])
                if props:
                    smiles = props[0].get('CanonicalSMILES') or props[0].get('IsomericSMILES')
                    if smiles and smiles.strip():
                        result = smiles.strip()
                        pubchem_cache[name_lower] = result
                        save_cache()
                        return result, "success (PubChem)"
                return None, "PubChem 返回空数据"
            elif r.status_code == 503:
                time.sleep(2.0 * (attempt + 1))
                continue
            elif r.status_code == 404:
                return None, "PubChem 未找到该化合物 (404)"
            else:
                time.sleep(1.0 * (attempt + 1))
                continue
        except requests.exceptions.SSLError:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, "SSL 连接失败"
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, "查询超时，PubChem 服务器无响应"
        except requests.exceptions.ConnectionError:
            time.sleep(1.0 * (attempt + 1))
            continue
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, f"网络异常: {str(e)}"
    return None, "PubChem 持续不可用，请稍后重试"
