"""
DisSolve - Centralised session state key constants.
Use these instead of raw strings to avoid typos across modules.
"""


class StateKey:
    # ── Core prediction state ──
    SMILES_INPUT = "smiles_input_box"
    PREDICTED_SMILES = "predicted_smiles"
    PREDICTED_LOGS = "predicted_logS"
    PREDICTED_PKA = "predicted_pka"
    AI_EXPLANATION = "ai_explanation"
    CACHED_FEATURES = "cached_features"
    SHAP_VALUES = "shap_values"
    SHAP_NAMES = "shap_names"
    OOD_RISK = "ood_risk"
    OOD_RESULT = "ood_result"

    # ── Search state ──
    SEARCH_STATE = "search_state"
    SEARCH_QUERY = "search_query"
    FUZZY_MATCHES = "fuzzy_matches"
    FUZZY_BEST = "fuzzy_best"
    FUZZY_SMILES = "fuzzy_smiles"

    # ── Widget keys ──
    MOL_SEARCH_FILTER = "mol_search_filter"
    MOLECULE_SELECT_RADIO = "molecule_select_radio"
    SEARCH_NAME = "search_name"
    SEARCH_BTN = "search_btn"
    USE_FUZZY_MATCH = "use_fuzzy_match"
    SKIP_TO_PUBCHEM = "skip_to_pubchem"
    CLEAR_AI = "clear_ai"
    GEN_AI = "gen_ai"

    # ── Model selection ──
    SELECTED_MODEL = "selected_model_type"
    PREDICTED_LOGS_RF = "predicted_logS_rf"
    PREDICTED_LOGS_GNN = "predicted_logS_gnn"

    # ── History ──
    PREDICTION_HISTORY = "prediction_history"

    # ── Batch prediction ──
    BATCH_RESULTS = "batch_results"

    # ── Molecule name tracking ──
    CURRENT_MOLECULE_NAME = "_current_molecule_name"

    # ── Internal ──
    TARGET_TAB = "_target_tab"
