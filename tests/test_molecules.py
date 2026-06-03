"""Tests for molecules.py — local DB, search index, PubChem cache."""

import json
import os
import pytest

from molecules import (
    _load_molecule_db,
    build_search_index,
    MOLECULE_DB,
    SEARCH_INDEX,
    load_cache,
    save_cache,
    pubchem_cache,
)


class TestMoleculeDatabase:
    """Tests for the local molecule database."""

    def test_db_loaded(self):
        """MOLECULE_DB should be a non-empty dict."""
        assert isinstance(MOLECULE_DB, dict)
        assert len(MOLECULE_DB) > 0

    def test_custom_input_entry(self):
        """"(自定义输入)" key should exist in the database."""
        assert "(自定义输入)" in MOLECULE_DB

    def test_db_contains_common_molecules(self):
        """Database should contain well-known molecules (bilingual keys)."""
        names_combined = " ".join(MOLECULE_DB.keys()).lower()
        common = {"ethanol", "aspirin", "caffeine", "benzene"}
        found = [c for c in common if c in names_combined]
        assert len(found) >= 3, f"Only found {found} out of {common} in keys"

    def test_smiles_are_valid_strings(self):
        """All SMILES values should be non-empty strings or the custom input marker."""
        for name, smiles in MOLECULE_DB.items():
            if name == "(自定义输入)":
                assert smiles == ""
            else:
                assert isinstance(smiles, str)
                assert len(smiles) > 0

    def test_db_file_exists(self):
        """The molecule_db.json file should exist."""
        path = os.path.join(os.path.dirname(__file__), "..", "data", "molecule_db.json")
        assert os.path.exists(path), f"DB file not found at {path}"

    def test_db_file_valid_json(self):
        """The molecule_db.json file should be valid JSON."""
        path = os.path.join(os.path.dirname(__file__), "..", "data", "molecule_db.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert len(data) >= 100  # Expect at least 100 molecules


class TestBuildSearchIndex:
    """Tests for the search index."""

    def test_index_is_dict(self):
        """Search index should be a non-empty dict."""
        assert isinstance(SEARCH_INDEX, dict)
        assert len(SEARCH_INDEX) > 0

    def test_lowercase_index(self):
        """Search index keys should be lowercase."""
        for key in SEARCH_INDEX:
            assert key == key.lower(), f"Key '{key}' is not lowercase"

    def test_index_contains_single_word_parts(self):
        """Index should contain individual word parts (e.g. 'ethanol')."""
        assert "ethanol" in SEARCH_INDEX
        assert "aspirin" in SEARCH_INDEX
        assert "caffeine" in SEARCH_INDEX

    def test_all_index_values_valid_smiles(self):
        """All index values (except custom input) should be valid SMILES or empty."""
        from rdkit import Chem
        potential_issues = []
        for name, smiles in SEARCH_INDEX.items():
            if name == "(自定义输入)":
                continue
            if smiles == "":
                continue  # Custom input alias
            # Some index entries might point to the same SMILES — just check it parses
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                potential_issues.append((name, smiles))
        assert len(potential_issues) == 0, (
            f"Found {len(potential_issues)} invalid SMILES in index: {potential_issues[:3]}"
        )


class TestPubChemCache:
    """Tests for PubChem cache operations."""

    CACHE_TEST_FILE = "test_pubchem_cache.json"

    def setup_method(self):
        """Clean up any leftover test cache file."""
        if os.path.exists(self.CACHE_TEST_FILE):
            os.remove(self.CACHE_TEST_FILE)

    def teardown_method(self):
        """Clean up test cache file after each test."""
        if os.path.exists(self.CACHE_TEST_FILE):
            os.remove(self.CACHE_TEST_FILE)

    def test_load_cache_no_file(self):
        """load_cache should not error if cache file doesn't exist."""
        if os.path.exists(self.CACHE_TEST_FILE):
            os.remove(self.CACHE_TEST_FILE)
        # Should not raise
        load_cache()

    def test_save_and_load_roundtrip(self, monkeypatch):
        """Cache should survive a save/load roundtrip."""
        monkeypatch.setattr("molecules.CACHE_FILE", self.CACHE_TEST_FILE)
        monkeypatch.setattr("molecules.pubchem_cache", {})

        # Modify cache
        import molecules
        molecules.pubchem_cache["test"] = "SMILES"
        molecules.save_cache()

        # Reload
        molecules.pubchem_cache = {}
        molecules.load_cache()
        assert molecules.pubchem_cache.get("test") == "SMILES"

    def test_corrupted_cache_does_not_crash(self, monkeypatch):
        """A corrupted cache file should be handled gracefully (empty cache)."""
        monkeypatch.setattr("molecules.CACHE_FILE", self.CACHE_TEST_FILE)
        # Write invalid JSON
        with open(self.CACHE_TEST_FILE, "w") as f:
            f.write("{{{bad json}}")

        import molecules
        molecules.pubchem_cache = {}
        molecules.load_cache()
        assert len(molecules.pubchem_cache) == 0

    def test_save_cache_io_error_does_not_crash(self, monkeypatch):
        """If save_cache fails due to IOError, it should silently pass."""
        def failing_write(*args, **kwargs):
            raise OSError("Permission denied")

        monkeypatch.setattr("builtins.open", failing_write)
        save_cache()  # Should not raise


class TestLoadMoleculeDb:
    """Tests for _load_molecule_db()."""

    def test_successful_load(self):
        """Should load the actual molecule database successfully."""
        db = _load_molecule_db()
        assert isinstance(db, dict)
        assert len(db) > 0
        assert "(自定义输入)" in db

    def test_file_not_found_graceful(self, monkeypatch):
        """Missing file should return minimal dict."""
        monkeypatch.setattr("molecules.MOLECULE_DB_PATH", "/nonexistent/path.json")
        db = _load_molecule_db()
        assert db == {"(自定义输入)": ""}
