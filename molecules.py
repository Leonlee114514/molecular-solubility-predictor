"""
DisSolve - Local molecule database and PubChem search integration.
"""

import json
import os
import time
import urllib.parse

import requests

# ========== 本地分子库（从 JSON 文件加载）==========
MOLECULE_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "molecule_db.json")

def _load_molecule_db():
    """Load the molecule database from JSON file."""
    try:
        with open(MOLECULE_DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
        if "(自定义输入)" not in db:
            db["(自定义输入)"] = ""
        return db
    except (FileNotFoundError, json.JSONDecodeError):
        return {"(自定义输入)": ""}

MOLECULE_DB = _load_molecule_db()


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
                    smiles = props[0].get('CanonicalSMILES') or props[0].get('IsomericSMILES') or props[0].get('ConnectivitySMILES')
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
