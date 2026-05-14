import requests
import urllib.parse
import time
import json
import os

CACHE_FILE = "pubchem_cache.json"
pubchem_cache = {}

def load_cache():
    global pubchem_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                pubchem_cache = json.load(f)
        except:
            pubchem_cache = {}

def save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(pubchem_cache, f, ensure_ascii=False, indent=2)
    except:
        pass

load_cache()

def search_pubchem_final(name, max_retries=3):
    """
    最终版 PubChem 搜索（基于知乎文章技巧优化）
    - verify=False: 跳过 SSL 验证（解决国内网络握手失败）
    - time.sleep(1): 严格符合官方 1 req/s 限制
    - 本地缓存 + 容错处理
    """
    if not name or not name.strip():
        return None, "名称不能为空"
    
    name_clean = name.strip()
    name_lower = name_clean.lower()
    
    # 0. 查缓存
    if name_lower in pubchem_cache:
        return pubchem_cache[name_lower], "success (cached)"
    
    # 1. 频率控制（知乎文章用 time.sleep(1)，官方限制 5 req/s，这里用 1.2s 更保险）
    time.sleep(1.2)
    
    encoded = urllib.parse.quote(name_clean)
    
    # 2. 核心请求（加入 verify=False，这是知乎文章的关键技巧！）
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/JSON"
    
    for attempt in range(max_retries):
        try:
            # verify=False 跳过 SSL 证书验证，解决国内网络握手问题
            r = requests.get(url, timeout=20, verify=False)
            
            if r.status_code == 200:
                data = r.json()
                
                # 检查 Fault
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
                wait = 2.0 * (attempt + 1)
                print(f"  ⚠️ 503 服务器繁忙，等待 {wait}s 后重试...")
                time.sleep(wait)
                continue
            
            elif r.status_code == 404:
                return None, "PubChem 未找到该化合物 (404)"
            
            else:
                return None, f"PubChem HTTP {r.status_code}: {r.text[:100]}"
                
        except requests.exceptions.SSLError as e:
            # SSL 错误，如果 verify=False 还报错，说明网络层有问题
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, f"SSL 连接失败: {str(e)}"
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, "查询超时，PubChem 服务器无响应"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, f"网络异常: {str(e)}"
    
    return None, "PubChem 持续不可用，请稍后重试"

# 测试
if __name__ == "__main__":
    test_list = ["Aspirin", "Ibuprofen", "Acyclovir", "Unknown12345"]
    for name in test_list:
        print(f"\n🔍 {name}")
        result, status = search_pubchem_final(name)
        print(f"   {status}")
        if result:
            print(f"   SMILES: {result}")
