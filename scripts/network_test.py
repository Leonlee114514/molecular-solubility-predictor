"""
网络诊断脚本 - 测试能否访问 PubChem
"""

import requests
import urllib.parse

print("=" * 50)
print("🔍 网络诊断开始")
print("=" * 50)

# 测试1：访问百度（验证基础网络）
print("\n【测试1】访问百度...")
try:
    r = requests.get("https://www.baidu.com", timeout=10)
    print(f"   结果: HTTP {r.status_code}")
    if r.status_code == 200:
        print("   ✅ 基础网络正常")
    else:
        print("   ⚠️ 基础网络异常")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试2：访问 PubChem（直接）
print("\n【测试2】访问 PubChem 主站...")
try:
    r = requests.get("https://pubchem.ncbi.nlm.nih.gov", timeout=15)
    print(f"   结果: HTTP {r.status_code}")
    if r.status_code == 200:
        print("   ✅ PubChem 主站可访问")
    else:
        print(f"   ⚠️ PubChem 返回 HTTP {r.status_code}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试3：访问 PubChem API（查询 Aspirin）
print("\n【测试3】访问 PubChem API（查 Aspirin）...")
try:
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/Aspirin/property/IsomericSMILES/JSON"
    r = requests.get(url, timeout=15)
    print(f"   结果: HTTP {r.status_code}")
    print(f"   返回内容前200字: {r.text[:200]}")
    if r.status_code == 200:
        print("   ✅ PubChem API 正常")
    else:
        print(f"   ⚠️ PubChem API 异常")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试4：访问 Kimi API（验证另一个国外服务）
print("\n【测试4】访问 Kimi API 服务器...")
try:
    r = requests.get("https://api.moonshot.cn", timeout=10)
    print(f"   结果: HTTP {r.status_code}")
    if r.status_code in [200, 404]:
        print("   ✅ Kimi 服务器可访问")
    else:
        print(f"   ⚠️ 返回 HTTP {r.status_code}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

print("\n" + "=" * 50)
print("诊断完成，请把上面的结果复制给我")
print("=" * 50)
