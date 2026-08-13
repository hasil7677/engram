"""Run once locally to create a test tenant + API key.

Usage: python scripts/register_tenant.py mykey123 tenant_demo
"""
import sys

sys.path.insert(0, ".")

from app.core.tenant_store import register_tenant_key

if __name__ == "__main__":
    api_key, tenant_id = sys.argv[1], sys.argv[2]
    register_tenant_key(api_key, tenant_id)
    print(f"Registered tenant '{tenant_id}' with key '{api_key}'")
