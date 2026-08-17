"""Run once locally to create a test tenant + API key.

    python scripts/register_tenant.py tenant_demo            # server-generated key
    python scripts/register_tenant.py tenant_demo mykey123   # fixed key, local only

The one-argument form is what you want: it mints a real key the same way the
admin API does. The two-argument form exists only so local fixtures and docs can
pin a known value — never use it for anything reachable from outside your machine.
"""

import sys

sys.path.insert(0, ".")

from app.core.tenant_store import create_key, register_tenant_key
from app.db.postgres import init_schema

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        raise SystemExit(1)

    init_schema()
    tenant_id = sys.argv[1]

    if len(sys.argv) == 3:
        api_key = sys.argv[2]
        register_tenant_key(api_key, tenant_id, name="local-fixture")
        print(f"Registered tenant '{tenant_id}' with supplied key '{api_key}'")
    else:
        api_key, info = create_key(tenant_id, name="local")
        print(f"Registered tenant '{tenant_id}'")
        print(f"  key: {api_key}")
        print(f"  id:  {info.id}  (use this to revoke: DELETE /v1/admin/tenants/{tenant_id}/keys/{info.id})")
        print("  This is the only time the key is shown.")
