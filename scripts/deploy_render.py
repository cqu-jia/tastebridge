"""Create (or update) the TasteBridge web service on Render via its API.

Reads QLOO_API_KEY / NEBIUS_API_KEY / NEBIUS_MODEL from .env and asks for a Render API key
without echoing it. No key is ever printed.

Before running: sign in to Render with GitHub, give Render access to the tastebridge repo
(dashboard -> New -> Web Service -> connect GitHub), and create an API key
(Account Settings -> API Keys).

    .venv/bin/python scripts/deploy_render.py
"""
import getpass
import json
import sys
import urllib.error
import urllib.request

from dotenv import dotenv_values

API = "https://api.render.com/v1"
REPO = "https://github.com/cqu-jia/tastebridge"
NAME = "tastebridge"


def call(method, path, token, body=None):
    req = urllib.request.Request(
        API + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"Render API {method} {path} failed ({e.code}): {e.read().decode()[:400]}")


def main():
    env = dotenv_values(".env")
    missing = [k for k in ("QLOO_API_KEY", "NEBIUS_API_KEY", "NEBIUS_MODEL") if not env.get(k)]
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}")
    env_vars = [
        {"key": "QLOO_API_KEY", "value": env["QLOO_API_KEY"]},
        {"key": "QLOO_BASE_URL", "value": env.get("QLOO_BASE_URL") or "https://hackathon.api.qloo.com"},
        {"key": "NEBIUS_API_KEY", "value": env["NEBIUS_API_KEY"]},
        {"key": "NEBIUS_BASE_URL", "value": env.get("NEBIUS_BASE_URL") or "https://api.tokenfactory.nebius.com/v1/"},
        {"key": "NEBIUS_MODEL", "value": env["NEBIUS_MODEL"]},
        {"key": "PYTHON_VERSION", "value": "3.11.9"},
    ]

    token = getpass.getpass("Render API key (input hidden): ").strip()
    owners = call("GET", "/owners?limit=20", token)
    if not owners:
        sys.exit("No Render workspace found for this API key.")
    owner = owners[0]["owner"]
    print(f"Workspace: {owner.get('name')}")

    existing = [s["service"] for s in call("GET", f"/services?name={NAME}&limit=20", token)]
    if existing:
        service = existing[0]
        print("Service exists, updating environment variables and redeploying...")
        call("PUT", f"/services/{service['id']}/env-vars", token, env_vars)
        call("POST", f"/services/{service['id']}/deploys", token, {"clearCache": "do_not_clear"})
    else:
        print("Creating service...")
        service = call("POST", "/services", token, {
            "type": "web_service",
            "name": NAME,
            "ownerId": owner["id"],
            "repo": REPO,
            "branch": "main",
            "autoDeploy": "yes",
            "envVars": env_vars,
            "serviceDetails": {
                "runtime": "python",
                "plan": "free",
                "region": "frankfurt",
                "healthCheckPath": "/health",
                "envSpecificDetails": {
                    "buildCommand": "pip install -r requirements.txt",
                    "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
                },
            },
        })["service"]

    url = (service.get("serviceDetails") or {}).get("url")
    print(f"Done. Build takes 2-5 minutes. Site: {url}")
    print(f"Logs: {service.get('dashboardUrl')}")


if __name__ == "__main__":
    main()
