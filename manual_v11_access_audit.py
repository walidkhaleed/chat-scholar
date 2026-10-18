
from fastapi.testclient import TestClient
from main import app

client = TestClient(
    app,
    raise_server_exceptions=False
)

endpoints = [
    "/health",
    "/documents",
    "/rubrics",
    "/pdf-sessions",
]

print("\n=== CHAT SCHOLAR ACCESS AUDIT ===\n")

for endpoint in endpoints:
    response = client.get(endpoint)

    print(
        f"{endpoint:25} "
        f"HTTP {response.status_code}"
    )

    if endpoint != "/health":
        if response.status_code == 200:
            print(
                "  WARNING: Endpoint accessible "
                "without authentication"
            )
        elif response.status_code in (401, 403):
            print("  Protected against anonymous access")
        else:
            print("  Review response and endpoint behavior")

print("\n=== AUDIT FINISHED ===")
