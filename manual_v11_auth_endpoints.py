
from getpass import getpass

from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

print("\n=== V11 LOGIN ENDPOINT TEST ===\n")

# Test 1: Anonymous access
response = client.get("/auth/me")
print("Anonymous user rejected:", response.status_code == 401)

# Test 2: Incorrect password
response = client.post(
    "/auth/login",
    json={"password": "definitely-wrong-password"}
)
print("Incorrect password rejected:", response.status_code == 401)

# Test 3: Correct password
password = getpass("Enter Chat Scholar password: ")

if not password:
    raise SystemExit(
        "ERROR: No password was captured. Try running this "
        "script directly in an interactive terminal."
    )

response = client.post(
    "/auth/login",
    json={"password": password}
)

print("Login HTTP status:", response.status_code)
print("Correct password accepted:", response.status_code == 200)

if response.status_code != 200:
    if response.status_code == 422:
        print("Login request failed input validation.")
    elif response.status_code == 401:
        print("Password did not match the saved password hash.")
    else:
        print("Unexpected login error.")
    raise SystemExit(1)

# Test 4: Authenticated session
response = client.get("/auth/me")
print("Authenticated session accepted:", response.status_code == 200)

# Test 5: Logout
response = client.post("/auth/logout")
print("Logout successful:", response.status_code == 200)

# Test 6: Session after logout
response = client.get("/auth/me")
print("Session rejected after logout:", response.status_code == 401)

print("\n=== TEST COMPLETE ===")
