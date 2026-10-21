
from getpass import getpass

from fastapi.testclient import TestClient
from main import app
from csrf_guard import csrf_enforcement_enabled


def main():
    print("\n=== V11 ACTIVE CSRF SECURITY TEST ===\n")

    if not csrf_enforcement_enabled():
        raise SystemExit(
            "FAIL: CSRF enforcement is not enabled."
        )

    print("CSRF enforcement: ENABLED")

    with TestClient(app) as client:

        # TEST 1: Anonymous access
        response = client.get("/documents")

        assert response.status_code == 401
        print("1. Anonymous access rejected: PASS")

        # TEST 2: Authenticate
        password = getpass(
            "Enter Chat Scholar password: "
        )

        response = client.post(
            "/auth/login",
            json={"password": password}
        )

        assert response.status_code == 200, (
            f"Login failed: HTTP {response.status_code}"
        )

        del password

        print("2. Login successful: PASS")

        # TEST 3: Obtain CSRF token
        response = client.get("/auth/csrf")

        assert response.status_code == 200

        csrf_token = response.json()["csrf_token"]

        assert len(csrf_token) == 43
        print("3. CSRF token received: PASS")

        # TEST 4: POST without CSRF
        response = client.post(
            "/rubrics",
            json={}
        )

        assert response.status_code == 403
        print("4. Missing CSRF rejected: PASS")

        # TEST 5: POST with invalid CSRF
        response = client.post(
            "/rubrics",
            json={},
            headers={
                "X-CSRF-Token": "invalid-token"
            }
        )

        assert response.status_code == 403
        print("5. Invalid CSRF rejected: PASS")

        # TEST 6: POST with valid CSRF
        # Intentionally invalid rubric payload.
        # HTTP 422 proves the request reached
        # FastAPI's normal data validation.
        response = client.post(
            "/rubrics",
            json={},
            headers={
                "X-CSRF-Token": csrf_token
            }
        )

        assert response.status_code == 422
        print("6. Valid CSRF accepted: PASS")

        # TEST 7: GET does not require CSRF
        response = client.get("/rubrics")

        assert response.status_code == 200
        print("7. Authenticated GET works: PASS")

        # TEST 8: Logout with CSRF
        response = client.post(
            "/auth/logout",
            headers={
                "X-CSRF-Token": csrf_token
            }
        )

        assert response.status_code == 200
        print("8. CSRF-protected logout: PASS")

        # TEST 9: Access after logout
        response = client.get("/documents")

        assert response.status_code == 401
        print("9. Access after logout rejected: PASS")

    print("\n=== ALL 9 CHECKS PASSED ===")


if __name__ == "__main__":
    main()
