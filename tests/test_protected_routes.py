
from getpass import getpass

from fastapi.testclient import TestClient
from main import app


PROTECTED_ENDPOINTS = [
    "/documents",
    "/rubrics",
    "/pdf-sessions",
]


def check_endpoints(client, expected_status):
    all_passed = True

    for endpoint in PROTECTED_ENDPOINTS:
        response = client.get(endpoint)

        passed = response.status_code == expected_status

        print(
            f"{endpoint:20} "
            f"HTTP {response.status_code} "
            f"| Expected {expected_status} "
            f"| {'PASS' if passed else 'FAIL'}"
        )

        if not passed:
            all_passed = False

    return all_passed


def main():
    print("\n=== V11 PROTECTED ROUTES TEST ===\n")

    all_passed = True

    with TestClient(app) as client:

        # TEST 1: Anonymous access
        print("[1] Anonymous access")

        result = check_endpoints(
            client,
            expected_status=401
        )

        all_passed = all_passed and result

        # TEST 2: Login
        print("\n[2] Login")

        password = getpass(
            "Enter Chat Scholar password: "
        )

        if not password:
            raise SystemExit(
                "No password entered."
            )

        response = client.post(
            "/auth/login",
            json={"password": password}
        )

        login_ok = response.status_code == 200

        print(
            "Login:",
            "PASS" if login_ok else "FAIL",
            f"(HTTP {response.status_code})"
        )

        if not login_ok:
            raise SystemExit(
                "Login failed. Stopping tests."
            )

        # TEST 3: Authenticated access
        print("\n[3] Authenticated access")

        result = check_endpoints(
            client,
            expected_status=200
        )

        all_passed = all_passed and result

        # TEST 4: Logout
        print("\n[4] Logout")

        response = client.post("/auth/logout")

        logout_ok = response.status_code == 200

        print(
            "Logout:",
            "PASS" if logout_ok else "FAIL"
        )

        all_passed = all_passed and logout_ok

        # TEST 5: Access after logout
        print("\n[5] Access after logout")

        result = check_endpoints(
            client,
            expected_status=401
        )

        all_passed = all_passed and result

    print("\n=== FINAL RESULT ===")

    if all_passed:
        print("ALL AUTHENTICATION TESTS PASSED")
    else:
        print("ONE OR MORE TESTS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
