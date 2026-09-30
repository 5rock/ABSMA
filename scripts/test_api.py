import requests

base_url = "http://127.0.0.1:5000"

def test_endpoints():
    print("Testing GET /")
    r = requests.get(base_url + "/")
    print("Status:", r.status_code)

    print("\nTesting GET /health")
    r = requests.get(base_url + "/health")
    print("Status:", r.status_code)
    print("Body:", r.json())

    # Create dummy user
    print("\nTesting POST /api/auth/register")
    r = requests.post(base_url + "/api/auth/register", json={
        "name": "Test User",
        "email": "testuser@example.com",
        "password": "password123"
    })
    print("Status:", r.status_code)
    print("Body:", r.json())

    # Login
    print("\nTesting POST /api/auth/login")
    session = requests.Session()
    r = session.post(base_url + "/api/auth/login", json={
        "email": "testuser@example.com",
        "password": "password123"
    })
    print("Status:", r.status_code)
    print("Body:", r.json())

    # Me
    print("\nTesting GET /api/auth/me")
    r = session.get(base_url + "/api/auth/me")
    print("Status:", r.status_code)
    print("Body:", r.json())

    # Analyze
    print("\nTesting POST /api/analyze")
    r = session.post(base_url + "/api/analyze", json={
        "review_text": "The lead actor was brilliant, but the ending was disappointing."
    })
    print("Status:", r.status_code)
    print("Body:", r.json())

    # History
    print("\nTesting GET /api/analysis/history")
    r = session.get(base_url + "/api/analysis/history")
    print("Status:", r.status_code)
    print("Body:", r.json())

    # Logout
    print("\nTesting POST /api/auth/logout")
    r = session.post(base_url + "/api/auth/logout")
    print("Status:", r.status_code)
    print("Body:", r.json())

if __name__ == "__main__":
    test_endpoints()
