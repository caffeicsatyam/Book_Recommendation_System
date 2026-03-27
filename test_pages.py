import requests

urls = ["http://127.0.0.1:5000/", "http://127.0.0.1:5000/index", "http://127.0.0.1:5000/Book_Library"]

for url in urls:
    try:
        response = requests.get(url)
        print(f"URL: {url} | Status Code: {response.status_code}")
        if response.status_code == 200:
            if "BOOK BAY" in response.text:
                print(f"  Success: 'BOOK BAY' branding found.")
            else:
                print(f"  Warning: 'BOOK BAY' branding NOT found.")
        else:
            print(f"  Error: Page returned {response.status_code}")
    except Exception as e:
        print(f"  Error accessing {url}: {e}")
