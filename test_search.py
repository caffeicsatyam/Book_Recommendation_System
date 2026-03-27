import requests

url = "http://127.0.0.1:5000/search"
data = {"search_query": "Harry Potter and the Chamber of Secrets (Book 2)"}

try:
    response = requests.post(url, data=data)
    print(f"Status Code: {response.status_code}")
    # We expect HTML since it renders recommend.html
    if "MATCHED" in response.text and "RECOMMENDED" in response.text:
        print("Success: Found MATCHED and RECOMMENDED labels in response.")
    else:
        print("Response does not contain expected labels.")
        # Print a snippet of the response to see what happened
        print(response.text[:500])
except Exception as e:
    print(f"Error: {e}")
