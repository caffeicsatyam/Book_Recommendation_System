from app import app
import sys

with app.test_client() as c:
    try:
        r = c.get('/Book_Library')
        print(f"Status Code: {r.status_code}")
        if r.status_code != 200:
            print("Response text:", r.get_data(as_text=True)[:2000])
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
