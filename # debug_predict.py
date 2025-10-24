import requests, time

URL = "http://127.0.0.1:5000/predict"

# Test SMS messages
tests = [
    "Congratulations! You have won a free iPhone. Click here to claim.",
    "Your account is suspended. Verify at http://phish.example now.",
    "Hey bro, kal milte hain?",
    "Free entry to concert — reply YES to claim",
    "Reminder: Submit assignment before midnight"
]

print("🚀 Sending test messages to", URL)

for t in tests:
    try:
        r = requests.post(URL, json={"message": t}, timeout=6)
        print("\n📩 Message:", t)
        if r.status_code == 200:
            data = r.json()
            prediction = data.get("prediction", "No prediction")
            print("✅ Prediction:", prediction)
        else:
            print(f"⚠️ Server returned error ({r.status_code}):", r.text)
    except Exception as e:
        print("❌ Connection Error:", e)
    time.sleep(0.3)
