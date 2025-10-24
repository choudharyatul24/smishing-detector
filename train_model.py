import requests
import time
import csv
import os

URL = "http://127.0.0.1:5000/predict"

tests = [
    "Congratulations! You have won a free iPhone. Click here to claim.",
    "Your account is suspended. Verify at http://phish.example now.",
    "Hey bro, kal milte hain?",
    "Free entry to concert — reply YES to claim",
    "Reminder: Submit assignment before midnight"
]

output_file = "test_api_results.csv"

results = []

print("🚀 Sending tests to", URL)

for t in tests:
    try:
        r = requests.post(URL, json={"message": t}, timeout=6)
        print("\n📩 Message:", t)

        if r.status_code == 200:
            data = r.json()
            results.append({
                "message": t,
                "prediction": data.get("prediction"),
                "score": data.get("decision_score"),
                "reason": data.get("reason")
            })
            print("✅ Prediction:", data.get("prediction"))
            print("   Score:", data.get("decision_score"))
            print("   Reason:", data.get("reason"))
        else:
            print(f"⚠ Error: {r.status_code}")
            results.append({"message": t, "prediction": None, "score": None, "reason": f"HTTP {r.status_code}"})

    except requests.exceptions.RequestException as e:
        print("❌ Connection Error:", e)
        results.append({"message": t, "prediction": None, "score": None, "reason": "Connection Error"})

    time.sleep(0.3)

# Save results
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["message", "prediction", "score", "reason"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n📂 Results saved to {os.path.abspath(output_file)}")
