import requests
import time
import csv
import os

# Flask API endpoint
URL = "http://127.0.0.1:5000/predict"

# Test SMS messages
tests = [
    "Congratulations! You have won a free iPhone. Click here to claim.",
    "Your account is suspended. Verify at http://phish.example now.",
    "Hey bro, kal milte hain?",
    "Free entry to concert — reply YES to claim",
    "Reminder: Submit assignment before midnight"
]

# CSV file to save results
output_file = "test_api_results.csv"

print("🚀 Sending tests to", URL)
results = []

for t in tests:
    try:
        # Send POST request with JSON payload
        r = requests.post(URL, json={"message": t}, timeout=6)
        print("\n📩 Message:", t)

        if r.status_code == 200:
            data = r.json()
            prediction = data.get("prediction")
            score = data.get("decision_score")
            reason = data.get("reason")

            print(f"✅ Prediction: {prediction}")
            print(f"   Score: {score}")
            print(f"   Reason: {reason}")

            results.append({
                "message": t,
                "prediction": prediction,
                "score": score,
                "reason": reason
            })
        else:
            print(f"⚠️ Error: {r.status_code}, Response: {r.text[:200]}")
            results.append({
                "message": t,
                "prediction": None,
                "score": None,
                "reason": f"HTTP {r.status_code}"
            })

    except requests.exceptions.Timeout:
        print("❌ Connection Timeout")
        results.append({
            "message": t,
            "prediction": None,
            "score": None,
            "reason": "Timeout"
        })
    except requests.exceptions.ConnectionError as ce:
        print("❌ Connection Error:", ce)
        results.append({
            "message": t,
            "prediction": None,
            "score": None,
            "reason": "Connection Error"
        })
    except Exception as e:
        print("❌ Other Error:", e)
        results.append({
            "message": t,
            "prediction": None,
            "score": None,
            "reason": str(e)
        })

    time.sleep(0.3)  # slight delay to avoid spamming server

# Save results to CSV
if results:
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["message", "prediction", "score", "reason"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n📂 Results saved to {os.path.abspath(output_file)}")
