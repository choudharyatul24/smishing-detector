import joblib

# Absolute path se model aur vectorizer load karo
model = joblib.load("C:/Users/choud/OneDrive/Documents/flask/smishing_svm.pkl")
vectorizer = joblib.load("C:/Users/choud/OneDrive/Documents/flask/tfidf_vectorizer.pkl")

# Test ke liye example message
message = "Congratulations! Click here to claim your free prize"

# Message ko vectorize karo aur predict karo
X_test = vectorizer.transform([message])
prediction = model.predict(X_test)

print("Message:", message)
print("Prediction:", "Smishing" if prediction[0]==1 else "Legitimate")
