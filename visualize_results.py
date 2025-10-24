import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle

# Load dataset
import pandas as pd

df = pd.read_csv(r"C:\Users\choud\OneDrive\Documents\flask\combined_dataset.csv").dropna(subset=['message','label'])
print(df.head())


# 1. Label distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='label', palette='viridis')
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# 2. Message length distribution
df['msg_length'] = df['message'].apply(len)
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='msg_length', hue='label', bins=30, kde=True, palette='magma')
plt.title("Message Length Distribution")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()

# 3. Top words in smishing vs legitimate
def get_top_words(messages, n=20):
    words = " ".join(messages).split()
    return Counter(words).most_common(n)

top_smish = get_top_words(df[df['label']=='smishing']['message'])
top_legit = get_top_words(df[df['label']=='legitimate']['message'])

print("Top words in smishing messages:")
print(top_smish)
print("\nTop words in legitimate messages:")
print(top_legit)

# 4. Confusion matrix (if model files exist)
try:
    with open("smishing_svm.pkl","rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl","rb") as f:
        vectorizer = pickle.load(f)

    X = vectorizer.transform(df['message'])
    y_true = df['label']
    y_pred = model.predict(X)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
except FileNotFoundError:
    print("Model or vectorizer not found, skipping confusion matrix.")
