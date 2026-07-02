import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
train = pd.read_csv(r"C:\Users\aiswa\OneDrive\Desktop\python project\data train set.txt",
                    sep=';', names=['text', 'emotion'])
test = pd.read_csv(r"C:\Users\aiswa\OneDrive\Desktop\python project\dataset test 1.txt",
                   sep=';', names=['text', 'emotion'])

print("\n✅ Sample of Training Data:")
print(train.head())

plt.figure(figsize=(8,5))
sns.countplot(data=train, x='emotion', color='skyblue')
plt.title("Emotion Distribution in Training Data")
plt.show()
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)    
    text = re.sub(r"[^a-z'\s]", "", text)          
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words or w == 'not']
    return " ".join(words)

train["clean_text"] = train["text"].apply(clean_text)
test["clean_text"] = test["text"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2)
X_train = vectorizer.fit_transform(train["clean_text"])
X_test = vectorizer.transform(test["clean_text"])
y_train = train["emotion"]
y_test = test["emotion"]

model = LogisticRegression(max_iter=4000, class_weight="balanced", solver="saga", C=2)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\n✅ Model Evaluation Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

def predict_emotion(sentence):
    clean = clean_text(sentence)
    vec = vectorizer.transform([clean])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    top_indices = np.argsort(probs)[::-1][:3]
    top_emotions = [(classes[i], probs[i]*100) for i in top_indices]
    return top_emotions

print("\n Emotion Detection Ready! Type 'exit' anytime to quit.\n")
while True:
    user_input = input("Enter a sentence to detect emotion (or 'exit'): ")
    if user_input.lower() == "exit":
        print("👋 Exiting Emotion Detector. Thank you!")
        break

    top3 = predict_emotion(user_input)
    print(f"\n Detected Emotion: {top3[0][0]} (Confidence: {top3[0][1]:.2f}%)")
    print("Other possible emotions:")
    for e, c in top3[1:]:
        print(f"   • {e} ({c:.2f}%)")
    print()
