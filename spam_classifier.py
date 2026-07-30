import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def load_dataset(path="spam.csv"):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() == "category":
                continue
            category = row[0].strip()
            message = ",".join(row[1:]).strip() if len(row) > 1 else ""
            rows.append((category, message))

    data = pd.DataFrame(rows, columns=["Category", "Message"])
    data["Category"] = data["Category"].astype(str).str.strip()
    data["Message"] = data["Message"].fillna("").astype(str).str.strip()
    data = data[(data["Category"] != "") & (data["Message"] != "")]
    return data


# Load dataset
data = load_dataset()

# Features and labels
X = data["Message"]
y = data["Category"]

# Convert text into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# User input
print("\n------ Email Spam Classifier ------")
message = input("Enter Email Message: ").strip()

if not message:
    print("Please enter a message.")
    raise SystemExit(0)

message_vector = vectorizer.transform([message])
prediction = model.predict(message_vector)

print("\nPrediction:", prediction[0].upper())

if prediction[0] == "spam":
    print("⚠️ This email is SPAM.")
else:
    print("✅ This email is NOT SPAM.")