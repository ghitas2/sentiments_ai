from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
text = [
    "I love learning AI",
    "I hate moving in",
    "I am excited about my new life",
    "I am scared of new beginnings"
]

labels = [
    "positive",
    "negative",
    "positive",
    "negative"
]

# Convert text to vectors
vectorizer = CountVectorizer()  #Class CountVectorizer
X = vectorizer.fit_transform(text)

# Train the model
model = MultinomialNB()
model.fit(X, labels)

# Predict new data
test_text = ["Ice cream"]
X_test = vectorizer.transform(test_text)  # Use the same 'vectorizer' here
prediction = model.predict(X_test)

print(f"Prediction: {prediction[0]}")
