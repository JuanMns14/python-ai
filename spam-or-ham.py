# Importar las bibliotecas necesarias
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# Cargar los datos desde el archivo CSV
df = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
df.columns = ["label", "text"]

punctuation = set(string.punctuation)


def tokenize(sentence):
    tokens = []
    for token in sentence.split():
        new_token = []
        for char in token:
            if char not in punctuation:
                new_token.append(char.lower())
        if new_token:
            tokens.append("".join(new_token))
    return tokens


train_text, test_text, train_labels, test_labels = train_test_split(
    df["text"], df["label"], stratify=df["label"]
)
print(f"Training examples: {len(train_text)}, testing examples {len(test_text)}")

vectorizer = CountVectorizer(tokenizer = tokenize, binary=True, token_pattern=None)
train_X = vectorizer.fit_transform(train_text)
test_X = vectorizer.transform(test_text)

classifier = LinearSVC()
classifier.fit(train_X, train_labels)

predictions = classifier.predict(test_X)
accuracy = accuracy_score(test_labels, predictions)

print(f"Accuracy: {accuracy:.4%}")

spam = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize!"
ham = "Can we reschedule our call to Monday morning? I'm tied up with meetings today."
examples = [spam, ham]

examples_X = vectorizer.transform(examples)
predictions = classifier.predict(examples_X)

for text, label in zip(examples, predictions):
  print(f"{label:2} - {text}")