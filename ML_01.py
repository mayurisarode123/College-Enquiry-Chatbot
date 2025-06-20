import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv('data_02.csv')

# Preprocess the data
X = data['user_utterances']
y = data['intent']

# Vectorization
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train the model
model = MultinomialNB()
model.fit(X_vectorized, y)

def get_response(user_input):
    user_input_vectorized = vectorizer.transform([user_input])
    predicted_intent = model.predict(user_input_vectorized)[0]
    response = data[data['intent'] == predicted_intent]['response'].values[0]
    return response

def main():
    print("Welcome to the College Chatbot! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == '__main__':
    main()
