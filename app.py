import pandas as pd
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
import datetime  # ✅ Added for timestamping unanswered queries

# -------------------- Load Dataset --------------------
data = pd.read_csv('data_03.csv')  # Your updated CSV file

# -------------------- Prepare Data --------------------
user_utterances = data['user_utterances'].tolist()
responses = data['response'].tolist()
intents = data['intent'].tolist()

# -------------------- Load Sentence Transformer --------------------
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and accurate

# -------------------- Encode All Utterances --------------------
utterance_embeddings = model.encode(user_utterances, convert_to_tensor=True)

# -------------------- Flask App Setup --------------------
app = Flask(__name__)

# -------------------- Response Fetch Function --------------------
def get_response(user_input):
    # ---------- Step 1: SentenceTransformer Matching ----------
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, utterance_embeddings)[0]
    top_result_idx = cosine_scores.argmax().item()
    top_score = cosine_scores[top_result_idx].item()
    print(f"[INFO] BERT Match Score: {top_score:.2f}, Matched Utterance: '{user_utterances[top_result_idx]}'")  # Debug

    # ---------- Step 2: RapidFuzz Similarity Matching ----------
    rapid_scores = [fuzz.token_sort_ratio(user_input.lower(), utt.lower()) for utt in user_utterances]
    rapid_top_idx = rapid_scores.index(max(rapid_scores))
    rapid_top_score = max(rapid_scores)
    print(f"[INFO] RapidFuzz Score: {rapid_top_score}, Matched Utterance: '{user_utterances[rapid_top_idx]}'")  # Debug

    # ---------- Step 3: Decision Making ----------
    if top_score > 0.55:  # Semantic Match threshold
        return responses[top_result_idx]
    elif rapid_top_score > 75:  # Fuzzy Match threshold
        return responses[rapid_top_idx]
    else:
        # ---------- ✅ Log Unanswered Query ----------
        with open('unanswered_queries.txt', 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {user_input}\n")
        return "I'm sorry, I couldn't understand that. Could you please rephrase or contact us at 8887776660?"

# -------------------- Flask Routes --------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index_chatboat.html')

@app.route('/course')
def course():
    return render_template('course.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog-single.html')

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

# -------------------- Chatbot API --------------------
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    response = get_response(user_input)
    return jsonify({'response': response})

# -------------------- Run Flask App --------------------
if __name__ == '__main__':
    app.run(debug=True)
