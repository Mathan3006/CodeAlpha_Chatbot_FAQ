import json
import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load FAQ data from JSON
with open("input.json", "r") as f:
    faq_data = json.load(f)

# Extract questions and answers
questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# Text preprocessing
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Preprocess all questions
preprocessed_questions = [preprocess(q) for q in questions]

# Vectorize with TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

# Get best match response
def get_answer(user_input):
    user_input_processed = preprocess(user_input)
    user_vec = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    best_match_index = similarity.argmax()
    if similarity[0][best_match_index] < 0.3:
        return "Sorry, I don't understand that."
    return answers[best_match_index]

# Handle user input
def ask():
    user_q = entry.get()
    if user_q.strip() == "":
        return
    response = get_answer(user_q)
    chat.config(state=tk.NORMAL)
    chat.insert(tk.END, "You: " + user_q + "\n")
    chat.insert(tk.END, "Bot: " + response + "\n\n")
    chat.config(state=tk.DISABLED)
    chat.yview(tk.END)
    entry.delete(0, tk.END)

# ----------------- GUI -----------------
root = tk.Tk()
root.title("FAQ Chatbot - CodeAlpha")
root.geometry("550x600")

# Chat frame with scrollbar
chat_frame = tk.Frame(root)
chat_frame.pack(pady=10, padx=10, expand=True, fill='both')

scrollbar = tk.Scrollbar(chat_frame)
scrollbar.pack(side=tk.RIGHT, fill='y')

chat = tk.Text(chat_frame, yscrollcommand=scrollbar.set, bg="white", font=("Helvetica", 12), state=tk.DISABLED)
chat.pack(side=tk.LEFT, expand=True, fill='both')

scrollbar.config(command=chat.yview)

# Entry box
entry = tk.Entry(root, font=("Helvetica", 14))
entry.pack(pady=5, padx=10, fill='x')
entry.bind("<Return>", lambda event: ask())  # Press Enter to send

# Send button
send_btn = tk.Button(root, text="Send", command=ask, bg="blue", fg="white", font=("Helvetica", 12))
send_btn.pack(pady=5)

# Launch the GUI
root.mainloop()
