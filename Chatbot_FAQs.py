import nltk
import numpy as np
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import json
import os
from datetime import datetime

# Download required NLTK data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading stopwords...")
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab/english')
    except LookupError:
        print("Downloading punkt_tab data...")
        nltk.download('punkt_tab', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading wordnet...")
        nltk.download('wordnet', quiet=True)

# Download NLTK data before proceeding
download_nltk_data()

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# FAQ data - questions and answers (can be loaded from a file in a real application)
faqs = [
    {
        "question": "What is your return policy?",
        "answer": "We offer a 30-day return policy for all unused items in their original packaging. You can initiate a return through your account dashboard or by contacting customer service.",
        "category": "Orders & Returns"
    },
    {
        "question": "How do I track my order?",
        "answer": "Once your order ships, you'll receive a tracking number via email. You can also log into your account and view the order status page to see tracking information and delivery estimates.",
        "category": "Orders & Returns"
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept all major credit cards (Visa, MasterCard, American Express), PayPal, Apple Pay, and Google Pay. All payments are processed securely.",
        "category": "Payment"
    },
    {
        "question": "How can I contact customer service?",
        "answer": "Our customer service team is available Monday to Friday, 9 AM to 6 PM EST. You can reach us by phone at 1-800-123-4567, by email at support@example.com, or through the live chat on our website.",
        "category": "Support"
    },
    {
        "question": "Do you offer international shipping?",
        "answer": "Yes, we ship to over 50 countries worldwide. International shipping rates and delivery times vary depending on the destination. Customs fees may apply for international orders.",
        "category": "Shipping"
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping within the US takes 3-5 business days. Express shipping options are available at checkout for delivery within 1-2 business days. International shipping times vary by location.",
        "category": "Shipping"
    },
    {
        "question": "Can I change or cancel my order?",
        "answer": "You can change or cancel your order within 1 hour of placing it. After that, your order enters our processing system and cannot be modified. If you need to make changes after this window, please contact customer service immediately.",
        "category": "Orders & Returns"
    },
    {
        "question": "Are my personal details secure?",
        "answer": "Yes, we take data security very seriously. All personal and payment information is encrypted using industry-standard SSL technology. We never share your information with third parties without your consent.",
        "category": "Security"
    },
    {
        "question": "Do you have a physical store?",
        "answer": "We currently operate as an online-only retailer. This allows us to keep our prices competitive and offer a wider selection of products. However, we occasionally participate in pop-up events and markets.",
        "category": "Company"
    },
    {
        "question": "How do I create an account?",
        "answer": "You can create an account during checkout by selecting the 'Create Account' option. Alternatively, you can sign up from the login page on our website. Accounts allow you to track orders, save preferences, and checkout faster.",
        "category": "Account"
    },
    {
        "question": "What is your privacy policy?",
        "answer": "We value your privacy and are committed to protecting your personal information. Our privacy policy outlines how we collect, use, and safeguard your data. You can read the full policy on our website.",
        "category": "Security"
    },
    {
        "question": "How do I reset my password?",
        "answer": "To reset your password, click on the 'Forgot Password' link on the login page. Enter your email address, and we'll send you a link to create a new password.",
        "category": "Account"
    }
]

# Text preprocessing function with advanced techniques
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # Rejoin tokens
    processed_text = ' '.join(lemmatized_tokens)
    
    return processed_text

# Prepare FAQ data
faq_questions = [faq["question"] for faq in faqs]
faq_answers = [faq["answer"] for faq in faqs]
faq_categories = list(set([faq["category"] for faq in faqs]))

# Preprocess all FAQ questions
processed_faqs = [preprocess_text(question) for question in faq_questions]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
tfidf_matrix = vectorizer.fit_transform(processed_faqs)

# Function to find the best matching FAQ
def get_response(user_input):
    # Preprocess user input
    processed_input = preprocess_text(user_input)
    
    # Vectorize user input
    input_vector = vectorizer.transform([processed_input])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(input_vector, tfidf_matrix)
    
    # Get the index of the most similar FAQ
    best_match_index = similarities.argmax()
    best_match_score = similarities[0, best_match_index]
    
    # Set a threshold for matching
    if best_match_score > 0.2:
        return faq_answers[best_match_index], best_match_score, faqs[best_match_index]["category"]
    else:
        return "I'm sorry, I don't have an answer for that question. Please try rephrasing or contact our customer service for assistance.", best_match_score, "Unknown"

# Create GUI
class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("IntelliFAQ - AI-Powered Customer Support")
        master.geometry("900x700")
        master.configure(bg="#f8f9fa")
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.primary_color = "#4e73df"
        self.secondary_color = "#858796"
        self.accent_color = "#36b9cc"
        self.light_bg = "#f8f9fa"
        self.dark_text = "#5a5c69"
        
        # Create header
        self.header_frame = tk.Frame(master, bg=self.primary_color, height=80)
        self.header_frame.pack(fill=tk.X)
        self.header_frame.pack_propagate(False)
        
        self.title_label = tk.Label(self.header_frame, text="IntelliFAQ", font=("Arial", 24, "bold"), fg="white", bg=self.primary_color)
        self.title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        self.subtitle_label = tk.Label(self.header_frame, text="AI-Powered Customer Support", font=("Arial", 12), fg="white", bg=self.primary_color)
        self.subtitle_label.pack(side=tk.LEFT, padx=0, pady=30)
        
        # Create main content frame
        self.main_frame = tk.Frame(master, bg=self.light_bg)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create left sidebar with categories
        self.sidebar_frame = tk.Frame(self.main_frame, bg="white", width=200, relief=tk.RAISED, bd=1)
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.sidebar_frame.pack_propagate(False)
        
        sidebar_title = tk.Label(self.sidebar_frame, text="FAQ Categories", font=("Arial", 12, "bold"), bg="white", fg=self.dark_text, pady=10)
        sidebar_title.pack(fill=tk.X)
        
        ttk.Separator(self.sidebar_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)
        
        # Add category buttons
        for category in sorted(faq_categories):
            btn = tk.Button(self.sidebar_frame, text=category, font=("Arial", 10), bg="white", fg=self.dark_text, relief=tk.FLAT, anchor=tk.W, command=lambda c=category: self.filter_by_category(c))
            btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Create chat area
        self.chat_frame = tk.Frame(self.main_frame, bg="white", relief=tk.RAISED, bd=1)
        self.chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create chat display
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, width=60, height=25,
                                                    font=("Arial", 10), bg="white", fg=self.dark_text,
                                                    relief=tk.FLAT, bd=0)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.chat_display.configure(state='disabled')
        
        # Configure tags for different message types
        self.chat_display.tag_config('user_msg', foreground=self.primary_color, font=("Arial", 10, "bold"))
        self.chat_display.tag_config('bot_msg', foreground=self.dark_text, font=("Arial", 10))
        self.chat_display.tag_config('question_tag', foreground=self.primary_color, font=("Arial", 10, "bold"))
        
        # Create input area
        self.input_frame = tk.Frame(self.chat_frame, bg="white")
        self.input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.user_input = tk.Entry(self.input_frame, font=("Arial", 12), relief=tk.FLAT, bd=1, bg="#f1f3f4", fg=self.dark_text)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", self.send_message)
        
        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message, bg=self.primary_color, fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT, width=8, height=1)
        self.send_button.pack(side=tk.RIGHT)
        
        # Create suggestion area
        self.suggestion_frame = tk.Frame(self.chat_frame, bg="white", height=50)
        self.suggestion_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.suggestion_frame.pack_propagate(False)
        
        suggestion_label = tk.Label(self.suggestion_frame, text="Quick questions:", font=("Arial", 10), bg="white", fg=self.secondary_color)
        suggestion_label.pack(side=tk.LEFT)
        
        self.suggestion_canvas = tk.Canvas(self.suggestion_frame, bg="white", height=30, highlightthickness=0)
        self.suggestion_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Add welcome message
        self.add_bot_message("Hello! I'm IntelliFAQ, your AI-powered customer support assistant. How can I help you today?")
        
        # Add some suggested questions
        self.add_suggestions()
        
    def add_suggestions(self):
        suggestions = [
            "What is your return policy?",
            "How do I track my order?",
            "What payment methods do you accept?",
            "How can I contact customer service?"
        ]
        
        x_offset = 0
        for suggestion in suggestions:
            btn_id = self.suggestion_canvas.create_text(x_offset + 5, 15, text=suggestion, anchor=tk.W,font=("Arial", 9), fill=self.primary_color)
            text_bbox = self.suggestion_canvas.bbox(btn_id)
            btn_bg = self.suggestion_canvas.create_rectangle(text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5, fill="#e8f0fe", outline="")
            self.suggestion_canvas.tag_lower(btn_bg, btn_id)
            
            # Make it clickable
            self.suggestion_canvas.tag_bind(btn_id, "<Button-1>", lambda e, s=suggestion: self.suggestion_clicked(s))
            self.suggestion_canvas.tag_bind(btn_bg, "<Button-1>", lambda e, s=suggestion: self.suggestion_clicked(s))
            x_offset = text_bbox[2] + 20
    
    def suggestion_clicked(self, suggestion):
        self.user_input.delete(0, tk.END)
        self.user_input.insert(0, suggestion)
        self.send_message()
    
    def filter_by_category(self, category):
        filtered_faqs = [faq for faq in faqs if faq["category"] == category]
        
        # Display filtered FAQs
        self.chat_display.configure(state='normal')
        self.chat_display.delete(1.0, tk.END)
        
        self.add_bot_message(f"Here are our FAQs about {category}:")
        for i, faq in enumerate(filtered_faqs, 1):
            self.chat_display.insert(tk.END, f"\n{i}. {faq['question']}\n", 'question_tag')
        
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)
    
    def add_user_message(self, message):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"You: {message}\n", 'user_msg')
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)
        
        # Add to conversation history
        self.conversation_history.append({
            "sender": "user",
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def add_bot_message(self, message):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"IntelliFAQ: {message}\n\n", 'bot_msg')
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)
        
        # Add to conversation history
        self.conversation_history.append({
            "sender": "bot",
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def send_message(self, event=None):
        user_message = self.user_input.get().strip()
        if user_message:
            self.add_user_message(user_message)
            self.user_input.delete(0, tk.END)
            
            # Show typing indicator in a separate thread
            threading.Thread(target=self.show_typing_indicator, args=(user_message,)).start()
    
    def show_typing_indicator(self, user_message):
        # Simulate typing delay
        time.sleep(1)
        
        # Get response
        response, score, category = get_response(user_message)
        
        # Update GUI in the main thread
        self.master.after(0, self.display_response, response, score, category)
    
    def display_response(self, response, score, category):
        # Add confidence indicator for high scores
        if score > 0.5:
            confidence = " (High confidence)"
        elif score > 0.3:
            confidence = " (Medium confidence)"
        else:
            confidence = " (Low confidence)"
        
        self.add_bot_message(response + confidence)
        
        # Log the interaction
        self.log_interaction(response, score, category)
    
    def log_interaction(self, response, score, category):
        # In a real application, this would save to a database or file
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_input": self.conversation_history[-2]["message"] if len(self.conversation_history) >= 2 else "Unknown",
            "bot_response": response,
            "confidence_score": float(score),
            "category": category,
            "success": score > 0.2
        }
        
        # Print to console for demonstration
        print(f"Interaction logged: {json.dumps(log_entry, indent=2)}")

# Main function
def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()