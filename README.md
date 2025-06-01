# Buyers-Voice-Review-Sentiment-Analyzer
## 🔗 Live Demo

Click the badge below to open the deployed Streamlit app:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://buyer-s-voice-a-review-sentiment-analyzer.streamlit.app/)

## 🚰 End-to-End NLP Project – Buyer’s Voice: A Review Sentiment Analyzer

**Buyer’s Voice** is a full-stack, NLP-powered sentiment analysis system designed to analyze customer product reviews and classify them into positive or negative sentiments. The project is deployed as an interactive **Streamlit web application** and is capable of ingesting raw text input, performing deep preprocessing, and providing real-time predictions with explanations. This project is ideal for showcasing **NLP, ML deployment, and data pipeline skills** on your resume or portfolio.

---

### 📦 Project Overview

This application is built around real-world business needs in e-commerce and customer analytics. It simulates how platforms like Amazon or Flipkart analyze millions of product reviews to extract customer sentiment and improve product/service decisions.

The system uses a combination of:

* Custom text preprocessing pipeline
* Pre-trained Word2Vec embeddings
* Scikit-learn-based ML model (e.g., XGBoost, Logistic Regression)
* Streamlit for real-time interactivity

---

### 🧱 Core Components

| Component                  | Description                                                                                                                                         |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Preprocessing Pipeline** | Includes URL removal, punctuation cleanup, contractions expansion, emoji-to-text translation, stopword removal, lemmatization, and slang correction |
| **Word Embeddings**        | Word2Vec model trained on a large corpus of reviews                                                                                                 |
| **Model**                  | Trained sentiment classification model using `scikit-learn`                                                                                         |
| **UI**                     | Streamlit web interface for inputting reviews and seeing predictions                                                                                |
| **Deployment**             | Hosted on Streamlit Cloud for public access                                                                                                         |

---

### 🎯 Business Use Case

Businesses often struggle to understand how buyers feel about their products at scale. This app solves that by enabling:

* **Instant sentiment detection** from product reviews
* **Clean and interpretable predictions** through NLP
* **Scalable pipeline** for real-time deployment or batch processing

---

### 🔍 Key Features

* ✅ Clean, modular NLP pipeline with slang, emojis, and stopwords handled
* ✅ Word2Vec embedding generation for semantic understanding
* ✅ Scikit-learn classification model for sentiment prediction
* ✅ User-friendly Streamlit interface
* ✅ Deployable end-to-end on the cloud
* ✅ Real-world error handling and logging included

---

### 🧪 Tech Stack Used

* **Python**
* **NLTK**, **TextBlob**, **Contractions**, **emoji**
* **Word2Vec (Gensim)** for embedding generation
* **Scikit-learn** and **XGBoost** for modeling
* **Streamlit** for UI and deployment
* **Matplotlib / Seaborn** for EDA (if added)
