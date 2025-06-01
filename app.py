
import streamlit as st

#importing all the necessary libraries
import pandas as pd
from PIL import Image
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import nltk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
import string
from textblob import TextBlob
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
import gensim
from gensim.utils import simple_preprocess
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import emoji
import contractions
from gensim.models import Word2Vec

import pickle
import numpy as np
import streamlit as st

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

st.markdown("""
<style>
@keyframes float {
  0%, 100% {
    transform: translateY(0px);
  }
  50% {
    transform: translateY(-8px);
  }
}
.floating-text {
  animation: float 3s ease-in-out infinite;
  display: inline-block;
}
</style>

<h2 class="floating-text" style="
    background: linear-gradient(90deg, #11998e, #38ef7d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
    font-size: 2.4em;
    text-align: center;
    padding: 15px 0;
    text-shadow: 0 0 5px rgba(0,0,0,0.2);
">
    Buyer's Voice: A Review Sentiment Analyzer
</h2>
""", unsafe_allow_html=True)
user_input = st.text_input(
    "📝 How do you feel about your purchase? Let's find out!",
    help="Type your honest review or opinion about the product here."
)
with open("CustomerSentiment/catboost1_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)


vector_model = Word2Vec.load("CustomerSentiment/final_model")


import cupy as cp
vocab_set = set(vector_model.wv.index_to_key)

def document_vector(doc):
    words = [word for word in doc.split() if word in vocab_set]
    if words:
        return cp.asnumpy(cp.mean(vector_model.wv[words], axis=0))
    else:
        return np.zeros(vector_model.vector_size)



stop_words=set(stopwords.words('english'))
lemma=WordNetLemmatizer()
negation_words = {"no", "nor", "not", "ain", "aren't", "couldn't", "didn't", "doesn't","hadn't", "hasn't", "haven't", "isn't", "shouldn't", "wasn't","weren't", "won't", "wouldn't","mightn't","needn't"}
stop_words = stop_words-negation_words
correction_dict = {'bday': 'birthday', 'gr8': 'great', 'luv': 'love', 'ur': 'your', 'pls': 'please', 'thx': 'thanks', 'u': 'you', 'brb': 'be right back', 'idk': 'I don\'t know', 'omg': 'oh my god', 'lol': 'laugh out loud', 'tbh': 'to be honest', 'fyi': 'for your information', 'lmk': 'let me know', 'btw': 'by the way', 'asap': 'as soon as possible', 'smh': 'shaking my head', 'ttyl': 'talk to you later', 'ppl': 'people', 'nvm': 'never mind', 'cya': 'see you', 'rofl': 'rolling on the floor laughing', 'omw': 'on my way', 'wdym': 'what do you mean', 'fomo': 'fear of missing out', 'yolo': 'you only live once', 'lmao': 'laughing my ass off', 'gtg': 'got to go', 'wbu': 'what about you', 'bbl': 'be back later', 'bff': 'best friends forever', 'gm': 'good morning', 'gn': 'good night', 'np': 'no problem', 'gg': 'good game', 'afk': 'away from keyboard', 'yup': 'yes', 'nah': 'no', 'yass': 'yes', 'plz': 'please', 'thru': 'through', 'gr8t': 'great', 'wat': 'what', 'wht': 'what', 'howdy': 'hello', 'g2g': 'got to go', 'l8r': 'later', 'no1': 'no one', 'cuz': 'because', 'bro': 'brother', 'sis': 'sister', 'imho': 'in my humble opinion', 'ftw': 'for the win', 'tmi': 'too much information', 'jmho': 'just my humble opinion', 'tbh': 'to be honest', 'btw': 'by the way', 'jk': 'just kidding', 'afaik': 'as far as I know', 'ik': 'I know', 'wfh': 'work from home', 'lmk': 'let me know', 'swag': 'style, confidence', 'fam': 'family', 'thnx': 'thanks', 'gr8ful': 'grateful', 'wyd': 'what you doing', 'sd': 'social distancing', 'pplz': 'people', 'seeya': 'see you', 'yay': 'yes', 'hbu': 'how about you', 'tho': 'though', 'm8': 'mate', 'gr8ful': 'grateful', 'gimme': 'give me', 'fml': 'f**k my life', 'qik': 'quick', 'realy': 'really', 'yr': 'your', 'wtf': 'what the f**k', 'bffl': 'best friends for life', '2morrow': 'tomorrow', '2nite': 'tonight', 'wth': 'what the hell', 'stfu': 'shut the f**k up', 'ngl': 'not gonna lie', 'tbh': 'to be honest', 'smh': 'shaking my head', 'hbd': 'happy birthday', 'gg': 'good game', 'n00b': 'newbie', 'pmu': 'pissed me off', 'rotfl': 'rolling on the floor laughing', 'sol': 'shout out loud', 'omfg': 'oh my f**king god', 'srsly': 'seriously', 'dunno': 'don\'t know', 'bbl': 'be back later', 'lolz': 'laugh out loud', 'l8': 'late', 'fr': 'for real', 'plz': 'please', 'stoked': 'excited', 'lit': 'awesome', 'noob': 'newbie', 'h8': 'hate', 'xoxo': 'hugs and kisses', 'smh': 'shaking my head', 'yolo': 'you only live once','plz':'please','gn':'good night'}


#nlp=spacy.load("en_core_web_lg",disable=['ner','parse'])
tqdm.pandas()
def remove_HTML_tags(text):
    pattern = re.compile('<.*?>')  
    return re.sub(pattern, '', text)
def lowercasing(text):
    return text.lower()
def remove_URL(text):
    pattern = re.compile(r'https?://\S+|www\.\S+') 
    return re.sub(pattern, '', text)
def remove_punc(text):
    exclude = string.punctuation
    return text.translate(str.maketrans('', '', exclude))
def spell_correction(text):
    textblb = TextBlob(text)
    return textblb.correct().string
def remove_emoji(text):
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"  # Emoticons
                               u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                               u"\U0001F1E0-\U0001F1FF"  # Regional Indicator Symbols
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"  # Enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
def demojify(text):
    return emoji.demojize(text)
def expand_contractions(text):
    return contractions.fix(text)
def remove_stopwords(text):
    list_words = [word for word in word_tokenize(text) if word not in stop_words]
    return ' '.join(list_words)
def remove_extra_whitespaces(text):
    return re.sub(r'\s+', ' ', text).strip()
def lemmatization(text):
    l=[lemma.lemmatize(word) for word in word_tokenize(text)]
    return ' '.join(l)
def replace_slang_in_review(text):
    words = word_tokenize(text) 
    corrected_text = [correction_dict[word] if word in correction_dict else word for word in words]
    return ' '.join(corrected_text)  
tqdm.pandas()

if user_input:
    input_data = [user_input]
    #for preprocessing
    
    input_data= [lowercasing(text) for text in input_data]
    input_data= [remove_extra_whitespaces(text) for text in input_data]
    input_data= [remove_HTML_tags(text) for text in input_data]
    input_data= [remove_URL(text) for text in input_data]
    input_data= [remove_punc(text) for text in input_data]
    input_data= [expand_contractions(text) for text in input_data]
    input_data= [replace_slang_in_review(text) for text in input_data]
    input_data= [demojify(text) for text in input_data]
    input_data= [remove_stopwords(text) for text in input_data]
    input_data= [lemmatization(text) for text in input_data]
    X_input = []
    
    # Iterate over each sentence pair (heading, body)
    for body in input_data:
        body_vector = document_vector(body)  # Convert body to vector
        X_input.append(body_vector)
    
    # Convert the list to a numpy array
    X_input = np.array(X_input)
    
    # Make predictions using the trained model
    predictions = loaded_model.predict(X_input)
    
    if predictions == 0:
        st.markdown("### 😠 Negative Sentiment")
        st.error("The sentiment of the text is **Negative**.")
        
    else:
        st.markdown("### 😊 Positive Sentiment")
        st.success("The sentiment of the text is **Positive**.")
        

st.markdown("""
<div style="background-color:#fff3cd; padding:15px; border-left:6px solid #ffa500; border-radius:5px; font-size:14px;">
    <strong>⚠️ <span style="color:red;">Caution:</span></strong><br>
    ⪼ This system provides predictions based solely on product purchase data. It does <em>not</em> interpret human <strong><em>emotions</em></strong>, <strong><em>intentions</em></strong>, or <strong><em>feelings</em></strong>.<br>
    ⪼ Predictions are <strong><span style="color:red;">strictly</span></strong> related to buyer-product interactions.
</div>
""", unsafe_allow_html=True)
    # If you want to map the predictions to actual labels (if you used a classifier):
    # predicted_labels = np.argmax(predictions)
    
    # st.markdown(predicted_labels)
    # st.markdown(predictions)
