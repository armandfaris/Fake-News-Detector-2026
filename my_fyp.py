import streamlit as st
import numpy as np
import joblib
import re
import os
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import PyPDF2
import nltk
import warnings
from urllib.parse import urlparse
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
import joblib
from gensim.models import KeyedVectors

def get_hybrid_vector(text, tfidf_model, w2v_vectors):
    # This turns your news text into a 100-dimension "meaning" vector
    words = text.lower().split()
    doc_tfidf = tfidf_model.transform([text]).toarray().flatten()
    word_index = tfidf_model.vocabulary_
    
    vectors, weights = [], []
    for word in words:
        if word in w2v_vectors and word in word_index:
            idx = word_index[word]
            weight = doc_tfidf[idx]
            if weight > 0:
                # We multiply the meaning by the weight
                vectors.append(w2v_vectors[word] * weight)
                weights.append(weight)
                
    if not vectors:
        return np.zeros(100)
    return np.sum(vectors, axis=0) / np.sum(weights)

# --- 0. INITIALIZATION & WARNING FILTERS ---
nltk.download('stopwords', quiet=True)
# This stops the "InconsistentVersionWarning" from cluttering your app
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- 1. UTILITY FUNCTIONS ---
def verify_source(url=""):
    verified_list = {
        "thestar.com.my": "THE STAR", "nst.com.my": "NEW STRAITS TIMES",
        "malaysiakini.com": "MALAYSIAKINI", "bernama.com": "BERNAMA",
        "freemalaysiatoday.com": "FMT", "bharian.com.my": "BERITA HARIAN",
        "hmetro.com.my": "HARIAN METRO", "reuters.com": "REUTERS",
        "bbc.com": "BBC", "cnn.com": "CNN", "aljazeera.com": "AL JAZEERA"
    }
    if not url or url.strip() == "": return "No Link Provided", "Unverified"
    try:
        clean_url = url.strip().lower()
        if not clean_url.startswith(('http://', 'https://')): clean_url = 'https://' + clean_url
        domain = urlparse(clean_url).netloc.replace("www.", "")
        if domain in verified_list: return verified_list[domain], "Verified"
        return f"External ({domain})", "Unverified"
    except: return "Unknown Source", "Unverified"

def get_latest_news():
    return [
        {"title": "Malaysia Economic Outlook 2026", "source": "Bernama", "url": "https://www.bernama.com"},
        {"title": "Tech Advancements in Southeast Asia", "source": "The Star", "url": "https://www.thestar.com.my"},
        {"title": "New Cybersecurity Measures Announced", "source": "NST", "url": "https://www.nst.com.my"}
    ]

def extract_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.title.string if soup.title else "Untitled URL"
        
        # 1. THE AGGRESSIVE BLACKLIST
        # Add phrases that appear in menus, pop-ups, and browser warnings
        blacklist = [
            "follow us", "subscribe", "disclaimer", "privacy policy", 
            "best viewed on", "chrome browser", "keep you posted",
            "latest promotion", "fill the form", "enjoy this feature",
            "copyright", "all rights reserved", "terms of use",
            "wisma bernama", "portal kerjaya", "sign in", "create account"
        ]
        
        # 2. Extract paragraphs
        paragraphs = soup.find_all('p')
        clean_paragraphs = []
        
        for p in paragraphs:
            text = p.get_text().strip()
            text_lower = text.lower()
            
            # 3. THE "TRIPLE FILTER"
            # Filter A: Must be longer than 40 characters (ignores "Share this", etc.)
            # Filter B: Must not contain any word from our blacklist
            # Filter C: Must not start with "By " (often ignores photo credits)
            if len(text) > 40:
                if not any(phrase in text_lower for phrase in blacklist):
                    if not text.startswith("By "):
                        clean_paragraphs.append(text)
        
        # 4. Join and return
        body_content = ' '.join(clean_paragraphs)
        
        # Final safety check: if we filtered TOO MUCH, return a warning
        if len(body_content) < 100:
            return title.strip(), "Warning: Content extraction might be blocked by a paywall or cookie consent."
            
        return title.strip(), body_content.strip()
    except Exception as e: 
        return None, f"Fetch Failed: {e}"

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        return "PDF Document", "".join([p.extract_text() or "" for p in pdf_reader.pages])
    except: return None, "Read Failed"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# --- 2. SETUP & ASSET LOADING ---
st.set_page_config(page_title="Fake News Detector", layout="wide")
base_path = os.path.dirname(__file__)

@st.cache_resource
def load_assets():
    # Load the 3 Hybrid components
    model = joblib.load('logistic_model.pkl')
    vectorizer = joblib.load('vectorizer_volume.pkl')
    # Use KeyedVectors for the .bin file
    w2v_vectors = KeyedVectors.load_word2vec_format('w2v_vectors.bin', binary=True)
    
    # Create the Explainer
    e = LimeTextExplainer(class_names=['Authentic', 'Deceptive'])
    
    return model, vectorizer, w2v_vectors, e

def hybrid_predict_proba(texts):
    vectors = []
    for t in texts:
        vectors.append(get_hybrid_vector(clean_text(t), vectorizer, w2v_vectors))
    return model.predict_proba(np.array(vectors))

try:
    # Use w2v_vectors instead of w2v_model here
    model, vectorizer, w2v_vectors, explainer = load_assets()
except Exception as e:
    st.error(f"Asset Error: {e}"); st.stop()
    

# --- 3. CSS CUSTOMIZATION ---
st.markdown("""
    <style>
    .stApp { background-color: #f1f3f6; } 
    div[data-testid='stVerticalBlock'] > div:has(div.stTextArea) { 
        background-color: white !important; padding: 30px; border-radius: 15px; 
        border: 1px solid #d1d5db; box-shadow: 0px 4px 20px rgba(0,0,0,0.08); 
    }
    .stTextInput label, .stTextArea label { font-weight: 800 !important; color: #1e293b !important; font-size: 1.1rem !important; }
    .stTextInput input, .stTextArea textarea { border: 2px solid #cbd5e1 !important; border-radius: 10px !important; }
    div.stButton > button { 
        background-color: #1e1e1e !important; color: white !important; 
        border-radius: 50px !important; font-weight: bold !important; 
        height: 3.5em !important; width: 100% !important; border: none !important;
    }
    div.stButton > button:hover { background-color: #56A65F !important; transform: translateY(-2px); }
    .main-title { color: #1e293b; font-size: 3rem !important; font-weight: 800; }
    .main-title2 { color: #1e293b; font-size: 2.2rem !important; font-weight: 800; }
    .highlight { color: #56A65F; }
    .news-card { background-color: white; padding: 15px; border-radius: 10px; border-left: 6px solid #56A65F; margin-bottom: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. STATE MANAGEMENT & ALERT LOGIC ---
if 'run_analysis' not in st.session_state: st.session_state.run_analysis = False
if 'final_title' not in st.session_state: st.session_state.final_title = ""
if 'final_body' not in st.session_state: st.session_state.final_body = ""
if 'final_link' not in st.session_state: st.session_state.final_link = ""
if 'reset_counter' not in st.session_state: st.session_state.reset_counter = 0
if 'show_toast' not in st.session_state: st.session_state.show_toast = False

# Trigger the alert if a fetch was just successful
if st.session_state.show_toast:
    st.toast("✅ Article content successfully fetched!", icon='📰')
    st.session_state.show_toast = False

# --- 5. UI LAYOUT ---

# We check the state FIRST before defining columns
if st.session_state.run_analysis == False:
    # --- PAGE 1: INPUT VIEW (With Left Panel) ---
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown('<p class="main-title">Fake News Detector<br><span class="highlight">Preserve the truth.</span></p>', unsafe_allow_html=True)
        st.divider()
        # The news items are now LOCKED to Page 1
        for item in get_latest_news():
            st.markdown(f'<div class="news-card"><small style="color:#56A65F;font-weight:bold;">{item["source"]}</small><br><a href="{item["url"]}" target="_blank" style="text-decoration:none;color:#1e293b;font-weight:700;">{item["title"]}</a></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="main-title2">Analysis Panel</p>', unsafe_allow_html=True)
        tabs = st.tabs(["Text", "URL", "PDF"])
        
        with tabs[0]:
            st.session_state.final_title = st.text_input("Article Title", value=st.session_state.final_title)
            st.session_state.final_body = st.text_area("Article Content", height=200, value=st.session_state.final_body)
            st.session_state.final_link = st.text_input("Source Link (Optional)", value=st.session_state.final_link)

        with tabs[1]:
            u_col, b_col = st.columns([5, 1.2])
            with u_col:
                u_in = st.text_input("Enter URL", key=f"u_{st.session_state.reset_counter}", label_visibility="collapsed")
            with b_col:
                if st.button("Fetch", key="fetch_btn"):
                    if u_in:
                        with st.spinner(""):
                            t, b = extract_text_from_url(u_in)
                            if t and b:
                                st.session_state.final_title, st.session_state.final_body, st.session_state.final_link = t, b, u_in
                                st.session_state.show_toast = True # Queue the alert
                                st.rerun()

        with tabs[2]:
            st.markdown("### **PDF Extraction**")
            f = st.file_uploader("Upload PDF", type="pdf", key=f"p_{st.session_state.reset_counter}")
            if f and st.button("Extract Content"):
                t, b = extract_text_from_pdf(f)
                if t and b:
                    st.session_state.final_title, st.session_state.final_body = t, b
                    st.rerun()

        can_analyze = len(st.session_state.final_title.strip()) >= 3 and len(st.session_state.final_body.strip()) >= 10
        
        b_col1, b_col2 = st.columns([1, 1]) 
        with b_col1:
            if st.button("Check Authenticity", disabled=not can_analyze, use_container_width=True):
                st.session_state.run_analysis = True
                st.rerun()
        with b_col2:
            if st.button("Clear All", use_container_width=True):
                st.session_state.final_title = ""; st.session_state.final_body = ""; st.session_state.final_link = ""
                st.session_state.reset_counter += 1
                st.rerun()

else:
    # --- PAGE 2: FORENSIC REPORT (FULL WIDTH BLANK PAGE) ---
    # We use a single container here to make it look like a new page
    if st.button("⬅ Back to Analysis"):
        st.session_state.run_analysis = False
        st.rerun()

    with st.spinner("Analyzing..."):
        full = f"TITLE: {st.session_state.final_title} | BODY: {st.session_state.final_body}"
        cleaned = clean_text(full)
        # 1. Convert text to 100-dim hybrid vector
        h_vec = get_hybrid_vector(cleaned, vectorizer, w2v_vectors)
        # 2. Predict using the 100-dim vector
        prob = model.predict_proba(h_vec.reshape(1, -1))[0][1]
        src, status = verify_source(st.session_state.final_link)
        # Use the hybrid wrapper function instead of c_pipeline
        exp = explainer.explain_instance(cleaned, hybrid_predict_proba, num_features=10)

    st.markdown('<p class="main-title2">Forensic Report</p>', unsafe_allow_html=True)
    
    # We use a 3-column layout to center the report or keep it wide
    r1, r2 = st.columns(2)
    with r1:
        if prob >= 0.5: st.error(f"**Deceptive Pattern** ({prob*100:.1f}%)")
        else: st.success(f"**Authentic Content** ({(1-prob)*100:.1f}%)")
    with r2:
        if status == "Verified": st.info(f"**Verified:** {src}")
        else: st.warning(f"**External:** {src}")

    st.divider()
    mode = st.radio("Analysis View:", ["Heatmap", "Chart"], horizontal=True)
    if "Heatmap" in mode:
        components.html(exp.as_html(), height=800, scrolling=True)
    else:
        df_l = pd.DataFrame(exp.as_list(), columns=['Phrase', 'Weight'])
        df_l['Class'] = df_l['Weight'].apply(lambda x: 'Deceptive' if x > 0 else 'Authentic')
        fig = px.bar(df_l, x='Weight', y='Phrase', orientation='h', color='Class', 
                     color_discrete_map={'Deceptive': '#ef5350', 'Authentic': '#66bb6a'}, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)