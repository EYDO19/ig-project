import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Recommender AI",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {background-color:#11120D;color:#FFFBF4;}
section[data-testid="stSidebar"] {background-color:#565449;}
h1,h2,h3 {color:#FFFBF4;}
.stButton>button {background-color:#D8CFBC;color:#11120D;border-radius:14px;border:none;padding:0.6rem 1.2rem;font-weight:bold;}
.movie-card {background-color:#565449;padding:15px;border-radius:14px;margin-bottom:10px;}
.movie-title {font-size:18px;font-weight:bold;}
.movie-id {color:#D8CFBC;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA (CACHED) ----------------
@st.cache_data
def load_data():

    BASE_DIR = os.path.dirname(__file__)

    movies = pd.read_csv(os.path.join(BASE_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(BASE_DIR, "ratings.csv"))

    movies["title"] = movies["title"].str.lower()
    ratings = ratings.dropna()

    return movies, ratings


movies, ratings = load_data()

# ---------------- CONTENT MODEL (CACHED) ----------------
@st.cache_resource
def build_content_model(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"].fillna(""))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    title_to_index = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

    return cosine_sim, title_to_index

cosine_sim, title_to_index = build_content_model(movies)

def content_recommend(title, top_n=10):
    title = title.lower()

    if title not in title_to_index:
        return movies.sample(top_n)[["movieId", "title"]].to_dict("records")

    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    idxs = [i[0] for i in sim_scores]
    return movies.iloc[idxs][["movieId", "title"]].to_dict("records")

# ---------------- COLLAB MODEL (CACHED) ----------------
@st.cache_resource
def build_collab_model(ratings):
    user_item = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    svd = TruncatedSVD(n_components=50, random_state=42)
    mat = svd.fit_transform(user_item)
    recon = np.dot(mat, svd.components_)

    user_map = {u: i for i, u in enumerate(user_item.index)}
    movie_ids = list(user_item.columns)

    return recon, user_map, movie_ids, user_item

recon, user_map, movie_ids, user_item = build_collab_model(ratings)

def collab_recommend(user_id, top_n=10):
    if user_id not in user_map:
        return movies.sample(top_n)[["movieId", "title"]].to_dict("records")

    uidx = user_map[user_id]
    scores = recon[uidx]

    top = np.argsort(scores)[::-1][:top_n]
    ids = [movie_ids[i] for i in top]

    return movies[movies["movieId"].isin(ids)][["movieId", "title"]].to_dict("records")

# ---------------- HYBRID ----------------
def hybrid_recommend(user_id, title, top_n=10):
    c = content_recommend(title, top_n * 2)
    cf = collab_recommend(user_id, top_n * 2)

    scores = {}

    for i, m in enumerate(c):
        scores[m["movieId"]] = scores.get(m["movieId"], 0) + 1 / (i + 1)

    for i, m in enumerate(cf):
        scores[m["movieId"]] = scores.get(m["movieId"], 0) + 1 / (i + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ids = [i[0] for i in ranked[:top_n]]

    return movies[movies["movieId"].isin(ids)][["movieId", "title"]].to_dict("records")

# ---------------- UI ----------------
st.title("🎬 Movie Recommender AI")

col1, col2 = st.columns(2)

with col1:
    selected_movie = st.selectbox(
        "Choose a movie",
        sorted(movies["title"].unique())
    )

with col2:
    user_id = st.number_input("User ID", min_value=1, value=1)

method = st.radio("Method", ["Content-Based", "Collaborative Filtering", "Hybrid"])

# ---------------- BUTTON ----------------
if st.button("Recommend"):

    if method == "Content-Based":
        recs = content_recommend(selected_movie)
    elif method == "Collaborative Filtering":
        recs = collab_recommend(user_id)
    else:
        recs = hybrid_recommend(user_id, selected_movie)

    st.subheader("Results")

    for r in recs:
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">🎥 {r['title']}</div>
            <div class="movie-id">ID: {r['movieId']}</div>
        </div>
        """, unsafe_allow_html=True)
