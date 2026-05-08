import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🎬 Movie Recommender System",
    page_icon="🍿",
    layout="wide"
)

# =========================
# TITLE UI
# =========================
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🎬 Hybrid Movie Recommendation System</h1>
    <h4 style='text-align: center; color: gray;'>Content-Based + Collaborative Filtering (SVD)</h4>
    <hr>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    movies_path = os.path.join(BASE_DIR, "movies.csv")
    ratings_path = os.path.join(BASE_DIR, "ratings.csv")

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    return movies, ratings

movies, ratings = load_data()

# =========================
# CONTENT MODEL
# =========================
@st.cache_resource
def build_content_model(movies):
    movies["genres"] = movies["genres"].fillna("").astype(str)

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    title_to_index = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

    return cosine_sim, title_to_index

cosine_sim, title_to_index = build_content_model(movies)

# =========================
# COLLAB MODEL
# =========================
@st.cache_resource
def build_collab_model(ratings):
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    user_item = train.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    svd = TruncatedSVD(n_components=50, random_state=42)
    matrix = svd.fit_transform(user_item)
    reconstructed = np.dot(matrix, svd.components_)

    user_map = {u: i for i, u in enumerate(user_item.index)}

    return user_item, reconstructed, user_map

user_item, reconstructed, user_map = build_collab_model(ratings)

# =========================
# FUNCTIONS
# =========================

def content_recommend(title, top_n=10):
    if title not in title_to_index:
        return []

    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_idx = [i[0] for i in sim_scores]

    return movies.iloc[movie_idx]


def collab_recommend(user_id, top_n=10):
    if user_id not in user_map:
        return pd.DataFrame()

    idx = user_map[user_id]
    scores = reconstructed[idx]

    top = np.argsort(scores)[::-1][:top_n]
    movie_ids = [user_item.columns[i] for i in top]

    return movies[movies["movieId"].isin(movie_ids)]


def hybrid_recommend(user_id, title, top_n=10):
    c = content_recommend(title, top_n * 2)
    col = collab_recommend(user_id, top_n * 2)

    scores = {}

    for i, m in enumerate(c["movieId"]):
        scores[m] = scores.get(m, 0) + 1 / (i + 1)

    for i, m in enumerate(col["movieId"]):
        scores[m] = scores.get(m, 0) + 1 / (i + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = [m[0] for m in ranked[:top_n]]

    return movies[movies["movieId"].isin(top_ids)]

# =========================
# SIDEBAR UI
# =========================
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ["🎯 Content-Based", "👥 Collaborative", "🔥 Hybrid"]
)

user_id = st.sidebar.number_input("User ID", min_value=1, value=1)
movie_title = st.sidebar.selectbox("Select Movie", movies["title"].unique())

# =========================
# MAIN UI
# =========================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎬 Input")
    st.write("User ID:", user_id)
    st.write("Movie:", movie_title)

    run = st.button("🚀 Get Recommendations")

with col2:
    st.subheader("🍿 Recommendations")

    if run:
        if mode == "🎯 Content-Based":
            result = content_recommend(movie_title)
        elif mode == "👥 Collaborative":
            result = collab_recommend(user_id)
        else:
            result = hybrid_recommend(user_id, movie_title)

        if len(result) == 0:
            st.warning("No recommendations found!")
        else:
            for i, row in result.iterrows():
                st.markdown(
                    f"""
                    <div style='padding:10px; margin:5px; border-radius:10px; background:#111; color:white'>
                        🎬 <b>{row['title']}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =========================
# FOOTER
# =========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
