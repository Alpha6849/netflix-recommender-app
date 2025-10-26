import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import requests
import urllib.parse

# -----------------------------
# 1️⃣ Load Data and Models
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/netflix_titles.csv")
    df['country'] = df['country'].fillna('Unknown')
    df['type'] = df['type'].fillna('Movie')
    df['listed_in'] = df['listed_in'].fillna('Unknown')
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(0).astype(int)
    df['decade'] = (df['release_year'] // 10 * 10).astype(int)
    df['main_country'] = df['country'].apply(lambda x: x.split(',')[0].strip())
    df['main_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0].strip())
    return df

@st.cache_data
def load_models():
    rf_model = joblib.load("model/randomforest_netflix_model.pkl")
    tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return rf_model, tfidf_vectorizer

df = load_data()
rf_model, tfidf_vectorizer = load_models()

# TF-IDF matrix caching
@st.cache_data
def compute_tfidf_matrix(descriptions):
    return tfidf_vectorizer.transform(descriptions.fillna(""))

tfidf_matrix = compute_tfidf_matrix(df['description'])

# Encode categorical columns
le_rating = LabelEncoder()
df['rating_encoded'] = le_rating.fit_transform(df['rating'].astype(str))

le_country = LabelEncoder()
df['country_encoded'] = le_country.fit_transform(df['main_country'])

le_type = LabelEncoder()
df['type_encoded'] = le_type.fit_transform(df['type'].astype(str))

le_genre = LabelEncoder()
df['genre_encoded'] = le_genre.fit_transform(df['main_genre'])

# -----------------------------
# 2️⃣ Streamlit Layout
# -----------------------------
st.set_page_config(page_title="Netflix Analysis & Recommender", layout="wide")
st.title("🎬 Netflix Analysis & Recommender")

menu = ["Home", "Predict Rating", "Recommend Shows"]
choice = st.sidebar.selectbox("Menu", menu)

# -----------------------------
# 3️⃣ Home Page
# -----------------------------
if choice == "Home":
    st.header("📊 Dataset Overview")
    st.dataframe(df.head(10))
    st.write("**Total titles:**", df.shape[0])
    st.write("**Movies:**", df[df['type'] == "Movie"].shape[0])
    st.write("**TV Shows:**", df[df['type'] == "TV Show"].shape[0])
    st.bar_chart(df['rating_encoded'].value_counts())

# -----------------------------
# 4️⃣ Predict Rating
# -----------------------------
elif choice == "Predict Rating":
    st.header("🤖 Predict Netflix Rating")

    try:
        is_movie = st.selectbox("Is it a Movie?", ["Yes", "No"])
        country = st.selectbox("Country", sorted(df['main_country'].unique()))
        decade = st.selectbox("Decade", sorted(df['decade'].unique()))
        genre = st.selectbox("Genre", sorted(df['main_genre'].unique()))

        if is_movie == "Yes":
            duration = st.number_input(
                "Duration (in minutes)", min_value=1, max_value=500, value=90
            )
        else:
            seasons = st.number_input("Number of Seasons", min_value=1, max_value=20, value=1)
            mins_per_ep = st.number_input("Minutes per Episode", min_value=10, max_value=180, value=30)
            duration = seasons * mins_per_ep

        type_encoded = le_type.transform(["Movie" if is_movie=="Yes" else "TV Show"])[0]
        country_encoded = le_country.transform([country])[0]
        genre_encoded = le_genre.transform([genre])[0]

        X_input = [[type_encoded, country_encoded, duration, decade, genre_encoded]]

        if st.button("Predict"):
            pred_code = rf_model.predict(X_input)[0]
            pred_label = le_rating.inverse_transform([int(pred_code)])[0]
            desc = f"{duration} min Movie" if is_movie=="Yes" else f"{seasons} Seasons (~{mins_per_ep} min/ep)"
            st.success(f"🎯 Predicted Netflix rating for **{desc}, {country}, {decade}s, Genre: {genre}** is: **{pred_label}**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -----------------------------
# 5️⃣ Recommend Shows (with Posters)
# -----------------------------
elif choice == "Recommend Shows":
    st.header("🎯 Recommend Similar Shows")

    show_title = st.selectbox("Choose a show:", df['title'].dropna().tolist())
    top_n = st.slider("Number of Recommendations", 1, 10, 5)

    # 🔑 Unhidden OMDb key (for local testing)
    YOUR_OMDB_KEY = "c247010b"  

    # Poster fetching with caching
    @st.cache_data
    def fetch_poster_cached(title, api_key):
        try:
            title_encoded = urllib.parse.quote(title)
            url = f"http://www.omdbapi.com/?t={title_encoded}&apikey={api_key}"
            data = requests.get(url).json()
            poster = data.get("Poster")
            if poster and poster != "N/A":
                return poster
            else:
                return "https://via.placeholder.com/200x300?text=No+Image"
        except:
            return "https://via.placeholder.com/200x300?text=Error"

    # Recommendation function
    def recommend_show_advanced(title, df, tfidf_matrix, top_n=5):
        idx = df[df['title']==title].index[0]
        show_type = df.loc[idx, 'type']
        show_genres = set(str(df.loc[idx,'listed_in']).split(', '))
        show_country = df.loc[idx,'main_country']
        show_decade = df.loc[idx,'decade']

        # Filter shows
        df_filtered = df[
            (df['type']==show_type) &
            (df['main_country']==show_country) &
            (df['decade']==show_decade) &
            (df['listed_in'].apply(lambda x: len(show_genres.intersection(set(str(x).split(', '))))>0))
        ]
        filtered_indices = df_filtered.index.tolist()

        # Compute similarity only for filtered shows
        filtered_tfidf = tfidf_matrix[filtered_indices]
        sim_scores = cosine_similarity(tfidf_matrix[idx], filtered_tfidf)[0]
        top_indices = [filtered_indices[i] for i in sim_scores.argsort()[::-1] if filtered_indices[i] != idx][:top_n]

        return df.iloc[top_indices]['title'].tolist()

    if st.button("Get Recommendations"):
        recs = recommend_show_advanced(show_title, df, tfidf_matrix, top_n)
        st.success("Top Recommendations with Posters:")

        n_cols = min(3, top_n)
        cols = st.columns(n_cols)
        for i, rec in enumerate(recs):
            col = cols[i % n_cols]
            with col:
                poster_url = fetch_poster_cached(rec, YOUR_OMDB_KEY)
                st.image(poster_url, use_container_width=True)
                st.caption(rec)
