import streamlit as st
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load data
# -------------------------------
ratings_df = pd.read_csv("data/sample_data.csv")
item_df = pd.read_csv("data/item_data.csv")

# Load collaborative filtering model
svd_model = pickle.load(open("models/svd_model.pkl", "rb"))

# -------------------------------
# Build similarity matrix (Content-Based)
# -------------------------------
@st.cache_resource
def build_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["title"])
    return cosine_similarity(tfidf_matrix)

cosine_sim = build_similarity_matrix(item_df)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📚 Book Recommendation System")

mode = st.radio(
    "Choose Recommendation Type",
    ("Based on a book you like", "Based on a user")
)

# -------------------------------
# Content-Based Recommendation
# -------------------------------
if mode == "Based on a book you like":
    book = st.selectbox("Select a Book", item_df["title"].unique())

    if st.button("Recommend"):
        idx = item_df[item_df["title"] == book].index[0]

        scores = list(enumerate(cosine_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

        recommended_titles = (
            item_df.iloc[[i[0] for i in scores]]["title"]
            .drop_duplicates()
            .tolist()
        )

        st.subheader("Recommended Books")
        for title in recommended_titles:
            st.write("•", title)

# -------------------------------
# Collaborative Filtering
# -------------------------------
elif mode == "Based on a user":
    user = st.selectbox("Select User ID", ratings_df["user_id"].unique())

    if st.button("Recommend"):
        seen_items = set(
            ratings_df[ratings_df["user_id"] == user]["item_id"]
        )
        all_items = set(ratings_df["item_id"].unique())
        candidates = all_items - seen_items

        predictions = [
            (iid, svd_model.predict(user, iid).est)
            for iid in candidates
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)

        top_items = [i[0] for i in predictions[:5]]

        recommended_titles = (
            ratings_df[ratings_df["item_id"].isin(top_items)]["title"]
            .drop_duplicates()
            .tolist()
        )

        st.subheader("Recommended Books")
        for title in recommended_titles:
            st.write("•", title)


