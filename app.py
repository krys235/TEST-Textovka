import ast
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from itertools import chain


# ---------- LOAD MODEL & DATA ----------

@st.cache_resource
def load_model():
    model_name = "intfloat/multilingual-e5-base"
    return SentenceTransformer(model_name)

@st.cache_data
def load_data():
    df = pd.read_csv("merged_with_embeddings_metacsv")

    def parse_list(col):
        return col.apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
        )

    df["genres_list"] = parse_list(df["genres_list"])
    df["categories_list"] = parse_list(df["categories_list"])
    return df

@st.cache_data
def load_embeddings():
    return np.load("game_embeddings.npy")


model = load_model()
df = load_data()
embeddings = load_embeddings()


# ---------- GENRES & CATEGORIES ----------

all_genres = sorted(set(chain.from_iterable(df["genres_list"])))
all_categories = sorted(set(chain.from_iterable(df["categories_list"])))

ui_categories = [
    "single-player", "multi-player", "co-op", "online co-op",
    "pvp", "online pvp", "mmo",
    "full controller support", "partial controller support",
    "vr only", "vr support", "vr supported",
    "steam achievements", "steam cloud", "steam workshop",
]
visible_categories = [c for c in all_categories if c in ui_categories]


# ---------- FILTER + SCORING ----------

import pandas as pd

def build_filter_mask(df, selected_genres=None, selected_categories=None):
    mask = pd.Series(True, index=df.index)

    if selected_genres:
        mask &= df["genres_list"].apply(
            lambda g: all(genre in g for genre in selected_genres)
        )

    if selected_categories:
        mask &= df["categories_list"].apply(
            lambda c: all(cat in c for cat in selected_categories)
        )

    return mask


def boost_title_match(df_results, query, boost=0.03):
    q = query.lower().strip()
    df_results = df_results.copy()
    title_match = df_results["name"].str.lower().str.contains(q).astype(float)
    df_results["final_score"] = df_results["similarity"] + title_match * boost
    return df_results.sort_values("final_score", ascending=False)


def recommend_games(query, selected_genres=None, selected_categories=None, top_k=10):
    mask = build_filter_mask(df, selected_genres, selected_categories)
    df_sub = df[mask]
    emb_sub = embeddings[mask.values]

    if df_sub.empty:
        return df_sub

    query_text = "query: " + query
    q_vec = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    sims = emb_sub @ q_vec

    df_sub = df_sub.copy()
    df_sub["similarity"] = sims

    df_sub = boost_title_match(df_sub, query)
    return df_sub.head(top_k)


# ---------- STREAMLIT UI ----------

st.set_page_config(page_title="Steam Game Recommender", layout="wide")

st.title("🎮 Steam Game Recommender")

st.markdown(
    "Zadej, jakou hru nebo jaký typ zážitku hledáš, "
    "vyber žánry / kategorie a my ti doporučíme hry."
)

col_query, col_filters = st.columns([2, 1])

with col_query:
    user_query = st.text_input("User query (např. 'Skyrim-like open world RPG', 'cozy farming without combat')")

with col_filters:
    selected_genres = st.multiselect("Genres", all_genres)
    selected_categories = st.multiselect("Categories", visible_categories)

top_k = st.slider("Počet doporučených her", min_value=5, max_value=30, value=10, step=5)

if st.button("Recommend") and user_query.strip():
    results = recommend_games(
        query=user_query,
        selected_genres=selected_genres,
        selected_categories=selected_categories,
        top_k=top_k,
    )

    if results.empty:
        st.warning("Pro tyto filtry jsme nenašli žádné hry 😢 Zkus ubrat omezení.")
    else:
        for _, row in results.iterrows():
            with st.container():
                st.subheader(f"{row['name']}  —  (score: {row['final_score']:.3f})")
                st.write(f"**Genres:** {', '.join(row['genres_list'])}")
                st.write(f"**Categories:** {', '.join(row['categories_list'])}")
                st.write(f"**Metacritic:** {row['metacritic_score']}")
                st.write(row["short_description"])
                st.markdown("---")
