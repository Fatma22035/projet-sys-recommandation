import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_mic_recorder import speech_to_text
from sentence_transformers import SentenceTransformer
from recommender import recommend, search_by_text_query

st.set_page_config(
    page_title="Restaurants similaires — Nouakchott",
    page_icon="🍽️",
    layout="wide"
)


@st.cache_data
def load_data():
    df_clean = pd.read_csv('restaurants_clean.csv')
    df_text = pd.read_csv('restaurants_with_text.csv')
    embed_sim = np.load('embed_similarity_matrix.npy')
    embeddings = np.load('embeddings.npy')
    return df_clean, df_text, embed_sim, embeddings


df_clean, df_text, embed_sim, embeddings = load_data()


@st.cache_resource
def load_embed_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


embed_model = load_embed_model()

# ============================================================
# SIDEBAR — NAVIGATION
# ============================================================
st.sidebar.title("🍽️ Navigation")
page = st.sidebar.radio(
    "Choisir une page",
    ["Recommandation par restaurant", "Recherche par description", "Statistiques"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**À propos**\n\n"
    f"Système de recommandation de restaurants similaires à Nouakchott, "
    f"basé sur le contenu des avis clients et parfois mode fallback géographique.\n\n"
)

# ============================================================
# PAGE 1 — RECOMMANDATION PAR RESTAURANT
# ============================================================
if page == "Recommandation par restaurant":
    st.title("Trouve des restaurants similaires")
    st.markdown("Sélectionne un restaurant pour découvrir des établissements similaires.")

    df_clean['display_label'] = df_clean['title'] + " — " + df_clean['city_clean']

    selected_label = st.selectbox(
        "Choisis un restaurant",
        options=df_clean.sort_values('title')['display_label'].tolist()
    )
    selected_id = df_clean.loc[df_clean['display_label'] == selected_label, 'restaurant_id'].iloc[0]

    n_results = st.slider("Nombre de recommandations", 3, 10, 5)

    target_row = df_clean[df_clean['restaurant_id'] == selected_id].iloc[0]

    st.markdown("---")
    st.subheader(f"📍 {target_row['title']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Note", f"{target_row['rating_final']:.1f} ⭐" if target_row['has_rating'] else "Pas de note")
    col2.metric("Nb d'avis", int(target_row['reviewsCount']))
    col3.metric("Ville", target_row['city_clean'])

    if not target_row['has_text']:
        st.warning("⚠️ Ce restaurant n'a pas d'avis textuels. Recommandations basées sur la proximité géographique.")

    st.markdown("---")
    st.subheader(f"Top {n_results} restaurants similaires")

    results, method_used = recommend(selected_id, df_clean, df_text, embed_sim, n=n_results)

    if results.empty:
        st.info("Aucune recommandation trouvée.")
    else:
        if method_used == 'fallback_geo':
            st.caption("📏 Recommandations par proximité géographique + note (pas de texte disponible).")
        else:
            st.caption("💬 Recommandations basées sur le contenu des avis (embeddings sémantiques).")

        for _, row in results.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"**{row['title']}**")
                    if pd.notna(row.get('reviews_text_combined')):
                        st.caption(f"💬 {str(row['reviews_text_combined'])[:150]}...")
                    if 'distance_km' in row:
                        st.caption(f"📏 {row['distance_km']:.2f} km")
                with c2:
                    if 'similarity_score' in row:
                        st.metric("Similarité", f"{row['similarity_score']:.2f}")
                    if pd.notna(row.get('rating_final')):
                        st.metric("Note", f"{row['rating_final']:.1f} ⭐")

        st.markdown("---")
        st.subheader("🗺️ Localisation")
        map_points = results[['lat', 'lng']].dropna().rename(columns={'lng': 'lon'})
        target_point = pd.DataFrame({'lat': [target_row['lat']], 'lon': [target_row['lng']]})
        st.map(pd.concat([map_points, target_point], ignore_index=True))

# ============================================================
# PAGE 2  : RECHERCHE PAR DESCRIPTION LIBRE
# ============================================================
elif page == "Recherche par description":
    st.title("Décris ce que tu cherches")
    st.markdown(
        "Tape une description en langage naturel (en anglais de préférence, "
        "car les avis sont traduits en anglais), ou utilise le micro 🎤.\n\n"
        "*Exemples : \"spicy food and fast service\", \"cheap and good coffee\"*"
    )

    st.write("Ou parle directement :")
    voice_text = speech_to_text(
        language='en',
        start_prompt="🎤 Parler",
        stop_prompt="⏹️ Arrêter",
        just_once=True,
        key='voice_query'
    )

    query = st.text_input(
        "Que recherches-tu ?",
        value=voice_text if voice_text else "",
        placeholder="ex: cheap delicious food"
    )
    n_results_q = st.slider("Nombre de résultats", 3, 10, 5, key="n_query")

    if query:
        results_q = search_by_text_query(query, df_text, embed_model, embeddings, n=n_results_q)

        if results_q.empty:
            st.warning("Aucun restaurant ne correspond suffisamment à cette description.")
        else:
            st.subheader(f"Résultats pour : \"{query}\"")
            for _, row in results_q.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"**{row['title']}**")
                        st.caption(f"{str(row['reviews_text_combined'])[:150]}...")
                    with c2:
                        st.metric("Pertinence", f"{row['similarity_score']:.2f}")
                        if pd.notna(row.get('rating_final')):
                            st.metric("Note", f"{row['rating_final']:.1f} ⭐")

            st.markdown("---")
            st.subheader("🗺️ Localisation des résultats")
            map_points = results_q[['lat', 'lng']].dropna().rename(columns={'lng': 'lon'})
            if not map_points.empty:
                st.map(map_points)
            else:
                st.caption("Pas de coordonnées disponibles pour ces résultats.")
    else:
        st.info("Tape une description ci-dessus pour commencer.")

# ============================================================
# PAGE 3 — STATISTIQUES DU DATASET
# ============================================================
else:
    st.title("Statistiques")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total restaurants", len(df_clean))
    col2.metric("Avec texte exploitable", int(df_clean['has_text'].sum()))
    col3.metric("Avec note", int(df_clean['has_rating'].sum()))
    col4.metric("Note moyenne", f"{df_clean['rating_final'].mean():.2f} ⭐")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Répartition par ville")
        st.bar_chart(df_clean['city_clean'].value_counts())
    with c2:
        st.subheader("Distribution des notes")
        st.bar_chart(df_clean['rating_final'].dropna().round(0).value_counts().sort_index())

    st.markdown("---")
    st.subheader("Carte de tous les restaurants")
    map_data = df_clean[['lat', 'lng']].dropna().rename(columns={'lng': 'lon'})
    st.map(map_data)

    st.markdown("---")
    st.subheader("Aperçu des données nettoyées")
    st.dataframe(df_clean.head(20))
