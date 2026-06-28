
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, sin, cos, sqrt, atan2

def get_similar_restaurants(restaurant_id, df_text, sim_matrix, n=5):

    """Retourne les n restaurants les plus similaires à un restaurant donné (content-based, embeddings)."""

    matches = df_text.index[df_text['restaurant_id'] == restaurant_id]
    if len(matches) == 0:
        return pd.DataFrame()

    idx = matches[0]
    sims = sim_matrix[idx]
    order = sims.argsort()[::-1]
    order = order[order != idx]
    top_idx = order[:n]

    result = df_text.iloc[top_idx].copy()
    result['similarity_score'] = sims[top_idx]
    return result.reset_index(drop=True)


def haversine_distance(lat1, lng1, lat2, lng2):
    """Distance en km entre deux points GPS (formule de Haversine)."""
    R = 6371.0
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_fallback_recommendations(restaurant_id, df_clean, n=5, max_distance_km=10):

    """Pour les restaurants SANS texte : recommandation par proximité géographique + note."""
    
    target = df_clean[df_clean['restaurant_id'] == restaurant_id]
    if len(target) == 0:
        return pd.DataFrame()
    target = target.iloc[0]

    candidates = df_clean[df_clean['restaurant_id'] != restaurant_id].copy()
    candidates['distance_km'] = candidates.apply(
        lambda r: haversine_distance(target['lat'], target['lng'], r['lat'], r['lng']),
        axis=1
    )
    candidates = candidates[candidates['distance_km'] <= max_distance_km]
    candidates['rating_filled'] = candidates['rating_final'].fillna(0)
    candidates = candidates.sort_values(by=['distance_km', 'rating_filled'], ascending=[True, False])
    return candidates.head(n).reset_index(drop=True)


def recommend(restaurant_id, df_clean, df_text, sim_matrix, n=5, max_distance_km=10):

    """Choisit automatiquement entre content-based (embeddings) et fallback géo."""

    has_text = df_clean.loc[df_clean['restaurant_id'] == restaurant_id, 'has_text']
    if len(has_text) == 0:
        return pd.DataFrame(), 'not_found'

    if has_text.iloc[0]:
        results = get_similar_restaurants(restaurant_id, df_text, sim_matrix, n=n)
        return results, 'content_based'
    else:
        results = get_fallback_recommendations(restaurant_id, df_clean, n=n, max_distance_km=max_distance_km)
        return results, 'fallback_geo'


def search_by_text_query(query, df_text, embed_model, embeddings, n=5):
    
    """
    Chatbot : transforme une requête libre ("épicé et pas cher") en
    embedding sémantique, dans le même espace que les restaurants, puis
    renvoie les plus proches par similarité cosinus.
    """
    
    query_vec = embed_model.encode([query])
    sims = cosine_similarity(query_vec, embeddings).flatten()
    top_idx = sims.argsort()[::-1][:n]

    result = df_text.iloc[top_idx].copy()
    result['similarity_score'] = sims[top_idx]
    result = result[result['similarity_score'] > 0.3]  # seuil calibré sur l'échelle des embeddings
    return result.reset_index(drop=True)