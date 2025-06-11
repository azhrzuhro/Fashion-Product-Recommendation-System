import streamlit as st
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# ----------------- SETUP PAGE -----------------
st.set_page_config(page_title="Rekomendasi Produk Fashion", page_icon="🛍️", layout="wide")
st.title("🛍️ Sistem Rekomendasi Produk Fashion")
st.markdown("Berbasis **Collaborative Filtering Model** dengan TensorFlow & Streamlit.")

# ----------------- LOAD MODEL DAN ENCODER -----------------
@st.cache_resource
def load_model_and_encoders():
    model = tf.keras.models.load_model('save model/collaborative_filtering_model.keras')
    user_encoder = joblib.load('user_encoder.joblib')
    product_encoder = joblib.load('product_encoder.joblib')
    return model, user_encoder, product_encoder

model, user_encoder, product_encoder = load_model_and_encoders()

# ----------------- LOAD DATA PRODUK -----------------
@st.cache_data
def load_product_data():
    df = pd.read_csv('data/fashion_products.csv')
    if 'product_encoded' not in df.columns:
        df['product_encoded'] = df['Product ID'].map(lambda x: product_encoder.transform([x])[0])
    return df

product_data = load_product_data()

# ----------------- FUNGSI: PRODUK YANG DIRATING -----------------
def get_user_rated_products(user_id):
    rated = product_data[product_data['User ID'] == user_id]
    rated = rated[['Product ID', 'Product Name', 'Brand', 'Category', 'Price', 'Rating']]
    return rated.sort_values(by='Rating', ascending=False).drop_duplicates()

# ----------------- FUNGSI: DAPATKAN REKOMENDASI -----------------
def get_recommendations(user_id, top_n=5):
    if user_id not in user_encoder.classes_:
        return pd.DataFrame()
    
    user_encoded = user_encoder.transform([user_id])[0]
    product_ids = np.arange(len(product_encoder.classes_))
    user_product_combinations = np.array([[user_encoded, pid] for pid in product_ids])
    
    predictions = model.predict([user_product_combinations[:, 0], user_product_combinations[:, 1]], verbose=0)
    
    predicted_ratings_df = pd.DataFrame({
        'product_encoded': product_ids,
        'predicted_rating': predictions.flatten()
    })

    # Hindari produk yang sudah dirating user
    rated_products = product_data[product_data['User ID'] == user_id]['product_encoded'].values
    predicted_ratings_df = predicted_ratings_df[~predicted_ratings_df['product_encoded'].isin(rated_products)]

    recommended = predicted_ratings_df.nlargest(top_n, 'predicted_rating')
    recommended_products = product_data[product_data['product_encoded'].isin(recommended['product_encoded'])]

    return recommended_products[['Product ID', 'Product Name', 'Brand', 'Category', 'Price', 'Rating']].drop_duplicates()

# ----------------- SIDEBAR -----------------
st.sidebar.header("🔍 Rekomendasi Personalisasi")
available_users = sorted(product_data['User ID'].unique())
user_id = st.sidebar.selectbox("Pilih User ID", available_users)
top_n = st.sidebar.slider("Jumlah Rekomendasi", min_value=1, max_value=10, value=5)

# ----------------- PRODUK TERPOPULER -----------------
st.subheader("🌟 Rekomendasi Produk Terpopuler")
popular_products = product_data.sort_values(by='Rating', ascending=False).drop_duplicates('Product ID').head(5)
st.dataframe(popular_products[['Product ID', 'Product Name', 'Brand', 'Category', 'Price', 'Rating']])

# Visualisasi
st.subheader("📊 Visualisasi Rating Produk Terpopuler")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(popular_products['Product Name'], popular_products['Rating'], color='skyblue')
ax.set_ylabel("Rating")
ax.set_title("Top 5 Produk Terpopuler")
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# ----------------- TOMBOL DAN OUTPUT -----------------
if st.sidebar.button("🎯 Dapatkan Rekomendasi"):
    rated_products = get_user_rated_products(user_id)
    recommended_products = get_recommendations(user_id, top_n)

    if rated_products.empty:
        st.warning(f"User ID {user_id} belum memiliki rating.")
    else:
        st.subheader("✅ Produk yang Sudah Dirating")
        st.dataframe(rated_products)

    if recommended_products.empty:
        st.warning("❗ Tidak ada rekomendasi tersedia untuk user ini.")
    else:
        st.subheader(f"🎁 Rekomendasi Produk untuk User ID {user_id}")
        st.dataframe(recommended_products)

        # Detail tiap produk
        for _, row in recommended_products.iterrows():
            with st.expander(f"🔹 {row['Product Name']}"):
                st.write(f"**Brand:** {row['Brand']}")
                st.write(f"**Kategori:** {row['Category']}")
                st.write(f"**Harga:** ${row['Price']:.2f}")
                st.write(f"**Rating:** {row['Rating']} ⭐")

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("💡 Dibangun oleh Tim Data Jaya Jaya Maju | Powered by TensorFlow & Streamlit")
