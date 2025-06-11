import streamlit as st
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Load model dan encoder
model = ('save model/collaborative_filtering_model.keras')
user_encoder = joblib.load('user_encoder.joblib')
product_encoder = joblib.load('product_encoder.joblib')
product_data = pd.read_csv('data/fashion_products.csv')  # Pastikan dataset tersedia

if 'product_encoded' not in product_data.columns:
    product_data['product_encoded'] = product_data['Product ID'].map(lambda x: product_encoder.transform([x])[0])


def get_user_rated_products(user_id):
    rated_products = product_data[['User ID', 'Product ID']].drop_duplicates()
    rated_products = rated_products[rated_products['User ID'] == user_id]
    
    rated_product_details = product_data[product_data['Product ID'].isin(rated_products['Product ID'])]
    rated_product_details = rated_product_details[['Product ID', 'Product Name', 'Brand', 'Category', 'Price', 'Rating']]
    
    return rated_product_details.sort_values(by='Rating', ascending=False).reset_index(drop=True)

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
    
    rated_products = product_data[product_data['User ID'] == user_id]['product_encoded'].values
    predicted_ratings_df = predicted_ratings_df[~predicted_ratings_df['product_encoded'].isin(rated_products)]
    
    recommended = predicted_ratings_df.nlargest(top_n, 'predicted_rating')
    recommended_products = product_data[product_data['product_encoded'].isin(recommended['product_encoded'])]
    
    return recommended_products[['Product ID', 'Product Name', 'Brand', 'Category', 'Price', 'Rating']]

# Streamlit UI
st.title("📌 Dashboard Rekomendasi Produk")

# Tampilkan rekomendasi umum di halaman awal
st.subheader("🌟 Rekomendasi Produk Terpopuler")
popular_products = product_data.sort_values(by='Rating', ascending=False).head(5)
st.dataframe(popular_products[['Product ID', 'Product Name', 'Brand', 'Category', 'Price', 'Rating']])

# Visualisasi rating produk terpopuler
st.subheader("📊 Visualisasi Rating Produk Terpopuler")
plt.figure(figsize=(8, 6))
plt.bar(popular_products['Product Name'], popular_products['Rating'], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Nama Produk')
plt.ylabel('Rating')
plt.title('Rating Produk Terpopuler')
st.pyplot(plt)

st.sidebar.header("🔍 Pilih User ID")
user_id = st.sidebar.number_input("Masukkan User ID", min_value=1, step=1)

top_n = st.sidebar.slider("Jumlah Rekomendasi", min_value=1, max_value=10, value=5)

if st.sidebar.button("Dapatkan Rekomendasi"):
    rated_products = get_user_rated_products(user_id)
    recommended_products = get_recommendations(user_id, top_n)
    
    if rated_products.empty:
        st.warning(f"User ID {user_id} belum memiliki rating.")
    else:
        st.subheader("📌 Produk yang Sudah Dirating")
        st.dataframe(rated_products)
    
    if recommended_products.empty:
        st.warning("Tidak ada rekomendasi untuk user ini.")
    else:
        st.subheader(f"🌟 Rekomendasi Produk untuk User ID {user_id}")
        st.dataframe(recommended_products)
        
        for _, row in recommended_products.iterrows():
            with st.expander(f"🔹 {row['Product Name']}"):
                st.write(f"**Brand:** {row['Brand']}")
                st.write(f"**Kategori:** {row['Category']}")
                st.write(f"**Harga:** ${row['Price']:.2f}")
                st.write(f"**Rating:** {row['Rating']} ⭐")
