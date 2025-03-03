## Dashboard Rekomendasi Produk Fashion by AZHAR ZUHRO

### Latar Belakang
Industri fashion semakin berkembang dengan banyaknya pilihan produk yang tersedia. Pengguna sering kali kesulitan menemukan produk yang sesuai dengan preferensi mereka. Oleh karena itu, sistem rekomendasi berbasis kecerdasan buatan dapat membantu pengguna dalam menemukan produk yang paling relevan dengan preferensi mereka.

### Business Understanding
Untuk meningkatkan penjualan dan kepuasan pelanggan, perusahaan fashion memerlukan sistem yang dapat membantu pengguna menemukan produk yang sesuai dengan preferensi mereka dengan lebih cepat dan efisien. Dengan memberikan rekomendasi produk yang relevan, diharapkan pengguna dapat melakukan pembelian dengan keputusan yang lebih baik dan nyaman.

### Problem Statement
Banyak pengguna mengalami kesulitan dalam memilih produk fashion yang sesuai dengan selera mereka karena banyaknya pilihan yang tersedia. Sistem rekomendasi yang baik dapat membantu mengurangi waktu pencarian dan meningkatkan pengalaman belanja pengguna dengan memberikan rekomendasi produk yang dipersonalisasi berdasarkan riwayat interaksi mereka.

### Goals
- Membantu pengguna menemukan produk fashion yang sesuai dengan preferensi mereka.
- Mengurangi waktu pencarian produk.
- Meningkatkan pengalaman belanja pengguna dengan rekomendasi yang relevan.
- Meningkatkan penjualan dengan memberikan rekomendasi yang sesuai dengan kebutuhan pengguna.

### Solution Statement
Mengembangkan dashboard berbasis Streamlit yang menggunakan model collaborative filtering untuk memberikan rekomendasi produk fashion yang dipersonalisasi berdasarkan riwayat rating pengguna.

### Data Understanding
Dataset yang digunakan adalah `fashion_products.csv` yang berisi informasi tentang produk fashion dan rating pengguna. Kolom-kolom dalam dataset meliputi:
- **Product ID:** ID unik untuk setiap produk.
- **Product Name:** Nama produk.
- **Brand:** Merek produk.
- **Category:** Kategori produk.
- **Price:** Harga produk.
- **Rating:** Rating yang diberikan pengguna untuk produk tersebut.

### Kesimpulan
Dashboard ini membantu pengguna menemukan produk fashion yang sesuai dengan preferensi mereka melalui sistem rekomendasi yang dipersonalisasi. Dengan visualisasi dan fitur interaktif, pengguna dapat dengan mudah menjelajahi produk terpopuler dan mendapatkan rekomendasi berdasarkan riwayat rating mereka. Sistem ini juga diharapkan dapat meningkatkan kepuasan pelanggan dan mendorong peningkatan penjualan.

### Fitur
- **Lihat Produk Terpopuler:** Menampilkan produk fashion dengan rating tertinggi.
- **Visualisasi Rating Produk:** Grafik batang yang menunjukkan rating produk terpopuler.
- **Rekomendasi Berdasarkan Pengguna:** Memberikan rekomendasi produk berdasarkan rating pengguna menggunakan model collaborative filtering.

### Prasyarat
- Python 3.8+
- TensorFlow 2.x
- Streamlit

### Persiapan
1. Clone repositori:
```bash
git clone <https://github.com/azhrzuhro/fashion-product-recommendation>
cd <repo_folder>
```

2. Instal dependensi:
```bash
pip install -r requirements.txt
```

3. Pastikan file berikut tersedia:
- `save model/collaborative_filtering_model.keras` — Model rekomendasi yang telah dilatih
- `user_encoder.joblib` — Encoder untuk User ID
- `product_encoder.joblib` — Encoder untuk Product ID
- `data/fashion_products.csv` — Dataset produk fashion

4. Jalankan aplikasi:
```bash
streamlit run app.py
```

### Struktur File
```
.
|-- app.py
|-- save model/
|   |-- collaborative_filtering_model.keras
|-- data/
|   |-- fashion_products.csv
|-- user_encoder.joblib
|-- product_encoder.joblib
|-- requirements.txt
```

### Cara Penggunaan
- Buka sidebar dan masukkan User ID.
- Atur jumlah rekomendasi menggunakan slider.
- Klik "Dapatkan Rekomendasi" untuk melihat rekomendasi yang dipersonalisasi.
- Lihat detail produk dengan membuka bagian produk secara individu.

