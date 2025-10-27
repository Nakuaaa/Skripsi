import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import altair as alt
import pandas as pd

# CSS untuk tampilan soft dan elegan (coklat, cream, putih, hitam, pink)
st.markdown("""
    <style>
    /* Background cream */
    .stApp {
        background-color: #F5F5DC;
        color: #000000;
    }
    /* Header coklat */
    h1 {
        color: #8B4513;
        text-align: center;
        font-family: Arial, sans-serif;
        font-size: 36px;
    }
    /* Subheader pink */
    h3 {
        color: #FFB6C1;
        font-family: Arial, sans-serif;
    }
    /* Button coklat */
    .stButton > button {
        background-color: #8B4513;
        color: #FFFFFF;
        border-radius: 10px;
        font-weight: bold;
        padding: 10px 20px;
    }
    /* Success box pink */
    .stSuccess {
        background-color: rgba(255, 182, 193, 0.7);
        border-radius: 5px;
    }
    /* Gambar rekomendasi dengan border soft dan shadow */
    .rec-image {
        border: 2px solid #8B4513;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .rec-image:hover {
        transform: scale(1.05);
    }
    /* Sidebar coklat muda */
    .css-1lcbmhc {
        background-color: #D2B48C;
    }
    </style>
    """, unsafe_allow_html=True)

# Label kelas
NEW_LABEL_DICT = {
    0: 'T-shirt/Top', 1: 'Pants', 2: 'Dress', 3: 'Outerwear',
    4: 'Footwear', 5: 'Shirt', 6: 'Bag', 7: 'Socks',
    8: 'Hat', 9: 'Skirts'
}

# Path model
MODEL_PATH = 'model/vgg16_finetuned_best.keras'

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Fungsi prediksi
from tensorflow.keras.applications.vgg16 import preprocess_input

def predict_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Konversi dari RGB (PIL) ke BGR (cv2 style)
    img = cv2.resize(img, (60, 60))
    img_array = np.expand_dims(img, axis=0) / 255.0
    pred = model.predict(img_array)
    pred_class_idx = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred) * 100
    pred_class = NEW_LABEL_DICT.get(pred_class_idx, "Unknown")
    st.write(f"Debug - Predicted Class: {pred_class}, Confidence: {confidence:.2f}%, Index: {pred_class_idx}")  # Debug
    return pred_class, confidence


# Dictionary rekomendasi gambar dari data testing/training 
RECOMMENDATIONS = {
    'Dress': ['images/dress/dress1.jpg', 'images/dress/dress2.jpg', 'images/dress/dress3.jpg'],
    'Pants': ['images/pants/pants1.jpg', 'images/pants/pants2.jpg', 'images/pants/pants3.jpg'],
    'T-shirt/Top': ['images/t-shirt_top/t-shirt1.jpg', 'images/t-shirt_top/t-shirt2.jpg', 'images/t-shirt_top/t-shirt3.jpg'],
    'Outerwear': ['images/outerwear/outwear1.jpg', 'images/outerwear/outwear2.jpg', 'images/outerwear/outwear3.jpg'],
    'Footwear': ['images/footwear/footwear1.jpg', 'images/footwear/footwear2.jpg', 'images/footwear/footwear3.jpg'],
    'Shirt': ['images/shirt/shirt1.jpg', 'images/shirt/shirt2.jpg', 'images/shirt/shirt3.jpg'],
    'Bag': ['images/bag/bag1.jpg', 'images/bag/bag2.jpg', 'images/bag/bag3.jpg'],
    'Socks': ['images/socks/sock1.jpg', 'images/socks/sock2.jpg', 'images/socks/sock3.jpg'],
    'Hat': ['images/hat/hat1.jpg', 'images/hat/hat2.jpg', 'images/hat/hat3.jpg'],
    'Skirts': ['images/skirts/skirt1.jpg', 'images/skirts/skirt2.jpg', 'images/skirts/skirt3.jpg']
}

# Sidebar
st.sidebar.title("📚 Tentang Skripsi")
st.sidebar.write("**Judul:** Pengenalan Produk E-Commerce Berbasis Gambar Menggunakan CNN")
st.sidebar.write("**Model:** VGG16 (Akurasi 98.08%)")
st.sidebar.write("**Oleh:** Nabila Kurnia Aprianti")
st.sidebar.write("**NIM:**  09021182227003")
st.sidebar.write("**Program Studi:** Teknik Informatika")
st.sidebar.write("**Dosen Pembimbing1:**  Alvi Syahrini Utami, S.Si., M.Kom. ")
st.sidebar.write("**Dosen Pembimbing2:** Anggina Primanita, S.Kom., M.IT., P.hD.")
st.sidebar.image("images/logo-unsri.jpg", use_container_width=True) 

# Header
st.title("🛍️ Pengenalan & Rekomendasi Produk Fashion")
st.markdown("Upload gambar untuk prediksi kategori dan lihat rekomendasi produk serupa nya!")

# Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Gambar Anda")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="uploader")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

with col2:
    if uploaded_file is not None:
        st.subheader("🎯 Hasil Prediksi")
        with st.spinner("Memproses dengan CNN..."):
            pred_class, confidence = predict_image(image)
        
        st.success(f"**{pred_class}** | Confidence: **{confidence:.1f}%**")
        
        # Chart confidence
        data = pd.DataFrame({'Kategori': [pred_class], 'Confidence': [confidence]})
        chart = alt.Chart(data).mark_bar(color='#8B4513').encode(
            x='Kategori', y='Confidence'
        ).properties(width=200, height=150, title="Tingkat Keyakinan")
        st.altair_chart(chart, use_container_width=True)

# Rekomendasi Gambar Serupa (Hanya Gambar dari Data)
if uploaded_file is not None and confidence > 50:
    st.subheader("🖼️ Rekomendasi Produk Serupa")
    with st.expander("Lihat Gambar dari Data Training/Testing"):
        rec_images = RECOMMENDATIONS.get(pred_class, RECOMMENDATIONS['Dress'])  # Fallback ke Dress
        
        # Grid 3 kolom untuk gambar
        cols = st.columns(3)
        for i, img_path in enumerate(rec_images[:3]):  # 3 gambar max
            with cols[i]:
                st.image(img_path, width=150, use_container_width=True, caption="Produk Serupa")
                st.markdown(f'<div class="rec-image"></div>', unsafe_allow_html=True)  # Tambah efek hover

else:
    if uploaded_file is not None:
        st.info("Confidence rendah. Unggah gambar yang lebih jelas untuk rekomendasi!")

# Footer
st.markdown("---")
st.markdown('<p style="color: #8B4513;">*© 2025 Nabila Kurnia Aprianti - Skripsi Teknik Informatika Unsri*</p>', unsafe_allow_html=True)