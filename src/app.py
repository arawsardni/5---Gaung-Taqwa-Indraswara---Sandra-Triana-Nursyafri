from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lime.lime_tabular
import matplotlib.pyplot as plt
import os
from sklearn.base import BaseEstimator, TransformerMixin

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        """ Simpan nama fitur yang akan tetap digunakan """
        self.remaining_features = [col for col in X.columns if col not in self.features]
        return self

    def transform(self, X):
        """ Hapus fitur yang tidak diperlukan """
        return X.drop(columns=self.features, errors='ignore')

    def get_feature_names_out(self, input_features=None):
        """ Pastikan atribut sudah ada sebelum digunakan """
        if not hasattr(self, "remaining_features"):
            raise ValueError("fit() harus dipanggil sebelum get_feature_names_out()")
        return self.remaining_features

    
# Improved time converter with configurable divisor
class TimeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, features, divisor=365.25):
        self.features = features
        self.divisor = divisor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.features] = X_copy[self.features].abs() // self.divisor
        return X_copy
    
# Improved retiree handler
class HandleRetirees(BaseEstimator, TransformerMixin):
    def __init__(self, employment_length_col='EMPLOYMENT_LENGTH', retirement_value=365243):
        self.employment_length_col = employment_length_col
        self.retirement_value = retirement_value
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[X_copy[self.employment_length_col] == self.retirement_value, self.employment_length_col] = 0
        return X_copy
    
# Improved skewness adjuster with configurable transformation
class AdjustSkewness(BaseEstimator, TransformerMixin):
    def __init__(self, features, method='cbrt'):
        self.features = features
        self.method = method  # 'cbrt', 'log', or 'sqrt'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        if self.method == 'cbrt':
            X_copy[self.features] = np.cbrt(X_copy[self.features])
        elif self.method == 'log':
            # Add small constant to avoid log(0)
            X_copy[self.features] = np.log1p(X_copy[self.features])
        elif self.method == 'sqrt':
            X_copy[self.features] = np.sqrt(X_copy[self.features])
            
        return X_copy
    
# Improved one-hot encoder
class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features, handle_unknown="ignore"):
        self.features = features
        self.encoder = OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
        self.feature_names = None

    def fit(self, X, y=None):
        self.encoder.fit(X[self.features])
        self.feature_names = self.encoder.get_feature_names_out(self.features)
        return self

    def transform(self, X):
        encoded = self.encoder.transform(X[self.features])
        encoded_df = pd.DataFrame(encoded, 
                                columns=self.feature_names, 
                                index=X.index)
        return X.drop(columns=self.features).join(encoded_df)
    
# Improved ordinal encoder
class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature, mapping):
        self.feature = feature
        self.mapping = mapping
        self.default_value = -1  # Default value for unknown categories

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.feature] = X_copy[self.feature].map(self.mapping).fillna(self.default_value)
        return X_copy
    
# Improved scaler
class ScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features, scaler=None):
        self.features = features
        self.scaler = scaler or StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.features])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.features] = self.scaler.transform(X_copy[self.features])
        return X_copy
    
# Improved feature binarizer
class BinarizeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features, mapping=None):
        self.features = features
        self.mapping = mapping or {'Y': 1, 'N' : 0}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for feature in self.features:
            if feature in X_copy.columns:
                X_copy[feature] = X_copy[feature].map(self.mapping)
        return X_copy


# ====== Load Model dan Pipeline ======
# Fix the file path using raw string or proper escaping
MODEL_PATH = r"D:\File Gaung\Kuliah TIF UB\BCC\Intern 2025\Project\models\2nd_models\final_model.pkl"

# ====== Streamlit UI ======
st.set_page_config(page_title="Credit Card Approval AI", layout="centered")

st.title("🔍 AI Credit Card Approval System")
st.write("Masukkan informasi Anda untuk mengetahui kemungkinan persetujuan kredit.")

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ Model file '{MODEL_PATH}' tidak ditemukan! Pastikan sudah melakukan training dan menyimpan model.")
    st.stop()

try:
    # Load model yang telah disimpan
    with open(MODEL_PATH, "rb") as f:
        model_data = joblib.load(f)

    trained_models = model_data["models"]  # Semua model yang telah dilatih
    preprocessing_pipeline = model_data["preprocessing_pipeline"]  # Pipeline preprocessing
    optimal_thresholds = model_data["thresholds"]  # Threshold optimal
    
    # Extract feature names from model_data if available
    if "feature_names" in model_data:
        model_feature_names = model_data["feature_names"]
    else:
        model_feature_names = None
        
except Exception as e:
    st.error(f"⚠️ Error loading model: {str(e)}")
    st.stop()

# ====== Sidebar Input Form ======
st.sidebar.header("📝 Masukkan Data Customer")

# Dropdown untuk memilih model
try:
    selected_model_name = st.sidebar.selectbox("Pilih Model AI", list(trained_models.keys()))

    # Ambil model yang dipilih dan threshold optimalnya
    model = trained_models[selected_model_name]
    approval_threshold = optimal_thresholds[selected_model_name]

except Exception as e:
    st.error(f"⚠️ Error loading models: {str(e)}")
    st.stop()

# Numerical Features
age = st.sidebar.number_input("Umur", min_value=0, max_value=100, value=19)
income = st.sidebar.number_input("Pendapatan Tahunan ($)", min_value=0, max_value=1000000, value=100000)
employment_length = st.sidebar.number_input("Lama Bekerja (Tahun)", min_value=0, max_value=50, value=1)
cnt_fam_members = st.sidebar.number_input("Jumlah Anggota Keluarga", min_value=1, max_value=20, value=1)

# Categorical Features
flag_own_car = st.sidebar.selectbox("Memiliki Mobil?", ['Y', 'N'])
flag_own_realty = st.sidebar.selectbox("Memiliki Properti?", ['Y', 'N'])
flag_work_phone = st.sidebar.selectbox("Memiliki Ponsel Kerja", ['Y', 'N'])
name_income_type = st.sidebar.selectbox("Tipe Pendapatan", ['Working', 'Commercial associate', 'State servant', 'Student', 'Pensioner'])
name_education_type = st.sidebar.selectbox("Pendidikan", ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'])
name_family_status = st.sidebar.selectbox("Status Pernikahan", ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'])
name_housing_type = st.sidebar.selectbox("Jenis Tempat Tinggal", ['House / apartment', 'Rented apartment', 'Municipal apartment', 'With parents', 'Co-op apartment', 'Office apartment'])
occupation_type = st.sidebar.selectbox("Pekerjaan", ['Security staff', 'Sales staff', 'Accountants', 'Laborers', 'Managers',
                                                      'Drivers', 'Core staff', 'High skill tech staff', 'Cleaning staff',
                                                      'Private service staff', 'Cooking staff', 'Low-skill Laborers',
                                                      'Medicine staff', 'Secretaries', 'Waiters/barmen staff', 'HR staff',
                                                      'Realty agents', 'IT staff'])

# Konversi input ke DataFrame
user_input = pd.DataFrame({
    'AGE': [age],
    'ANNUAL_INCOME': [income],
    'EMPLOYMENT_LENGTH': [employment_length],
    'CNT_FAM_MEMBERS': [cnt_fam_members],
    'FLAG_OWN_CAR': [flag_own_car],
    'FLAG_OWN_REALTY': [flag_own_realty],
    'FLAG_WORK_PHONE' : [1 if flag_work_phone == 'Y' else 0],
    'EMPLOYMENT_STATUS': [name_income_type],
    'EDUCATION_LEVEL': [name_education_type],
    'MARITAL_STATUS': [name_family_status],
    'DWELLING_TYPE': [name_housing_type],
    'JOB_TITLE': [occupation_type]
})

# ====== Button Prediksi ======
if st.sidebar.button("🔮 Prediksi Kredit"):
    
    try:
        # Transform the input data using preprocessing pipeline
        processed_input = preprocessing_pipeline.transform(user_input)
        
        # Handle different model types with different feature naming conventions
        # Try different ways to get feature names depending on the model type
        if model_feature_names is not None:
            # Use feature names saved during model training if available
            features_for_model = model_feature_names
        elif hasattr(model, 'feature_names_in_'):
            features_for_model = model.feature_names_in_
        elif hasattr(model, 'feature_names_'):
            features_for_model = model.feature_names_
        elif hasattr(model, 'feature_name_'):
            features_for_model = model.feature_name_
        else:
            # If we can't get feature names, create generic names
            features_for_model = [f'feature_{i}' for i in range(processed_input.shape[1])]

            
        
        # Convert processed input to DataFrame
        processed_input_df = pd.DataFrame(processed_input, columns=features_for_model)

        
        # Make prediction
        probabilities = model.predict_proba(processed_input_df)[0]

        # Use threshold for prediction
        prediction = 1 if probabilities[1] >= approval_threshold else 0
        
        # Display results
        st.subheader(f"📊 Hasil Prediksi dengan Model: **{selected_model_name}**")
        st.write(f"Probabilitas Kredit Diterima: {probabilities[0]:.2f} \n Probabilitas Ditolak {probabilities[1]:.2f}")
        # Tambahkan kode ini di bagian prediksi
        st.write("Raw probabilities:", probabilities)
        st.write(f"Threshold: {approval_threshold:.2f}")
        
        if prediction == 1:
            st.error(f"❌ Kredit **Ditolak** dengan probabilitas {probabilities[1]:.2f}")  # 1 = BAD
        else:
            st.success(f"✅ Kredit **Diterima** dengan probabilitas {probabilities[0]:.2f}")  # 0 = GOOD

        import shap
        # SHAP Explanation
        explainer = shap.TreeExplainer(model)  # Jika model adalah tree-based seperti XGBoost atau RandomForest

        # Pastikan SHAP values tidak kosong
        shap_values = explainer.shap_values(processed_input_df)

        if shap_values is not None and len(shap_values) > 0:
            # Plot SHAP hanya jika hasilnya valid
            shap.summary_plot(shap_values, processed_input_df)
        else:
            st.warning("⚠️ Tidak ada nilai SHAP yang bisa ditampilkan. Periksa kembali model atau input data.")

        try:
            st.subheader("📊 Interpretasi Model dengan SHAP")
            
            # Store processed shap values in variables for reuse
            if isinstance(shap_values, list):
                # For models that return a list of arrays (like RandomForest)
                display_shap_values = shap_values[1]
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            elif len(np.array(shap_values).shape) == 3:
                # For 3D arrays [samples, features, classes]
                class_idx = 1 if prediction == 1 else 0
                display_shap_values = shap_values[0, :, class_idx]
                base_value = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                # For 2D arrays [samples, features]
                display_shap_values = shap_values[0]
                base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
            
            # Create tabs for different visualization options
            viz_tab1, viz_tab3, viz_tab4 = st.tabs(["Penjelasan Sederhana", "Waterfall Plot", "Summary Plot"])
            
            with viz_tab1:
                st.subheader("✅ Penjelasan Sederhana")
                
                def generate_friendly_explanation(shap_values, feature_names, prediction, input_data):
                    # Get the feature values
                    feature_values = input_data.iloc[0].values
                    
                    # Get absolute SHAP values for feature importance ranking
                    abs_shap = np.abs(shap_values)
                    
                    # Get the top 5 most important features
                    top_indices = np.argsort(abs_shap)[-5:][::-1]
                    
                    # Create a friendly explanation
                    decision = "DITOLAK" if prediction == 1 else "DITERIMA"
                    
                    # Start building the explanation - Changed background colors for better contrast
                    # Using darker background colors with white text for better readability
                    explanation = f"<div style='background-color: {'#9c2b2b' if prediction == 1 else '#1e5b1e'}; color: white; padding: 15px; border-radius: 10px;'>"
                    explanation += f"<h3>Mengapa Aplikasi Kredit Anda {decision}?</h3>"
                    
                    # Add decision icon
                    icon = "❌" if prediction == 1 else "✅"
                    explanation += f"<p style='font-size: 24px;'>{icon} Keputusan: <b>{decision}</b></p>"
                    
                    # Explain top factors
                    explanation += "<p><b>Faktor Utama yang Mempengaruhi Keputusan:</b></p>"
                    explanation += "<ol style='color: white;'>"
                    
                    for i, idx in enumerate(top_indices):
                        feature = feature_names[idx]
                        impact = shap_values[idx]
                        feature_value = feature_values[idx]
                        
                        # Make feature names more readable
                        readable_feature = feature.replace('_', ' ').title()
                        readable_feature = readable_feature.replace('Flag Own', 'Memiliki')
                        readable_feature = readable_feature.replace('Flag Work Phone', 'Punya Telepon Kerja')
                        readable_feature = readable_feature.replace('Car N', 'Mobil: Tidak')
                        readable_feature = readable_feature.replace('Car Y', 'Mobil: Ya')
                        readable_feature = readable_feature.replace('Realty N', 'Properti: Tidak')
                        readable_feature = readable_feature.replace('Realty Y', 'Properti: Ya')
                        readable_feature = readable_feature.replace('Employment Status', 'Status Pekerjaan')
                        readable_feature = readable_feature.replace('Job Title', 'Pekerjaan')
                        readable_feature = readable_feature.replace('Dwelling Type', 'Jenis Tempat Tinggal')
                        
                        # Determine impact direction and strength
                        if impact < 0:
                            direction = "mendukung PERSETUJUAN kredit"
                            color = "#8effad"  # Lighter green for contrast on dark background
                        else:
                            direction = "mendukung PENOLAKAN kredit"
                            color = "#ffacac"  # Lighter red for contrast on dark background
                        
                        # Determine strength text
                        strength = abs(impact)
                        if strength > 0.1:
                            strength_text = "sangat signifikan"
                        elif strength > 0.05:
                            strength_text = "cukup signifikan"
                        else:
                            strength_text = "sedikit"
                        
                        explanation += f"<li><b>{readable_feature}</b>: <span style='color:{color}'>{strength_text} {direction}</span></li>"
                    
                    explanation += "</ol>"
                    
                    # Add advice section if rejected
                    if prediction == 1:  # Rejected
                        explanation += "<h4>Saran untuk Meningkatkan Peluang Persetujuan:</h4>"
                        explanation += "<ul style='color: white;'>"
                        for idx in top_indices:
                            if shap_values[idx] > 0:  # Contributing to rejection
                                feature = feature_names[idx]
                                
                                if "INCOME" in feature.upper():
                                    explanation += "<li>Tingkatkan pendapatan Anda atau cari sumber pendapatan tambahan</li>"
                                elif "EMPLOYMENT" in feature.upper():
                                    explanation += "<li>Stabilkan riwayat pekerjaan Anda (lama bekerja)</li>"
                                elif "OWN_CAR_N" in feature.upper() or "OWN_REALTY_N" in feature.upper():
                                    explanation += "<li>Pertimbangkan untuk memiliki aset seperti mobil atau properti</li>"
                                elif "AGE" in feature.upper():
                                    explanation += "<li>Faktor usia memengaruhi keputusan. Aplikasi mungkin lebih cocok diajukan beberapa tahun lagi</li>"
                                elif "FAM_MEMBERS" in feature.upper():
                                    explanation += "<li>Jumlah anggota keluarga memengaruhi keputusan. Pertimbangkan untuk mengajukan kredit bersama anggota keluarga lain</li>"
                                elif "EDUCATION" in feature.upper():
                                    explanation += "<li>Tingkatkan kualifikasi pendidikan Anda jika memungkinkan</li>"
                                elif "JOB_TITLE" in feature.upper():
                                    explanation += "<li>Pekerjaan Anda memengaruhi keputusan. Pastikan sudah memiliki stabilitas pekerjaan</li>"
                        explanation += "</ul>"
                    
                    explanation += "</div>"
                    return explanation
                
                # Display friendly explanation
                explanation_html = generate_friendly_explanation(display_shap_values, processed_input_df.columns, prediction, processed_input_df)
                st.markdown(explanation_html, unsafe_allow_html=True)
            
            # with viz_tab2:
            #     st.subheader("📊 Force Plot: Pengaruh Fitur pada Keputusan")
            #     st.write("Diagram ini menunjukkan bagaimana setiap fitur mendorong keputusan (merah = menolak, biru = menyetujui)")
                
            #     try:
            #         # Create force plot
            #         fig_force = plt.figure(figsize=(10, 3))
            #         shap.plots.force(base_value, 
            #                         display_shap_values,
            #                         features=processed_input_df.iloc[0,:], 
            #                         feature_names=processed_input_df.columns,
            #                         matplotlib=True,
            #                         show=False)
            #         plt.tight_layout()
            #         st.pyplot(fig_force)
                    
            #         # Add explanation for force plot
            #         st.info("""
            #         **Cara membaca diagram ini:**
            #         - Bar **merah** mendorong ke arah PENOLAKAN kredit
            #         - Bar **biru** mendorong ke arah PERSETUJUAN kredit
            #         - Garis f(x) adalah hasil akhir prediksi
            #         - Panjang bar menunjukkan kekuatan pengaruh fitur tersebut
            #         """)
            #     except Exception as e:
            #         st.error(f"Error saat membuat Force Plot: {str(e)}")
            
            with viz_tab3:
                st.subheader("🌊 Waterfall Plot: Aliran Kontribusi Fitur")
                st.write("Diagram ini menunjukkan kontribusi satu per satu dari fitur-fitur terpenting")
                
                try:
                    # Create waterfall plot
                    fig_waterfall = plt.figure(figsize=(10, 8))
                    
                    # Perbaikan untuk handling base_value yang berbeda struktur
                    if isinstance(base_value, (int, float)):
                        base_val = base_value
                    elif hasattr(base_value, '__len__') and len(base_value) > 0:
                        base_val = base_value[0]
                    else:
                        # Jika base_value tidak bisa diakses, gunakan nilai default
                        base_val = 0
                        
                    # Pastikan display_shap_values memiliki dimensi yang benar
                    if len(display_shap_values.shape) > 1:
                        # Jika 2D, ambil baris pertama
                        shap_vals = display_shap_values[0]
                    else:
                        # Jika 1D, gunakan as-is
                        shap_vals = display_shap_values
                        
                    # Buat explanation object dengan lebih banyak handling error
                    explanation = shap.Explanation(
                        values=shap_vals,
                        base_values=base_val,
                        data=processed_input_df.iloc[0,:].values,
                        feature_names=processed_input_df.columns.tolist()
                    )
                    
                    # Buat waterfall plot
                    shap.plots.waterfall(explanation, show=False)
                    plt.tight_layout()
                    st.pyplot(fig_waterfall)
                    
                    # Add explanation for waterfall plot
                    st.info("""
                    **Cara membaca diagram ini:**
                    - Diagram menunjukkan bagaimana keputusan dibangun dari fitur teratas
                    - Bar **merah** meningkatkan probabilitas penolakan
                    - Bar **biru** menurunkan probabilitas penolakan (mendukung persetujuan)
                    - E[f(X)] adalah nilai prediksi rata-rata untuk semua pemohon
                    - f(x) adalah hasil akhir prediksi untuk pemohon ini
                    """)
                except Exception as e:
                    st.error(f"Error saat membuat Waterfall Plot: {str(e)}")
                    
                    # Fallback jika waterfall plot gagal
                    st.warning("Mencoba menampilkan alternatif visualisasi...")
                    try:
                        # Menampilkan bar plot sederhana sebagai alternatif
                        fig_alt = plt.figure(figsize=(10, 8))
                        
                        # Ambil top 10 fitur berdasarkan nilai absolut SHAP
                        if len(display_shap_values.shape) > 1:
                            shap_vals = display_shap_values[0]
                        else:
                            shap_vals = display_shap_values
                            
                        feature_names = processed_input_df.columns.tolist()
                        feature_importance = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': np.abs(shap_vals)
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        colors = ['red' if v > 0 else 'blue' for v in shap_vals[feature_importance.index]]
                        plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
                        plt.title('Top 10 Fitur Berdasarkan Nilai SHAP')
                        plt.xlabel('Nilai Absolut SHAP')
                        plt.tight_layout()
                        st.pyplot(fig_alt)
                        
                        st.info("""
                        **Visualisasi Alternatif:**
                        - Diagram batang menunjukkan fitur-fitur paling berpengaruh
                        - Warna **merah** menunjukkan kontribusi positif (menaikkan probabilitas penolakan)
                        - Warna **biru** menunjukkan kontribusi negatif (menurunkan probabilitas penolakan)
                        """)
                    except Exception as e2:
                        st.error(f"Gagal menampilkan visualisasi alternatif: {str(e2)}")
            
            with viz_tab4:
                st.subheader("📈 Summary Plot: Ringkasan Pengaruh Fitur")
                st.write("Diagram teknis yang menunjukkan dampak semua fitur secara bersamaan")
                
                try:
                    # Create traditional summary plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[1], processed_input_df, show=False)
                    elif len(np.array(shap_values).shape) == 3:
                        class_idx = 1 if prediction == 1 else 0
                        shap.summary_plot(shap_values[:, :, class_idx], processed_input_df, show=False)
                    else:
                        shap.summary_plot(shap_values, processed_input_df, show=False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Add explanation for summary plot
                    st.info("""
                    **Cara membaca diagram ini (untuk pengguna teknis):**
                    - Fitur diurutkan berdasarkan kepentingan (paling penting di atas)
                    - Posisi horizontal menunjukkan dampak pada keputusan:
                    - Nilai negatif (kiri) mendukung persetujuan
                    - Nilai positif (kanan) mendukung penolakan
                    - Warna menunjukkan nilai fitur (biru = rendah, merah = tinggi)
                    """)
                except Exception as e:
                    st.error(f"Error saat membuat Summary Plot: {str(e)}")

        except Exception as e:
            st.error(f"⚠️ Error saat menampilkan SHAP plot: {str(e)}")
            st.text(f"Struktur SHAP values: {np.array(shap_values).shape}")
    
            # Add more diagnostic information
            st.text(f"Tipe data SHAP values: {type(shap_values)}")
            if isinstance(shap_values, list):
                st.text(f"Jumlah elemen dalam list: {len(shap_values)}")
                for i, sv in enumerate(shap_values):
                    st.text(f"Shape elemen {i}: {np.array(sv).shape}")

    except Exception as e:
        st.error(f"⚠️ Error saat melakukan prediksi: {str(e)}")
        st.info("Coba periksa kembali data yang dimasukkan atau pilih model yang berbeda.")
    


# ====== Tambahan Footer ======
st.markdown(
    """
    ---
    👨‍💻 **Dibuat oleh:** Gaung Taqwa dan Sandra Triana\n
    📌 **Tujuan:** Membantu pengguna dalam menentukan kelayakan kredit dengan AI  
    """
)