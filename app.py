import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
#from googletrans import Translator
import base64

# ===================== BACKGROUND =====================
def set_bg_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg_image(r"agriculture.webp")

# ===================== CSS STYLING =====================
st.markdown("""
<style>
.stApp {font-family:'Segoe UI', sans-serif; color:black;}

/* Buttons */
button {background-color:#0B3D91 !important; color:white !important; border-radius:8px; font-weight:bold;}

/* Cards */
.card {background-color:#007BFF; color:white; padding:20px; border-radius:15px; margin-bottom:20px; box-shadow:0 4px 8px rgba(0,0,0,0.2); text-align:center;}

/* Section headers */
h2 {color:#0B3D91;}
</style>
""", unsafe_allow_html=True)

# ===================== TITLE =====================
st.markdown(
    '<h1 style="text-align:center; color:#0B3D91;">🌱 AGRINOVA – SMART IRRIGATION DASHBOARD 🌱</h1>'
    '<p style="text-align:center; font-size:18px;">Enter soil parameters to predict soil moisture (%) and visualize irrigation recommendations</p>',
    unsafe_allow_html=True
)

# ===================== LOAD DATA =====================
filename = r"Soil_moisture.csv"
df = pd.read_csv(filename)
df.columns = df.columns.str.strip().str.lower()
for col in df.columns:
    df[col] = df[col].astype(str).str.replace('%','').str.replace('mm','').str.strip()
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# ===================== FEATURES & TARGET =====================
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ===================== MODEL =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ===================== SIDEBAR =====================
st.sidebar.title("📊 Model Performance")
st.sidebar.metric("R² Score", round(r2,3))
st.sidebar.metric("RMSE", round(rmse,3))

# ===================== LANGUAGE =====================
#translator = Translator()
#lang_choice = st.sidebar.selectbox("🌍 Choose Language", ("English","తెలుగు","ಕನ್ನಡ","हिन्दी","اردو"))
#lang_map = {"English":"en","తెలుగు":"te","ಕನ್ನಡ":"kn","हिन्दी":"hi","اردو":"ur"}
#lang_code = lang_map[lang_choice]
#def t(text):
   # if lang_code=="en": return text
   # return translator.translate(text, dest=lang_code).text

# ===================== INPUT FIELDS =====================
st.markdown("## 📝 Input Soil Parameters")
input_cols = st.columns(len(X.columns))
inputs = []
for i,col in enumerate(X.columns):
    val = input_cols[i].text_input(label=(col), placeholder=(f"Enter {col}"))
    inputs.append(val)

# ===================== PREDICTION & DASHBOARD =====================
if st.button(("Predict Moisture")):
    try:
        input_array = np.array([float(i) for i in inputs]).reshape(1,-1)
    except ValueError:
        st.error(("Please enter valid numeric values"))
        st.stop()

    moisture = model.predict(input_array)[0]

    # ===================== CARD: Prediction =====================
    st.markdown(f'<div class="card"><h2>{("Predicted Soil Moisture")}</h2><h1>{round(moisture,2)}%</h1></div>', unsafe_allow_html=True)

    # ===================== IRRIGATION DECISION =====================
    LOW_THRESHOLD = 30
    HIGH_THRESHOLD = 60
    if moisture < LOW_THRESHOLD: rec_text="⚠️ Under-Irrigation Detected"
    elif moisture > HIGH_THRESHOLD: rec_text="⚠️ Over-Irrigation Detected"
    else: rec_text="✅ Optimal Irrigation Level"

    st.markdown(f'<div class="card"><h2>{("Irrigation Recommendation")}</h2><h3>{t(rec_text)}</h3></div>', unsafe_allow_html=True)

    # ===================== DASHBOARD TABS =====================
    tabs = st.tabs([("Soil Moisture Gauge"), t("Irrigation Graphs")])
    
    # ---------------- Soil Moisture Gauge Tab ----------------
    with tabs[0]:
        st.markdown("## 💧 Soil Moisture Gauge")
        st.progress(min(int(moisture),100))
        st.caption("🔴 Dry | 🟢 Optimal | 🔵 Over-Irrigation")
    
    # ---------------- Irrigation Graphs Tab ----------------
    with tabs[1]:
        field_capacity = 60
        deficit = max(field_capacity - moisture,0)

        # 5️⃣ Irrigation Requirement & 6️⃣ Moisture Deficit vs Volume side by side
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("### 1️⃣ Irrigation Requirement")
            levels = ["Low","Medium","High"]
            water_needed = [deficit*0.5, deficit*1.0, deficit*1.5]
            fig,ax = plt.subplots()
            ax.bar(levels, water_needed, color='deepskyblue', edgecolor='black')
            ax.set_ylabel("Water Quantity (mm)")
            ax.set_xlabel("Irrigation Level")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
        with col2:
            st.markdown("### 2️⃣ Moisture Deficit vs Irrigation Volume")
            deficit_range = np.linspace(0,60,10)
            irrigation_volume = deficit_range*1.2
            fig,ax = plt.subplots()
            ax.plot(deficit_range, irrigation_volume, marker='o', color='blue', linewidth=2)
            ax.set_xlabel("Moisture Deficit (%)")
            ax.set_ylabel("Irrigation Volume (mm)")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

        # 7️⃣ Irrigation Frequency & 8️⃣ Water Stress side by side
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("### 3️⃣ Irrigation Frequency Recommendation")
            freq = ["Daily","Alternate Days","Weekly"]
            suitability = [90 if moisture<30 else 40, 70 if 30<=moisture<=60 else 50, 80 if moisture>60 else 30]
            fig,ax = plt.subplots()
            ax.bar(freq, suitability, color='deepskyblue', edgecolor='black')
            ax.set_ylabel("Suitability Score")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
        with col2:
            st.markdown("### 4️⃣ Water Stress Risk Indicator")
            if moisture<30: stress=90
            elif moisture>60: stress=40
            else: stress=10
            st.progress(stress)
            st.caption("🔴 High | 🟡 Medium | 🟢 Low")

        # 9️⃣ Irrigation Efficiency & 🔟 Moisture Recovery side by side
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("### 5️⃣ Irrigation Efficiency Comparison")
            methods=["Traditional","Smart Irrigation"]
            water_use=[100,65]
            fig,ax = plt.subplots()
            ax.bar(methods,water_use,color='deepskyblue',edgecolor='black')
            ax.set_ylabel("Water Used (%)")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
        with col2:
            st.markdown("### 6️⃣ Moisture Recovery After Irrigation")
            time_points = list(range(6))
            recovery = moisture + np.array([0,5,10,12,13,14])
            fig,ax = plt.subplots()
            ax.plot(time_points, recovery, marker='o', color='blue', linewidth=2)
            ax.set_xlabel("Time After Irrigation (hrs)")
            ax.set_ylabel("Soil Moisture (%)")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

    st.markdown("### ✅ Smart irrigation analysis completed successfully")
    st.markdown("**Made with ❤️ by Team Agrinova**")




