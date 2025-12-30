import pandas as pd
import streamlit as st
import rasterio
import numpy as np
import base64
import joblib
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# ---------------------- 🎨 UI CONFIGURATION ----------------------
def set_bg_hack(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        font-family: 'Segoe UI', sans-serif;
        background-color: transparent;
    }}

    .title-box {{
        border: 3px solid white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background-color: rgba(0, 0, 0, 0.4);
        margin-top: 10px;
    }}
    .title-box h1 {{
        color: white !important;
        font-weight: bold;
        text-shadow: 1px 1px 4px black;
    }}

    .stSelectbox > div, .stFileUploader {{
        border: 2px solid white !important;
        border-radius: 10px;
        padding: 10px;
        background-color: rgba(0, 0, 0, 0.6);
        color: white !important;
    }}
    div[role="listbox"] > div {{
        color: black !important;
        background-color: white;
    }}
    .stFileUploader .css-1umxb2b {{
        color: white !important;
    }}

    .prediction-heading {{
        color: #fffb00;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
        text-shadow: 2px 2px 5px black;
    }}

    .stAlert, .stAlert p {{
        color: white !important;
        font-weight: bold;
    }}

    header [data-testid="stToolbar"] *, 
    header [data-testid="stStatusWidget"] * {{
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
    """, unsafe_allow_html=True)

set_bg_hack("deforestation image.png")

# ----------------------  MODEL SELECTION ----------------------
st.markdown("""
    <div class='title-box'>
        <h1> Deforestation Detection Dashboard</h1>
    </div>
""", unsafe_allow_html=True)

model_choice = st.selectbox(
    "Choose a Model for Prediction:",
    ["Random Forest", "SVM", "XGBoost", "Logistic Regression"]
)

uploaded_file = st.file_uploader(
    "Upload NDVI .tif file (3 bands: NDVI_before, NDVI_after, NDVI_diff)",
    type=["tif", "tiff"]
)
# ---------------------- FOREST INFO BANK ----------------------
forest_info = {
    "sundarban": (
        "The Sundarbans is the largest mangrove forest in the world, spanning approximately 10,000 square kilometers across India and Bangladesh. "
        "It is famous for its salt-tolerant mangrove trees such as Sundari (Heritiera fomes), Goran, Keora, and Gewa. "
        "These mangroves form dense thickets along tidal waterways, creating a unique and dynamic ecosystem. "
        "The forest is home to the iconic Royal Bengal Tiger, saltwater crocodiles, spotted deer, and hundreds of bird species. "
        "It acts as a crucial buffer protecting coastal areas from cyclones and erosion, and supports millions of livelihoods through fishing, honey collection, and tourism."
    ),

    "corbett": (
        "Jim Corbett National Park, located in Uttarakhand, India, covers an area of around 1,318 square kilometers. "
        "The forest is rich in Sal (Shorea robusta), Rohini, Haldu, and various species of bamboo. "
        "The lower regions are dominated by grasslands (chaurs), which support a high density of herbivores. "
        "Corbett is the oldest national park in India and a key tiger reserve, with diverse fauna including elephants, leopards, deer, and over 600 bird species. "
        "Its varied terrain — riverine belts, hills, grasslands, and dense forests — makes it a prime ecotourism destination."
    ),

    "satpura": (
        "Satpura Forest, part of the Satpura Range in central India, covers approximately 2,200 square kilometers within Satpura National Park. "
        "It is characterized by mixed deciduous forests dominated by Teak (Tectona grandis), Sal, Mahua, and various bamboo species. "
        "The rugged terrain includes sandstone peaks, deep valleys, and gorges, providing habitat for tigers, leopards, sloth bears, Indian bison (gaur), and numerous endemic birds and reptiles. "
        "Satpura offers a pristine, less crowded experience for wildlife enthusiasts seeking deeper connections with wilderness."
    ),

    "gir": (
        "Gir Forest National Park, located in Gujarat, India, spreads across roughly 1,412 square kilometers. "
        "The forest is mainly composed of dry deciduous trees like Teak, Flame of the Forest (Butea monosperma), Acacia, and Zizyphus species. "
        "Scrublands and patches of savannah grasslands also feature prominently. "
        "Gir is the only habitat of the Asiatic lion and supports leopards, hyenas, sambar deer, chital, and many bird and reptile species. "
        "It represents one of India’s great conservation success stories."
    ),

    "western":(
        "The Western Ghats, stretching over 1,600 kilometers along India's western coast, cover around 140,000 square kilometers. "
        "They host tropical evergreen forests, semi-evergreen forests, moist deciduous forests, and shola-grassland ecosystems. "
        "Important tree species include Rosewood, Mahogany, Teak, Myristica (wild nutmeg), and a wide variety of endemic orchids and medicinal plants. "
        "The Ghats support over 5,000 flowering plant species and numerous endemic amphibians and reptiles, playing a vital role in water security and monsoon regulation."
    ),

    "gachibowli": (
        "The Gachibowli Forest is a protected urban forest patch in Hyderabad, Telangana, covering about 150 acres. "
        "It mainly consists of native dry deciduous trees such as Neem, Banyan, Tamarind, and various Acacia species. "
        "This green patch provides habitat to peacocks, hares, small mammals, reptiles, and a variety of urban-adapted bird species. "
        "Besides maintaining local biodiversity, it acts as a crucial ecological buffer, reducing urban heat and improving air quality."
    ),

    "kaziranga": (
        "Kaziranga National Park, located in Assam, India, spans around 1,090 square kilometers along the Brahmaputra floodplains. "
        "Its vegetation includes tall elephant grasses, marshland vegetation, and semi-evergreen and deciduous forests. "
        "Important tree species include Silk Cotton (Bombax ceiba), Indian Gooseberry, and various figs and cane species. "
        "Kaziranga is renowned for hosting two-thirds of the world's one-horned rhinoceros, as well as elephants, tigers, swamp deer, wild buffalo, and over 480 bird species. "
        "The park's mosaic of grasslands and wetlands makes it one of Asia’s richest biodiversity hotspots."
    )
}

if uploaded_file:
    # Extract filename without extension
    filename = os.path.splitext(uploaded_file.name)[0].lower()

    # Try to match with forest_info keys
    forest_key = None
    for key in forest_info.keys():
        if key in filename:
            forest_key = key
            break

    if forest_key:
        st.markdown(f"""
        <div style='background-color: rgba(0, 0, 0, 0.5); padding: 15px; border-radius: 10px; color: white; margin-top: 1rem; border-left: 5px solid green;'>
        <b>🌳 Forest Information:</b><br>
        {forest_info[forest_key]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Forest information could not be determined from the file name.")

@st.cache_resource
def load_models():
    models = {
        "RandomForest": joblib.load("EarthEngine/RandomForest.pkl"),
        "SVM": joblib.load("EarthEngine/svm_rbf_smote.pkl"),
        "XGBoost": joblib.load("EarthEngine/gb_model.pkl"),
        "LogisticRegression": joblib.load("EarthEngine/LogisticRegression.pkl")
    }
    return models
models = load_models() 



model_key_map = {
    "Random Forest": "RandomForest",
    "SVM": "SVM",
    "XGBoost": "XGBoost",
    "Logistic Regression": "LogisticRegression"
}

# ---------------------- 🔍 PREDICTION ----------------------
if uploaded_file:
    with rasterio.open(uploaded_file) as src:
        data = src.read()
        height, width = data.shape[1:]

    ndvi_before, ndvi_after, ndvi_diff = data[0], data[1], data[2]

    with st.spinner("🔍 Processing... please wait..."):
        if model_choice != "UNET":
            X = np.stack([ndvi_before, ndvi_after, ndvi_diff], axis=-1).reshape(-1, 3)
            valid_mask = np.all(~np.isnan(X), axis=1)

            prediction_flat = np.zeros(X.shape[0], dtype=np.uint8)
            selected_model = models[model_key_map[model_choice]]

            if np.sum(valid_mask) > 0:
                prediction_flat[valid_mask] = selected_model.predict(X[valid_mask])

            prediction_map = prediction_flat.reshape((height, width))

        else:
            resized_input = cv2.resize(
                np.stack([ndvi_before, ndvi_after, ndvi_diff], axis=-1),
                (256, 256),
                interpolation=cv2.INTER_LINEAR
            ).astype(np.float32) / 255.0

            pred_mask = unet_model.predict(np.expand_dims(resized_input, axis=0))[0, :, :, 0]
            pred_resized = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            prediction_map = (pred_resized > 0.5).astype(np.uint8)


    ndvi_gray = (255 * (ndvi_before - np.min(ndvi_before)) / (np.max(ndvi_before) - np.min(ndvi_before))).astype(np.uint8)
    rgb = np.stack([ndvi_gray] * 3, axis=-1)

    overlay = np.zeros_like(rgb)
    overlay[..., 0] = 255

    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[prediction_map == 1] = 150

    blended = rgb.astype(np.float32)
    for c in range(3):
        blended[..., c] = (1 - alpha / 255) * blended[..., c] + (alpha / 255) * overlay[..., c]
    blended = blended.astype(np.uint8)

    st.markdown(f"<div class='prediction-heading'>🧭 Prediction using {model_choice}</div>", unsafe_allow_html=True)
    st.image(blended, caption="🟥 Red = Predicted Deforested Area", use_container_width=True)

    total_pixels = prediction_map.size
    deforested_pixels = int(np.sum(prediction_map == 1))
    percent = (deforested_pixels / total_pixels) * 100

    st.success(f"🧮 **Deforested Pixels:** {deforested_pixels}")
    st.info(f"🌐 **Deforestation Percentage:** {percent:.2f}%")

    fig, ax = plt.subplots()
    sizes = [100 - percent, percent]
    labels = ['Forest Before', 'Deforested']
    colors = ['green', 'red']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', textprops={'color': 'white'})
    ax.set_title('Forest Coverage Change', color='white')
    st.pyplot(fig)

    st.markdown(f"""
        <div style='background-color: rgba(0, 0, 0, 0.5); padding: 15px; border-radius: 10px; color: white;'>
        <b>🔍 Change Summary:</b><br>
        The forest once had full vegetation coverage. After deforestation, it lost {deforested_pixels} pixels,
        which is approximately {percent:.2f}% of the area. The visible red regions indicate significant
        environmental degradation in this region.
        </div>
    """, unsafe_allow_html=True)



# ---------------------- 🌱 SMART REPLANTING SECTION ----------------------

# 🌳 Add space and divider after deforestation section
st.markdown("""<br><hr style='border-top: 3px solid #4CAF50; margin-top: 40px; margin-bottom: 40px;'>""", unsafe_allow_html=True)

st.markdown("""
<div class='title-box' style='background-color:rgba(0, 0, 0, 0.5); padding:15px; border-radius:10px; margin-bottom:10px; font-family: 'Segoe UI', sans-serif;
            border: 3px solid white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;'>
    <h1 style='color:white;'>🌱 Smart Replanting Recommendation</h1>
</div>
""", unsafe_allow_html=True)


with st.container():
    # 🔘 Model selection box
    replant_model = st.selectbox(
        "🔍 Choose Smart Replanting Model:",
        ["Decision Tree", "Naive Bayes", "Gradient Boosting", "MLP"]
    )

    forest_env = {
        "sundarban": {"Region": "Sundarban", "Soil_Type": "Saline Loam", "Rainfall": 1850, "Temperature": 25.5},
        "kaziranga": {"Region": "Kaziranga", "Soil_Type": "Peaty Soil", "Rainfall": 2050, "Temperature": 27.5},
        "gir": {"Region": "Gir", "Soil_Type": "Dry Clay", "Rainfall": 920, "Temperature": 28.0},
        "satpura": {"Region": "Satpura", "Soil_Type": "Alluvial", "Rainfall": 1180, "Temperature": 24.0},
        "corbett": {"Region": "Corbett", "Soil_Type": "Alluvial", "Rainfall": 1250, "Temperature": 23.5},
        "western": {"Region": "Western", "Soil_Type": "Laterite", "Rainfall": 2200, "Temperature": 23.0},
        "gachibowli": {"Region": "Gachibowli", "Soil_Type": "Red Sandy", "Rainfall": 850, "Temperature": 30.0}
    }

    class_tree_map = {
        "ShadeTree": ["Gulmohar", "Banyan", "Indian Almond"],
        "FastGrower": ["Subabul", "Eucalyptus", "Bamboo"],
        "DroughtResistant": ["Neem", "Tamarind", "Zizyphus"],
        "Hardwood": ["Teak", "Haldu", "Sal"],
        "Mangrove": ["Sundari", "Keora", "Goran"],
        "Medicinal": ["Mahua", "Figs", "Wild Nutmeg"]
    }

    if uploaded_file and forest_key:
        region = forest_key
        env = forest_env[region]
        ndvi_avg = float(np.mean(ndvi_before))
        
        st.markdown(
    """
    <style>
    .white-box {
        background-color: white;
        color: black;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="white-box">
            <h4>📍 Insights on <b>{region.capitalize()}</b></h4>
            <ul>
                <li><b>Soil Type:</b> {env['Soil_Type']}</li>
                <li><b>Avg Rainfall:</b> {env['Rainfall']} mm</li>
                <li><b>Avg Temperature:</b> {env['Temperature']}°C</li>
                <li><b>NDVI:</b> {ndvi_avg:.2f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        model_paths = {
            "Decision Tree": "Smart replanting algo/decision_tree_model.pkl",
            "Naive Bayes": "Smart replanting algo/naive_bayes_model.pkl",
            "Gradient Boosting": "Smart replanting algo/xgb_model.pkl",
            "MLP": "Smart replanting algo/ann_model.keras"
        }

        if replant_model == "MLP":
            from tensorflow.keras.models import load_model
            model = load_model(model_paths[replant_model])
        else:
            model = joblib.load(model_paths[replant_model])

        scaler_paths = {
            "Decision Tree": "Smart replanting algo/dtree_scaler.pkl",
            "Naive Bayes": "Smart replanting algo/naive_bayes_scaler.pkl",
            "Gradient Boosting": "Smart replanting algo/xgb_scaler.pkl",
            "MLP": "Smart replanting algo/ann_scaler.pkl"
        }

        encoder_paths = {
            "Decision Tree": "Smart replanting algo/dtree_encoder.pkl",
            "Naive Bayes": "Smart replanting algo/naive_bayes_label_encoder.pkl",
            "Gradient Boosting": "Smart replanting algo/xgb_label_encoder.pkl",
            "MLP": "Smart replanting algo/ann_label_encoder.pkl"
        }

        feature_paths = {
            "Decision Tree": "Smart replanting algo/dtree_feature_columns.pkl",
            "Naive Bayes": "Smart replanting algo/naive_bayes_feature_columns.pkl",
            "Gradient Boosting": "Smart replanting algo/xgb_feature_columns.pkl",
            "MLP": "Smart replanting algo/ann_feature_columns.pkl"
        }

        scaler = joblib.load(scaler_paths[replant_model])
        label_encoder = joblib.load(encoder_paths[replant_model])
        feature_columns = joblib.load(feature_paths[replant_model])

        input_dict = {
            "NDVI": ndvi_avg,
            "Rainfall": env["Rainfall"],
            "Temperature": env["Temperature"],
            f"Region_{env['Region']}": 1,
            f"Soil_Type_{env['Soil_Type']}": 1
        }

        input_df = pd.DataFrame([0] * len(feature_columns), index=feature_columns).T
        for k, v in input_dict.items():
            if k in input_df.columns:
                input_df[k] = v

        input_scaled = scaler.transform(input_df)

        if replant_model == "MLP":
            probs = model.predict(input_scaled)[0]
        else:
            probs = model.predict_proba(input_scaled)[0]

        top2_idx = probs.argsort()[-2:][::-1]
        top2_classes = label_encoder.inverse_transform(top2_idx)
        top2_scores = probs[top2_idx]
        
        st.markdown(
    """
    <style>
    .white-box {
        background-color: white;
        color: black;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True)
        st.markdown("""
                     <div class="white-box">
                    <h4 style='color:white;'>🌿 Suggested Tree Classes</h4>
                    <ul style="font-size:16px;">
                    <li><b>{main}</b> — {main_list}</li>
                    <li><b>{alt}</b> (Alternate) — {alt_list}</li>
                    </ul>
                    <p><b>📊 Model Confidence:</b> {conf:.1f}%</p>
                    </div>
                    """.format(
                        main=top2_classes[0],
                        main_list=', '.join(class_tree_map[top2_classes[0]]),
                        alt=top2_classes[1],
                        alt_list=', '.join(class_tree_map[top2_classes[1]]),
                        conf=top2_scores[0]*100
                        ), unsafe_allow_html=True)
        st.progress(float(top2_scores[0]))

# ✔️ Only show chart if model has been loaded and prediction is done
        # 📊 Optional horizontal bar chart of confidence
        fig, ax = plt.subplots(figsize=(8, 4))
        classes = label_encoder.classes_
        probs_sorted_idx = np.argsort(probs)
        ax.barh(range(len(probs)), probs[probs_sorted_idx], color='seagreen')
        ax.set_yticks(range(len(probs)))
        ax.set_yticklabels(classes[probs_sorted_idx])
        ax.set_xlabel("Probability")
        ax.set_title("🌲 Model Confidence per Tree Class")
        st.pyplot(fig)
