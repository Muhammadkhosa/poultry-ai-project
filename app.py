
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
weight_model  = joblib.load('weight_model.pkl')
egg_model     = joblib.load('egg_model.pkl')
breed_encoder = joblib.load('breed_encoder.pkl')

# App config
st.set_page_config(page_title="Smart Poultry Farm AI", page_icon="🐔")

st.title("🐔 AI Smart Poultry Farm Management System")
st.markdown("**Beaconhouse National University | CSC-233 AI Lab | Spring 2026**")
st.markdown("---")

# Two tabs
tab1, tab2 = st.tabs(["🐔 Chicken Weight Predictor", "🥚 Egg Production Predictor"])

# ============================================================
# TAB 1: Weight Predictor
# ============================================================
with tab1:
    st.header("🐔 Chicken Weight Predictor")
    st.write("Predict chicken weight based on age and diet type.")
    st.markdown("---")

    # Diet Reference Table
    st.markdown("#### 🌾 Diet Type Reference Guide")
    diet_data = {
        "Diet": ["Diet 1", "Diet 2", "Diet 3", "Diet 4"],
        "Type": ["Control Diet", "Added Protein", "Mixed Supplement", "High Fat Diet"],
        "Description": [
            "Basic standard feed — no special additions",
            "Standard feed + extra protein supplement",
            "Standard feed + protein + vitamins (Best Growth)",
            "Standard feed + extra fat/energy"
        ],
        "Expected Growth": ["Slowest 🔴", "Moderate 🟡", "Fastest 🟢", "Good 🟡"]
    }
    st.table(pd.DataFrame(diet_data))
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        time = st.number_input(
            "📅 Chicken Age (Days)",
            min_value=0, max_value=21,
            value=10, step=1
        )
    with col2:
        diet = st.number_input(
            "🌾 Diet Type (1, 2, 3 or 4)",
            min_value=1, max_value=4,
            value=1, step=1
        )

    if st.button("🔍 Predict Weight", use_container_width=True):
        input_data = pd.DataFrame({"Time": [time], "Diet": [diet]})
        prediction = weight_model.predict(input_data)[0]

        st.success(f"### ⚖️ Predicted Weight: {prediction:.1f} grams")

        if prediction < 100:
            st.info("🐣 Chick is still young and growing.")
        elif prediction < 200:
            st.info("🐔 Chick is in mid-growth stage.")
        else:
            st.info("🦃 Chicken has reached a healthy mature weight!")

        st.markdown("---")
        st.markdown("**Input Summary:**")
        st.write(f"- Age: {time} days")
        st.write(f"- Diet: Diet {diet}")
        st.write(f"- Predicted Weight: {prediction:.1f}g")

# ============================================================
# TAB 2: Egg Production Predictor
# ============================================================
with tab2:
    st.header("🥚 Egg Production Predictor")
    st.write("Predict daily egg production based on breed, age and feed.")
    st.markdown("---")

    # Breed Reference Table
    st.markdown("#### 🐓 Breed Reference Guide")
    breed_data = {
        "Breed": ["Marans", "Ameraucana"],
        "Egg Color": ["Dark Brown", "Blue/Green"],
        "Avg Eggs/Day": ["3.5 - 5.5", "2.5 - 4.5"],
        "Best Age (Days)": ["150 - 400", "150 - 400"],
        "Production Level": ["High 🟢", "Moderate 🟡"]
    }
    st.table(pd.DataFrame(breed_data))
    st.markdown("---")

    col3, col4, col5 = st.columns(3)
    with col3:
        breed = st.selectbox("🐓 Breed", options=["Marans", "Ameraucana"])
    with col4:
        age = st.number_input(
            "📅 Chicken Age (Days)",
            min_value=24, max_value=990,
            value=300, step=1
        )
    with col5:
        feed = st.number_input(
            "🌾 Feed Amount (grams)",
            min_value=100, max_value=129,
            value=115, step=1
        )

    if st.button("🔍 Predict Egg Production", use_container_width=True):
        breed_encoded = breed_encoder.transform([breed])[0]
        input_data = pd.DataFrame({
            "Breed_encoded": [breed_encoded],
            "Age"          : [age],
            "AmountOfFeed" : [feed]
        })
        prediction = egg_model.predict(input_data)[0]
        prediction = max(0, prediction)

        st.success(f"### 🥚 Predicted Eggs Per Day: {prediction:.2f}")

        if prediction < 2:
            st.info("📉 Low production — check age and feed.")
        elif prediction < 4:
            st.info("📊 Average production for this breed.")
        else:
            st.info("📈 Excellent egg production!")

        st.markdown("---")
        st.markdown("**Input Summary:**")
        st.write(f"- Breed: {breed}")
        st.write(f"- Age: {age} days")
        st.write(f"- Feed: {feed}g")
        st.write(f"- Predicted Eggs/Day: {prediction:.2f}")

st.markdown("---")
st.markdown("*Developed by Group | BNU | Spring 2026*")
