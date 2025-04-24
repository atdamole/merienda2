import streamlit as st
import joblib
import numpy as np

# ✅ Cache model and encoder loading (runs once)
@st.cache_resource
def load_models():
    return {
        "flavor": joblib.load("dtc_model_flavor.pkl"),
        "topping": joblib.load("dtc_model_topping.pkl"),
        "drink": joblib.load("dtc_model_drink.pkl")
    }

@st.cache_resource
def load_encoders():
    return {
        "flavor": joblib.load("encoder_flavor.pkl"),
        "topping": joblib.load("encoder_topping.pkl"),
        "drink": joblib.load("encoder_drink.pkl"),
        "inputs": joblib.load("input_encoders.pkl")
    }

models = load_models()
encoders = load_encoders()

# 🔽 Dropdown options
input_encs = encoders["inputs"]
mood_list = input_encs["mood"].classes_.tolist()
weather_list = input_encs["weather"].classes_.tolist()
craving_list = input_encs["craving_level"].classes_.tolist()
last_meal_list = input_encs["last_meal"].classes_.tolist()
budget_list = input_encs["budget"].classes_.tolist()

# 🎨 UI
st.title("🥡 Merienda Matchmaker")
st.write("🎉 Hungry? Let’s match your cravings with the perfect pancit canton combo!\nPick your mood, weather, and vibe — we’ll do the rest. 🧠🍜")

# ⬇️ User inputs
mood = st.selectbox("🧠 Mood", mood_list)
weather = st.selectbox("🌦️ Weather", weather_list)
craving_level = st.selectbox("🔥 Craving Level", craving_list)
last_meal = st.selectbox("🍽️ Last Meal", last_meal_list)
budget = st.selectbox("💸 Budget", budget_list)

# ▶️ Predict button
if st.button("Get My Merienda Recommendation!"):
    features = [mood, weather, craving_level, last_meal, budget]
    encoded = [input_encs[col].transform([val])[0] for col, val in zip(input_encs.keys(), features)]
    encoded_np = np.array(encoded).reshape(1, -1)

    pred_flavor = encoders["flavor"].inverse_transform(models["flavor"].predict(encoded_np))[0]
    pred_topping = encoders["topping"].inverse_transform(models["topping"].predict(encoded_np))[0]
    pred_drink = encoders["drink"].inverse_transform(models["drink"].predict(encoded_np))[0]

    st.success(f"✨ Flavor Match: {pred_flavor}")
    st.success(f"🍳 Topping Pairing: {pred_topping}")
    st.success(f"🥤 Drink Suggestion: {pred_drink}")
