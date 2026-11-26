import pandas as pd
import streamlit as st
import plotly.express as px

from database import init_db, save_prediction, get_prediction_history, get_dataset_summary
from train_model import ensure_dataset, train_or_load_model, FEATURES, TARGET, DATA_DIR

st.set_page_config(page_title="Wine Quality Chemistry Explorer & Grader", page_icon="🧪", layout="wide")

st.title("🧪 Wine Quality Chemistry Explorer & Grader")
st.caption("Explore red and white wine chemistry, predict an estimated quality score, and review saved predictions.")

@st.cache_resource
def setup_project():
    init_db()
    data = ensure_dataset()
    model, metrics = train_or_load_model(data)
    return data, model, metrics

try:
    data, model, metrics = setup_project()
except Exception as exc:
    st.error("The app could not prepare its dataset or model.")
    st.write("Check your internet connection on first run, then try again.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("About")
    st.write("This educational project uses the UCI Wine Quality dataset and a Random Forest regression model.")
    st.write("Predictions are estimates, not a professional or laboratory assessment.")
    st.divider()
    st.metric("Dataset records", f"{len(data):,}")
    st.metric("Model test MAE", f"{metrics['mae']:.2f}")
    st.metric("Model test R²", f"{metrics['r2']:.3f}")

tab_predict, tab_explore, tab_history, tab_about = st.tabs(
    ["🔬 Predict quality", "📊 Explore dataset", "🗂️ Prediction history", "ℹ️ Project details"]
)

with tab_predict:
    st.subheader("Enter physicochemical measurements")
    wine_type = st.radio("Wine color", ["Red", "White"], horizontal=True)
    st.info("Enter measurements in the units used by the dataset. The model needs all 11 chemical features.")

    defaults = {
        "fixed acidity": 7.0,
        "volatile acidity": 0.50,
        "citric acid": 0.25,
        "residual sugar": 2.5,
        "chlorides": 0.07,
        "free sulfur dioxide": 15.0,
        "total sulfur dioxide": 45.0,
        "density": 0.996,
        "pH": 3.30,
        "sulphates": 0.60,
        "alcohol": 10.5,
    }
    ranges = {
        "fixed acidity": (4.0, 16.0, 0.1),
        "volatile acidity": (0.1, 1.6, 0.01),
        "citric acid": (0.0, 1.7, 0.01),
        "residual sugar": (0.5, 20.0, 0.1),
        "chlorides": (0.01, 0.3, 0.001),
        "free sulfur dioxide": (1.0, 80.0, 1.0),
        "total sulfur dioxide": (6.0, 300.0, 1.0),
        "density": (0.990, 1.005, 0.0001),
        "pH": (2.7, 4.2, 0.01),
        "sulphates": (0.2, 2.0, 0.01),
        "alcohol": (8.0, 15.0, 0.1),
    }

    with st.form("prediction_form"):
        cols = st.columns(3)
        values = {}
        for idx, feature in enumerate(FEATURES):
            lo, hi, step = ranges[feature]
            with cols[idx % 3]:
                values[feature] = st.number_input(
                    feature.title(),
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(defaults[feature]),
                    step=float(step),
                    format="%.4f" if feature == "density" else "%.3f" if feature == "chlorides" else "%.2f",
                    help=f"Dataset feature: {feature}",
                )
        submitted = st.form_submit_button("Predict wine quality", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{feature: values[feature] for feature in FEATURES}])
        prediction = float(model.predict(row)[0])
        prediction = max(0.0, min(10.0, prediction))
        rounded_score = int(round(prediction))
        if rounded_score <= 4:
            grade = "Low"
        elif rounded_score == 5:
            grade = "Average"
        elif rounded_score == 6:
            grade = "Good"
        else:
            grade = "Very good"

        save_prediction(wine_type, values, prediction, rounded_score, grade)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted quality score", f"{prediction:.2f} / 10")
        c2.metric("Rounded score", str(rounded_score))
        c3.metric("Estimated grade", grade)
        st.caption("The grade labels are a simple project-defined interpretation of the predicted score, not official wine certification.")
        st.success("Prediction saved to the local SQLite database.")

with tab_explore:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", f"{len(data):,}")
    c2.metric("Red wine records", f"{int((data['wine_type'] == 'Red').sum()):,}")
    c3.metric("White wine records", f"{int((data['wine_type'] == 'White').sum()):,}")

    color_filter = st.multiselect("Filter wine color", ["Red", "White"], default=["Red", "White"])
    filtered = data[data["wine_type"].isin(color_filter)]
    if filtered.empty:
        st.warning("Select at least one wine color.")
    else:
        left, right = st.columns(2)
        with left:
            fig = px.histogram(filtered, x="quality", color="wine_type", barmode="overlay",
                               title="Recorded quality-score distribution",
                               labels={"quality": "Recorded quality score", "wine_type": "Wine color"})
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.box(filtered, x="wine_type", y="alcohol", color="wine_type",
                         title="Alcohol measurements by wine color",
                         labels={"wine_type": "Wine color", "alcohol": "Alcohol (% by volume)"})
            st.plotly_chart(fig, use_container_width=True)

        feature_choice = st.selectbox("Explore a chemical feature", FEATURES, index=FEATURES.index("alcohol"))
        fig = px.scatter(filtered.sample(min(1500, len(filtered)), random_state=42),
                         x=feature_choice, y="quality", color="wine_type",
                         opacity=0.65, title=f"{feature_choice.title()} vs recorded quality",
                         labels={"quality": "Recorded quality score", "wine_type": "Wine color"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)

with tab_history:
    st.subheader("Saved predictions")
    history = get_prediction_history()
    if history.empty:
        st.info("No predictions saved yet. Make a prediction in the first tab.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        fig = px.scatter(history, x="created_at", y="predicted_score", color="wine_type",
                         hover_data=["rounded_score", "grade"],
                         title="Prediction history over time",
                         labels={"created_at": "Time", "predicted_score": "Predicted score", "wine_type": "Wine color"})
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download prediction history as CSV",
                           data=history.to_csv(index=False).encode("utf-8"),
                           file_name="prediction_history.csv", mime="text/csv")

with tab_about:
    st.subheader("How the application works")
    st.markdown("""
    1. Downloads the public red and white Wine Quality datasets on first run.
    2. Combines the records and adds a `wine_type` field.
    3. Trains a Random Forest regression model using the 11 numeric chemistry features.
    4. Evaluates the model on a held-out test set using MAE and R².
    5. Saves user predictions to SQLite so they can be reviewed later.

    **Important:** The prediction model uses the chemistry features, not the `wine_type` label itself. Wine color is displayed and saved as metadata. The quality scores in the source dataset are human-assigned sensory scores; the model learns patterns from those historical records.
    """)
    st.write("Feature list:", ", ".join(FEATURES))
    st.write("Source: UCI Machine Learning Repository — Wine Quality dataset.")
    st.link_button("Open UCI dataset page", "https://archive.ics.uci.edu/dataset/186/wine+quality")

