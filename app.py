import json
import io
import re
import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "parkinsons_model.joblib"
INFO_PATH = BASE_DIR / "models" / "model_info.json"
DATA_PATH = BASE_DIR / "data" / "parkinsons.data"
RESULT_PATH = BASE_DIR / "results" / "model_comparison.csv"
CM_PATH = BASE_DIR / "results" / "confusion_matrix.png"

VOICE_MODEL_PATH = BASE_DIR / "models" / "voice_model.joblib"
VOICE_INFO_PATH = BASE_DIR / "models" / "voice_model_info.json"
AUDIO_FEATURES_PATH = BASE_DIR / "data" / "audio_features.csv"
AUDIO_RESULTS_PATH = BASE_DIR / "results" / "audio_model_comparison.csv"
AUDIO_CM_PATH = BASE_DIR / "results" / "audio_confusion_matrix.csv"


st.set_page_config(
    page_title="Parkinson AI",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# CLINICAL WHITE UI THEME (visual styling only)
# --------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --primary: #1677b8;
        --teal: #0f9d9a;
        --ink: #17324d;
        --muted: #62758a;
        --line: #dce7f0;
        --soft: #f4f8fb;
    }
    .stApp { background: #f7fafc; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247,250,252,0.92); }
    [data-testid="stAppViewContainer"] > .main { background: #f7fafc; }
    .block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink) !important; letter-spacing: -0.025em; }
    h1 { font-size: clamp(1.8rem, 3vw, 2.5rem) !important; font-weight: 750 !important; }
    h2 { font-size: 1.45rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.12rem !important; font-weight: 650 !important; }
    p, label, li { color: #344b61; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid var(--line);
        border-radius: 16px; padding: 18px 20px; box-shadow: 0 4px 16px rgba(20,55,85,.045); }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--primary) !important; font-weight: 750; }
    [data-testid="stTabs"] [role="tablist"] { gap: 8px; border-bottom: 1px solid var(--line); }
    [data-testid="stTabs"] button[role="tab"] { border-radius: 10px 10px 0 0; padding: 12px 16px;
        color: #486176; font-weight: 650; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--primary); border-bottom: 3px solid var(--primary); background: #eaf4fb; }
    .stButton > button, [data-testid="stDownloadButton"] button { border-radius: 10px;
        border: 1px solid var(--primary); background: var(--primary); color: white;
        font-weight: 650; padding: .55rem 1rem; transition: all .15s ease; }
    .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
        background: #105f94; border-color: #105f94; color: white; }
    [data-testid="stFileUploader"] { border: 1px dashed #9bbbd1; border-radius: 14px;
        background: #ffffff; padding: 10px; }
    [data-testid="stAlert"] { border-radius: 12px; border-width: 1px; }
    [data-testid="stExpander"] { background: white; border: 1px solid var(--line); border-radius: 12px; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    hr { border-color: var(--line); margin: 1.4rem 0; }
    code { color: #0b668b; background: #eaf4f8; border-radius: 5px; }
    footer { visibility: hidden; }
    @media (max-width: 700px) { .block-container { padding: 1rem 1rem 2rem; } }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_voice_model():
    return joblib.load(VOICE_MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_info():
    with open(INFO_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_audio_feature_columns():
    audio_df = pd.read_csv(AUDIO_FEATURES_PATH, nrows=1)

    label_columns = {
        "label",
        "status",
        "target",
        "class",
        "diagnosis"
    }

    feature_columns = [
        column for column in audio_df.columns
        if column.lower() not in label_columns
    ]

    return feature_columns


model = load_model()
df = load_data()
info = load_info()

features = info["features"]

X = df[features]
y = df["status"]

_, test_indices = train_test_split(
    df.index,
    test_size=0.2,
    random_state=42,
    stratify=y
)

test_df = df.loc[test_indices].copy()


st.markdown(
    '<div style="display:inline-block;padding:5px 11px;border-radius:999px;'
    'background:#e6f4f7;color:#087e8b;font-size:0.78rem;font-weight:750;'
    'letter-spacing:.07em;text-transform:uppercase;margin-bottom:8px;">'
    'VOICE BIOMARKER ANALYTICS · MINI PROJECT</div>',
    unsafe_allow_html=True
)

st.title("🧠 Parkinson's Disease Prediction System")

st.markdown(
    "### Machine Learning Based Voice Biomarker Analysis"
)

st.warning(
    "Educational project only. This system is not a validated "
    "medical diagnostic tool. Do not use its predictions to make "
    "healthcare decisions."
)

st.markdown("---")


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("UCI Dataset Records", len(df))

with col2:
    st.metric("UCI Voice Features", len(features))

with col3:
    st.metric("ML Models Compared", 3)


st.markdown("---")


tab1, tab2, tab3, tab4 = st.tabs([
    "🎤 Voice Upload",
    "🔍 Feature Prediction",
    "📊 Model Comparison",
    "📈 Evaluation"
])


# --------------------------------------------------
# TAB 1: VOICE UPLOAD PREDICTION
# --------------------------------------------------

with tab1:

    st.header("🎤 Parkinson's Prediction from Voice")

    st.write(
        "Upload a WAV voice recording. The system extracts "
        "MFCC features and passes them to the separately "
        "trained audio classification model."
    )

    st.info(
        "This model was trained on the Healthy and Parkinsons "
        "folders of your audio dataset. Results depend on the "
        "quality and source of the recording."
    )

    if not VOICE_MODEL_PATH.exists():
        st.error(
            "Voice model not found. Run src/train_audio_model.py first."
        )

    elif not AUDIO_FEATURES_PATH.exists():
        st.error(
            "Audio feature dataset not found at data/audio_features.csv."
        )

    else:

        uploaded_audio = st.file_uploader(
            "Upload a WAV audio file",
            type=["wav"],
            key="voice_upload"
        )

        if uploaded_audio is not None:

            st.audio(
                uploaded_audio.getvalue(),
                format="audio/wav"
            )

            if st.button(
                "Analyze Voice",
                use_container_width=True
            ):

                try:

                    audio_bytes = uploaded_audio.getvalue()

                    audio, sample_rate = librosa.load(
                        io.BytesIO(audio_bytes),
                        sr=16000,
                        mono=True
                    )

                    if len(audio) == 0:
                        st.error("The uploaded audio file is empty.")
                        st.stop()

                    if not np.all(np.isfinite(audio)):
                        st.error(
                            "The audio contains invalid numerical values."
                        )
                        st.stop()

                    mfcc = librosa.feature.mfcc(
                        y=audio,
                        sr=sample_rate,
                        n_mfcc=20
                    )

                    mfcc_means = np.mean(mfcc, axis=1)
                    mfcc_stds = np.std(mfcc, axis=1)

                    feature_values = {}

                    for i in range(20):
                        feature_values[f"mfcc_{i + 1}_mean"] = (
                            float(mfcc_means[i])
                        )

                    for i in range(20):
                        feature_values[f"mfcc_{i + 1}_std"] = (
                            float(mfcc_stds[i])
                        )

                    audio_feature_columns = load_audio_feature_columns()

                    missing_features = [
                        column
                        for column in audio_feature_columns
                        if column not in feature_values
                    ]

                    if missing_features:
                        st.error(
                            "Feature names in audio_features.csv do not "
                            "match the app's MFCC feature names. "
                            "Check the column names in the CSV and "
                            "src/extract_audio_features.py."
                        )

                        st.write(
                            "Expected CSV columns:",
                            audio_feature_columns
                        )

                        st.stop()

                    voice_input = pd.DataFrame(
                        [[feature_values[column]
                          for column in audio_feature_columns]],
                        columns=audio_feature_columns
                    )

                    voice_model = load_voice_model()

                    prediction = int(
                        voice_model.predict(voice_input)[0]
                    )

                    st.subheader("Voice Analysis Result")

                    if prediction == 1:
                        st.error(
                            "Model Prediction: Parkinson's Class"
                        )
                    elif prediction == 0:
                        st.success(
                            "Model Prediction: Healthy Class"
                        )
                    else:
                        st.warning(
                            f"Model returned an unexpected label: {prediction}"
                        )

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.metric(
                            "Audio Duration",
                            f"{len(audio) / sample_rate:.2f} seconds"
                        )

                    with col_b:
                        st.metric(
                            "Processing Sample Rate",
                            f"{sample_rate} Hz"
                        )

                    st.caption(
                        "The output is a machine-learning classification, "
                        "not a medical diagnosis. Recording conditions, "
                        "speaker overlap in the dataset, and other "
                        "limitations can affect performance."
                    )

                except Exception as error:

                    st.error(
                        "Unable to analyze this recording. "
                        "Please try a valid WAV audio file."
                    )

                    st.exception(error)


# --------------------------------------------------
# TAB 2: EXISTING FEATURE-BASED PREDICTION
# --------------------------------------------------

with tab2:

    st.header("Voice Biomarker Prediction")

    mode = st.radio(
        "Choose input method",
        [
            "Select Dataset Sample",
            "Enter Features Manually"
        ],
        horizontal=True
    )

    input_values = None
    actual_status = None

    if mode == "Select Dataset Sample":

        st.write(
            "Select a sample from the held-out test set "
            "to demonstrate the model's prediction."
        )

        sample_numbers = list(range(len(test_df)))

        selected_number = st.selectbox(
            "Select Sample",
            sample_numbers,
            format_func=lambda n: f"Test Sample {n + 1}"
        )

        selected_row = test_df.iloc[selected_number]

        input_values = selected_row[features].to_dict()
        actual_status = int(selected_row["status"])

        with st.expander("View selected sample's voice features"):

            st.dataframe(
                pd.DataFrame([input_values]),
                use_container_width=True
            )

        predict_button = st.button(
            "Predict Selected Sample",
            use_container_width=True
        )

    else:

        st.write(
            "Enter numerical voice biomarkers obtained from "
            "a compatible dataset."
        )

        with st.form("manual_prediction_form"):

            input_values = {}
            columns = st.columns(3)

            for i, feature in enumerate(features):

                with columns[i % 3]:

                    input_values[feature] = st.number_input(
                        feature,
                        value=0.0,
                        format="%.6f"
                    )

            predict_button = st.form_submit_button(
                "Predict",
                use_container_width=True
            )

    if predict_button and input_values is not None:

        input_df = pd.DataFrame(
            [input_values],
            columns=features
        )

        prediction = int(model.predict(input_df)[0])

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "Model Prediction: Parkinson's Class (1)"
            )

        else:

            st.success(
                "Model Prediction: Control Class (0)"
            )

        if mode == "Select Dataset Sample":

            st.info(
                "Reference label: "
                + (
                    "Parkinson's Class (1)"
                    if actual_status == 1
                    else "Control Class (0)"
                )
            )

            if prediction == actual_status:
                st.success(
                    "Prediction matches the dataset label."
                )

            else:
                st.warning(
                    "Prediction differs from the dataset label."
                )

        st.caption(
            "Dataset labels are references for this demonstration, "
            "not substitutes for clinical assessment. Manual "
            "predictions using arbitrary values are not meaningful."
        )


# --------------------------------------------------
# TAB 3: MODEL COMPARISON
# --------------------------------------------------

with tab3:

    st.header("Machine Learning Model Comparison")

    st.subheader("UCI Voice Feature Models")

    if RESULT_PATH.exists():

        comparison = pd.read_csv(RESULT_PATH)

        st.dataframe(
            comparison.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}"
            }),
            use_container_width=True
        )

        if "Accuracy" in comparison.columns:

            st.subheader("Accuracy Comparison")

            st.bar_chart(
                comparison.set_index("Model")[["Accuracy"]]
            )

        if "F1 Score" in comparison.columns:

            st.subheader("F1 Score Comparison")

            st.bar_chart(
                comparison.set_index("Model")[["F1 Score"]]
            )

        if (
            "Model" in comparison.columns
            and "F1 Score" in comparison.columns
            and not comparison.empty
        ):

            best_row = comparison.loc[
                comparison["F1 Score"].idxmax()
            ]

            st.success(
                f"Best model in this comparison: "
                f"{best_row['Model']} "
                f"(F1 Score: {best_row['F1 Score']:.2%})"
            )

    else:

        st.info("UCI model comparison file not found.")

    st.markdown("---")

    st.subheader("Raw Audio Models")

    if AUDIO_RESULTS_PATH.exists():

        audio_comparison = pd.read_csv(AUDIO_RESULTS_PATH)

        st.dataframe(
            audio_comparison.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}"
            }),
            use_container_width=True
        )

        if (
            "Model" in audio_comparison.columns
            and "Accuracy" in audio_comparison.columns
        ):

            st.subheader("Audio Model Accuracy")

            st.bar_chart(
                audio_comparison.set_index("Model")[["Accuracy"]]
            )

        if (
            "Model" in audio_comparison.columns
            and "F1 Score" in audio_comparison.columns
            and not audio_comparison.empty
        ):

            best_audio = audio_comparison.loc[
                audio_comparison["F1 Score"].idxmax()
            ]

            st.success(
                f"Best audio model in this comparison: "
                f"{best_audio['Model']} "
                f"(F1 Score: {best_audio['F1 Score']:.2%})"
            )

    else:

        st.info("Audio model comparison file not found.")


# --------------------------------------------------
# TAB 4: EVALUATION
# --------------------------------------------------

with tab4:

    st.header("Model Evaluation")

    st.subheader("UCI Model Confusion Matrix")

    if CM_PATH.exists():

        st.image(
            str(CM_PATH),
            caption="Random Forest confusion matrix",
            use_container_width=True
        )

    else:

        st.info("UCI confusion matrix image not found.")

    st.markdown("### UCI Dataset Overview")

    status_counts = df["status"].value_counts().sort_index()

    status_summary = pd.DataFrame({
        "Class": [
            "Control (0)",
            "Parkinson's (1)"
        ],
        "Records": [
            int(status_counts.get(0, 0)),
            int(status_counts.get(1, 0))
        ]
    })

    st.dataframe(
        status_summary,
        use_container_width=True
    )

    st.bar_chart(
        status_summary.set_index("Class")
    )

    st.markdown("---")

    st.subheader("Audio Model Evaluation")

    if AUDIO_CM_PATH.exists():

        try:

            audio_cm = pd.read_csv(
                AUDIO_CM_PATH,
                index_col=0
            )

            st.dataframe(
                audio_cm,
                use_container_width=True
            )

        except Exception:

            st.info(
                "Audio confusion matrix exists but could not be displayed."
            )

    else:

        st.info("Audio confusion matrix file not found.")

    st.markdown("### Important Limitations")

    st.write(
        "- The UCI dataset is relatively small.\n"
        "- Both models use a particular train/test split.\n"
        "- Multiple recordings from the same participant can inflate "
        "performance if they occur in both training and test sets.\n"
        "- Participant-wise evaluation is needed to better estimate "
        "performance on unseen people.\n"
        "- The UCI model uses extracted numerical voice features.\n"
        "- The audio model uses MFCC statistics extracted from WAV files.\n"
        "- Neither model is clinically validated for diagnosing Parkinson's."
    )


st.markdown("---")

st.caption(
    "B.Tech Mini Project | Machine Learning | Educational Use Only"
)