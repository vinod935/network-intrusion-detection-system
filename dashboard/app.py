
import sys
import io
from pathlib import Path
import difflib

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.predict import NIDSPredictor


# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NIDS | Security Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 3. COLORS
# =========================================================

NAVY = "#153E5C"
NAVY_DARK = "#0D2F49"
ORANGE = "#F5A000"
GREEN = "#19966B"
RED = "#E45B4F"
BACKGROUND = "#E9EEF5"
TEXT = "#202D3B"
MUTED = "#8996A5"


# =========================================================
# 4. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #E9EEF5;
    color: #202D3B;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

#MainMenu, footer {
    visibility: hidden;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background-color: #153E5C;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: white !important;
    border: 1px solid #D8DEE7 !important;
    border-radius: 8px;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #153E5C !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: white !important;
    color: #153E5C !important;
}

/* BRAND */

.brand-icon {
    width: 85px;
    height: 85px;
    background: #0D2F49;
    border-radius: 50%;
    margin: 10px auto 15px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 42px;
}

.brand-name {
    text-align: center;
    font-size: 25px;
    font-weight: 750;
    letter-spacing: 2px;
    color: white;
}

.brand-subtitle {
    text-align: center;
    font-size: 11px;
    color: #C5D6E4;
    margin-top: 7px;
}

.sidebar-heading {
    color: #B7CCDC;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 22px;
    margin-bottom: 10px;
}

/* PAGE HEADINGS */

.dashboard-heading {
    font-size: 29px;
    font-weight: 750;
    color: #1D3044;
    margin-bottom: 4px;
}

.dashboard-subtitle {
    font-size: 13px;
    color: #8996A5;
    margin-bottom: 22px;
}

.section-heading {
    font-size: 20px;
    font-weight: 700;
    color: #1D3044;
    margin-top: 15px;
    margin-bottom: 12px;
}

/* METRIC CARDS */

.metric-card {
    background: white;
    border: 1px solid #E5EAF0;
    border-radius: 4px;
    padding: 19px;
    min-height: 135px;
    box-shadow: 0 5px 15px rgba(20,43,65,0.09);
}

.metric-card.primary {
    background: #153E5C;
    border: none;
}

.metric-label {
    font-size: 12px;
    color: #778697;
}

.metric-card.primary .metric-label {
    color: #D5E2EC;
}

.metric-icon {
    float: right;
    color: #F5A000;
    font-size: 19px;
}

.metric-value {
    font-size: 31px;
    font-weight: 650;
    color: #182D42;
    margin-top: 15px;
}

.metric-card.primary .metric-value {
    color: white;
}

.metric-note {
    font-size: 11px;
    color: #98A4B1;
    margin-top: 4px;
}

/* INFORMATION CARDS */

.info-card {
    background: white;
    border: 1px solid #E5EAF0;
    border-radius: 5px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 3px 12px rgba(20,43,65,0.05);
}

.info-title {
    font-size: 12px;
    color: #8492A3;
    margin-bottom: 9px;
}

.info-value {
    font-size: 21px;
    font-weight: 700;
    color: #153E5C;
}

.info-description {
    font-size: 11px;
    color: #98A4B1;
    margin-top: 5px;
}

/* CHART EMPTY STATE */

.empty-state {
    height: 260px;
    background: white;
    border: 1px solid #E3E9EF;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: #8996A5;
    border-radius: 4px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #9AA5B2;
    font-size: 11px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 5. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_predictor():
    return NIDSPredictor()


try:
    predictor = load_predictor()
    model_loaded = True
    model_error = None

except Exception as error:
    predictor = None
    model_loaded = False
    model_error = str(error)


# =========================================================
# 6. MODEL VALIDATION BENCHMARKS
# From the previously evaluated 60,000-record validation set
# =========================================================


# =========================================================
# 6. FAMILY-AWARE HGB VALIDATION BENCHMARKS
# =========================================================

VALIDATION_TOTAL = 60061

TN = 49759
FP = 12
FN = 12
TP = 10278

VALIDATION_ACCURACY = (TN + TP) / VALIDATION_TOTAL

ATTACK_PRECISION = TP / (TP + FP)
ATTACK_RECALL = TP / (TP + FN)

ATTACK_F1 = (
    2 * ATTACK_PRECISION * ATTACK_RECALL
    / (ATTACK_PRECISION + ATTACK_RECALL)
)

# =========================================================
# 7. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="brand-icon">🛡️</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-name">NIDS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'NETWORK SECURITY CENTER'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">MAIN MENU</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Overview",
            "Detection Center",
            "Manual Testing",
            "Model Evaluation"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">NETWORK DATA</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload flow dataset",
        type=["csv"],
        help="Upload a CSV containing all 69 required model features."
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">DETECTION ENGINE</div>',
        unsafe_allow_html=True
    )

    if model_loaded:
        st.markdown("🟢 Model loaded")
        st.caption("Hist Gradient Boosting")
        st.caption("69 input features")
        st.caption("Binary classification")
    else:
        st.error("Model loading failed")
        st.caption(model_error)

    st.divider()

    st.caption("CIC-IDS2017")
    st.caption("NIDS Project")


# =========================================================
# 8. PROCESS UPLOADED FILE
# =========================================================

# =========================================================
# 8. ROBUST CSV UPLOAD AND VALIDATION
# =========================================================

results = None
input_data = None
processing_error = None
missing_features = []
available_columns = []
invalid_numeric_columns = []

if uploaded_file is not None and model_loaded:

    try:
        file_bytes = uploaded_file.getvalue()

        # Check empty file
        if not file_bytes:
            raise ValueError("The uploaded CSV file is empty.")

        # Read CSV
        try:
            input_data = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding="utf-8-sig"
            )
        except UnicodeDecodeError:
            input_data = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding="latin1"
            )

        # Check empty dataset
        if input_data.empty:
            raise ValueError(
                "The CSV contains column headers but no records."
            )

        # Normalize column names
        input_data.columns = (
            input_data.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        # Check duplicate column names
        duplicates = input_data.columns[
            input_data.columns.duplicated()
        ].tolist()

        if duplicates:
            raise ValueError(
                f"Duplicate column names found: {duplicates}"
            )

        available_columns = input_data.columns.tolist()

        # Check required model features
        required_features = predictor.feature_columns

        missing_features = [
            feature for feature in required_features
            if feature not in input_data.columns
        ]

        if missing_features:

            raise ValueError(
                f"Missing {len(missing_features)} of "
                f"{len(required_features)} required model features."
            )

        # Check numeric values
        X = input_data[required_features].copy()

        for column in required_features:
            converted = pd.to_numeric(
                X[column], errors="coerce"
            )

            invalid_count = int(
                (converted.isna() & X[column].notna()).sum()
            )

            if invalid_count > 0:
                invalid_numeric_columns.append(
                    (column, invalid_count)
                )

        # Normalize CSV before passing it to predictor
        normalized_bytes = input_data.to_csv(
            index=False
        ).encode("utf-8")

        with st.spinner("Analyzing network traffic..."):

            results = predictor.predict(
                io.BytesIO(normalized_bytes)
            )

        if results.empty:
            raise ValueError("No network flows were predicted.")

    except pd.errors.EmptyDataError:
        processing_error = "The uploaded file contains no readable CSV data."

    except pd.errors.ParserError:
        processing_error = (
            "Invalid CSV structure. Check delimiters, quotes "
            "and row formatting."
        )

    except UnicodeDecodeError:
        processing_error = (
            "Unable to decode the file. Save it as UTF-8 CSV."
        )

    except ValueError as error:
        processing_error = str(error)

    except Exception as error:
        processing_error = (
            f"Unexpected prediction error: {type(error).__name__}: {error}"
        )



# =========================================================
# 9. STOP ON INVALID UPLOAD
# =========================================================


# =========================================================
# 9. USER-FRIENDLY ERROR HANDLING
# =========================================================

if not model_loaded:
    st.error("The prediction model could not be loaded.")
    st.code(model_error)
    st.stop()

if processing_error and page != "Manual Testing":

    st.error("CSV validation failed")

    st.warning(processing_error)

    if missing_features:

        st.markdown("### Missing model features")

        st.write(
            f"{len(missing_features)} required features "
            f"are missing from your uploaded CSV."
        )

        st.dataframe(
            pd.DataFrame({
                "Missing feature": missing_features
            }),
            use_container_width=True,
            hide_index=True
        )

        # Suggest similar column names
        suggestions = []

        for feature in missing_features:
            matches = difflib.get_close_matches(
                feature,
                available_columns,
                n=1,
                cutoff=0.75
            )

            if matches:
                suggestions.append({
                    "Required feature": feature,
                    "Possible matching column": matches[0]
                })

        if suggestions:
            st.markdown("### Possible column-name corrections")
            st.dataframe(
                pd.DataFrame(suggestions),
                use_container_width=True,
                hide_index=True
            )

        # Download correct feature-header template
        template = pd.DataFrame(
            columns=predictor.feature_columns
        )

        st.download_button(
            "Download required feature template",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="NIDS_required_features_template.csv",
            mime="text/csv"
        )

    if available_columns:
        with st.expander("View uploaded CSV columns"):
            st.write(available_columns)

    st.info(
        "Upload a CICFlowMeter-compatible CSV containing "
        "all 69 required model features. Extra metadata "
        "columns are allowed."
    )

    st.stop()

if invalid_numeric_columns:

    st.warning(
        "Some feature values are non-numeric. "
        "They will be converted to missing values and "
        "handled by the model's fitted preprocessing pipeline."
    )

    st.dataframe(
        pd.DataFrame(
            invalid_numeric_columns,
            columns=["Feature", "Invalid values"]
        ),
        use_container_width=True,
        hide_index=True
    )



# =========================================================
# 10. PREPARE PREDICTION RESULTS
# =========================================================

if results is not None:

    results = results.copy()

    results["Status"] = results["Prediction"].map({
        0: "BENIGN",
        1: "ATTACK"
    })

    results["Attack_Score"] = (
        results["Attack_Probability"] * 100
    )

    results["Flow_ID"] = range(1, len(results) + 1)

    total = len(results)

    benign = int((results["Prediction"] == 0).sum())
    attacks = int((results["Prediction"] == 1).sum())

    benign_rate = benign / total * 100
    attack_rate = attacks / total * 100

    average_score = results["Attack_Score"].mean()


# =========================================================
# 11. REUSABLE METRIC CARD
# =========================================================

def metric_card(label, value, icon, note, primary=False):

    css_class = "metric-card primary" if primary else "metric-card"

    st.markdown(f"""
    <div class="{css_class}">
        <div class="metric-label">
            {label}
            <span class="metric-icon">{icon}</span>
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 12. HOME PAGE - MODEL INFORMATION
# =========================================================

if page == "Home":

    st.markdown(
        '<div class="dashboard-heading">NIDS Model Home</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Model information, architecture and validation performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.success("Your intrusion detection model is loaded and ready.")

    st.markdown(
        '<div class="section-heading">Model Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        metric_card(
            "MODEL ALGORITHM",
            "HGB",
            "🧠",
            "Hist Gradient Boosting",
            True
        )

    with c2:
        metric_card(
            "INPUT FEATURES",
            "69",
            "▦",
            "Network-flow features"
        )

    with c3:
        metric_card(
            "OUTPUT CLASSES",
            "2",
            "⚖",
            "BENIGN and ATTACK"
        )

    with c4:
        metric_card(
            "VALIDATION RECORDS",
            "60,061",
            "✓",
            "Previously evaluated records"
        )

    st.markdown(
        '<div class="section-heading">Validation Performance</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4 = st.columns(4, gap="medium")

    with p1:
        metric_card(
            "ACCURACY",
            f"{VALIDATION_ACCURACY * 100:.2f}%",
            "★",
            "Overall correct predictions",
            True
        )

    with p2:
        metric_card(
            "ATTACK PRECISION",
            f"{ATTACK_PRECISION * 100:.2f}%",
            "✓",
            "Correctness of attack alerts"
        )

    with p3:
        metric_card(
            "ATTACK RECALL",
            f"{ATTACK_RECALL * 100:.2f}%",
            "⚠",
            "Actual attacks detected"
        )

    with p4:
        metric_card(
            "ATTACK F1-SCORE",
            f"{ATTACK_F1 * 100:.2f}%",
            "◎",
            "Precision-recall balance"
        )

    st.markdown(
        '<div class="section-heading">Model Architecture</div>',
        unsafe_allow_html=True
    )

    architecture = st.columns(3, gap="medium")

    architecture_data = [
        (
            "01 · INPUT",
            "69 Network Features",
            "CICFlowMeter-compatible network-flow measurements."
        ),
        (
            "02 · PREPROCESSING",
            "SimpleImputer",
            "Handles missing feature values using the fitted training imputer."
        ),
        (
            "03 · CLASSIFICATION",
            "HistGradientBoostingClassifier",
            "Predicts BENIGN (0) or ATTACK (1)."
        )
    ]

    for col, (label, title, description) in zip(
        architecture,
        architecture_data
    ):

        with col:

            st.markdown(f"""
            <div class="info-card">
                <div class="info-title">{label}</div>
                <div class="info-value">{title}</div>
                <div class="info-description">{description}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Validation Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    cm1, cm2 = st.columns([1.4, 1], gap="large")

    with cm1:

        confusion = pd.DataFrame(
            [
                [TN, FP],
                [FN, TP]
            ],
            index=["Actual BENIGN", "Actual ATTACK"],
            columns=["Predicted BENIGN", "Predicted ATTACK"]
        )

        st.dataframe(
            confusion,
            use_container_width=True
        )

    with cm2:

        st.markdown("#### Prediction summary")

        st.metric("Correct predictions", f"{TN + TP:,}")
        st.metric("False alarms", f"{FP:,}")
        st.metric("Missed attacks", f"{FN:,}")

    st.markdown(
        '<div class="section-heading">How the model classifies traffic</div>',
        unsafe_allow_html=True
    )

    class1, class2 = st.columns(2, gap="medium")

    with class1:

        st.success(
            "**BENIGN — Class 0**\n\n"
            "Network traffic classified as normal activity."
        )

    with class2:

        st.error(
            "**ATTACK — Class 1**\n\n"
            "Network traffic flagged as potentially malicious."
        )

    st.warning(
        "**Generalization limitation:** These metrics describe the "
        "60,000-record validation evaluation. In the earlier Friday "
        "unseen-day test, attack recall was only about 28.46%. "
        "Therefore, the validation accuracy must not be presented "
        "as real-world detection accuracy."
    )

    st.caption(
        "Validation benchmark: previously measured on the "
        "CIC-IDS2017 validation split. It is a stored benchmark, "
        "not recalculated from every uploaded file."
    )


# =========================================================
# 13. OVERVIEW PAGE
# =========================================================

elif page == "Overview":

    st.markdown(
        '<div class="dashboard-heading">Security Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Network traffic intelligence and intrusion monitoring'
        '</div>',
        unsafe_allow_html=True
    )

    if results is None:

        st.info(
            "Upload a compatible network-flow CSV from the sidebar "
            "to display traffic analytics."
        )

        st.stop()

    # Four traffic metrics
    cols = st.columns(4, gap="medium")

    with cols[0]:
        metric_card("TOTAL FLOWS", f"{total:,}", "▦",
                    "Analyzed network flows", True)

    with cols[1]:
        metric_card("BENIGN TRAFFIC", f"{benign:,}", "✓",
                    f"{benign_rate:.2f}% of traffic")

    with cols[2]:
        metric_card("DETECTED ATTACKS", f"{attacks:,}", "⚠",
                    f"{attack_rate:.2f}% of traffic")

    with cols[3]:
        metric_card("AVG. ATTACK SCORE", f"{average_score:.2f}%", "★",
                    "Mean model probability")

    st.write("")

    left, right = st.columns([2.5, 1], gap="medium")

    with left:

        st.markdown("### Detection Results")

        if "Target" in input_data.columns:

            actual = pd.to_numeric(
                input_data["Target"],
                errors="coerce"
            )

            valid = actual.notna()

            actual = actual[valid].astype(int)
            predicted = results.loc[valid, "Prediction"]

            comparison = pd.crosstab(
                actual.map({0: "BENIGN", 1: "ATTACK"}),
                predicted.map({
                    0: "Predicted BENIGN",
                    1: "Predicted ATTACK"
                })
            )

            comparison = comparison.reindex(
                index=["BENIGN", "ATTACK"],
                columns=[
                    "Predicted BENIGN",
                    "Predicted ATTACK"
                ],
                fill_value=0
            )

            x_values = comparison.index.tolist()
            benign_values = comparison.iloc[:, 0].tolist()
            attack_values = comparison.iloc[:, 1].tolist()

            chart_title = "Actual versus predicted"

        else:

            x_values = ["Uploaded Traffic"]
            benign_values = [benign]
            attack_values = [attacks]

            chart_title = "Predicted classifications"

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="BENIGN",
            x=x_values,
            y=benign_values,
            marker_color=NAVY
        ))

        fig.add_trace(go.Bar(
            name="ATTACK",
            x=x_values,
            y=attack_values,
            marker_color=ORANGE
        ))

        fig.update_layout(
            barmode="group",
            title=chart_title,
            height=300,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=45, b=10),
            yaxis=dict(gridcolor="#E7ECF1"),
            xaxis=dict(showgrid=False)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with right:

        st.markdown("### Traffic Composition")

        donut = go.Figure(data=[go.Pie(
            labels=["BENIGN", "ATTACK"],
            values=[benign, attacks],
            hole=0.70,
            marker=dict(
                colors=[NAVY, ORANGE],
                line=dict(color="white", width=3)
            ),
            textinfo="none"
        )])

        donut.update_layout(
            height=300,
            paper_bgcolor="white",
            showlegend=True,
            margin=dict(l=5, r=5, t=10, b=10),
            annotations=[dict(
                text=f"{attack_rate:.1f}%",
                x=0.5,
                y=0.5,
                font=dict(size=25, color=TEXT),
                showarrow=False
            )]
        )

        st.plotly_chart(
            donut,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.write("")

    bottom_left, bottom_right = st.columns([2.5, 1])

    with bottom_left:

        st.markdown("### Attack Probability Trend")

        window = max(1, total // 100)

        rolling_score = (
            results["Attack_Score"]
            .rolling(window=window, min_periods=1)
            .mean()
        )

        trend = go.Figure()

        trend.add_trace(go.Scatter(
            x=np.arange(1, total + 1),
            y=rolling_score,
            name="Average attack score",
            line=dict(color=NAVY, width=2),
            fill="tozeroy",
            fillcolor="rgba(29,82,120,0.17)"
        ))

        trend.update_layout(
            height=250,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=15, b=10),
            xaxis_title="Flow number",
            yaxis_title="Attack probability (%)",
            yaxis=dict(range=[0, 100], gridcolor="#E7ECF1")
        )

        st.plotly_chart(
            trend,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with bottom_right:

        st.markdown("### Detection Summary")

        st.metric("Total flows", f"{total:,}")
        st.metric("Benign flows", f"{benign:,}")
        st.metric("Attack flows", f"{attacks:,}")
        st.metric("Average attack score", f"{average_score:.2f}%")

    if attacks > 0:

        st.warning(
            f"{attacks:,} network flows were classified as ATTACK. "
            "Review them in the Detection Center."
        )

    else:

        st.success("No flows were classified as attacks in this upload.")


# =========================================================
# 14. DETECTION CENTER
# =========================================================

elif page == "Detection Center":

    st.markdown("## Network Detection Center")

    if results is None:

        st.info("Upload a compatible network-flow CSV first.")

    else:

        col1, col2 = st.columns(2)

        with col1:

            status_filter = st.selectbox(
                "Classification",
                ["ALL", "BENIGN", "ATTACK"]
            )

        with col2:

            minimum_score = st.slider(
                "Minimum attack probability (%)",
                0, 100, 0
            )

        filtered = results.copy()

        if status_filter != "ALL":

            filtered = filtered[
                filtered["Status"] == status_filter
            ]

        filtered = filtered[
            filtered["Attack_Score"] >= minimum_score
        ]

        st.metric("Matching flows", f"{len(filtered):,}")

        table = filtered[
            ["Flow_ID", "Status", "Attack_Score"]
        ].rename(columns={
            "Flow_ID": "Flow ID",
            "Attack_Score": "Attack Probability (%)"
        })

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            height=450
        )

        st.download_button(
            "Download Detection Report",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="NIDS_Detection_Report.csv",
            mime="text/csv"
        )



# =========================================================
# 15. MANUAL NETWORK FLOW TESTING
# =========================================================

elif page == "Manual Testing":

    st.markdown("## Manual Network Flow Testing")
    st.markdown(
        "Enter the 69 CICFlowMeter features for one network flow "
        "and submit them to the trained family-aware HGB model."
    )

    st.info(
        "Use real feature values from a CICFlowMeter flow record. "
        "The initial zero values are placeholders only and should "
        "not be treated as a meaningful test."
    )

    if "manual_test_result" not in st.session_state:
        st.session_state.manual_test_result = None

    if "manual_test_input" not in st.session_state:
        st.session_state.manual_test_input = None

    st.markdown("### Enter network-flow features")

    with st.form("manual_flow_test_form"):
        manual_values = {}
        feature_cols = st.columns(3)

        for index, feature in enumerate(predictor.feature_columns):
            with feature_cols[index % 3]:
                manual_values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",
                    key=f"manual_feature_{index}",
                    help=f"Enter the numeric value for {feature}."
                )

        st.divider()
        confirm_actual_values = st.checkbox(
            "I have entered actual numeric values from a network-flow record."
        )
        run_manual_test = st.form_submit_button(
            "Analyze Network Flow",
            type="primary",
            use_container_width=True
        )

    if run_manual_test:
        if not confirm_actual_values:
            st.warning(
                "Please confirm that these are actual network-flow values. "
                "Placeholder zeros should not be used as a real test."
            )
        else:
            try:
                manual_input = pd.DataFrame(
                    [manual_values],
                    columns=predictor.feature_columns
                )

                manual_result = predictor.predict(
                    io.BytesIO(manual_input.to_csv(index=False).encode("utf-8"))
                )

                st.session_state.manual_test_input = manual_input
                st.session_state.manual_test_result = manual_result

            except Exception as error:
                st.session_state.manual_test_input = None
                st.session_state.manual_test_result = None
                st.error(
                    f"Manual prediction failed: {type(error).__name__}: {error}"
                )

    if st.session_state.manual_test_result is not None:
        manual_result = st.session_state.manual_test_result.copy()
        manual_prediction = int(manual_result.iloc[0]["Prediction"])
        manual_probability = float(
            manual_result.iloc[0]["Attack_Probability"]
        )
        manual_status = str(manual_result.iloc[0]["Status"])

        st.divider()
        st.markdown("### Prediction result")

        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Classification", manual_status)
        result_col2.metric(
            "Attack probability",
            f"{manual_probability * 100:.4f}%"
        )
        result_col3.metric("Decision threshold", "50%")

        if manual_prediction == 1:
            st.error(
                "ATTACK: The model classified this flow as potentially malicious."
            )
        else:
            st.success(
                "BENIGN: The model classified this flow as benign."
            )

        st.caption(
            "This is a model prediction, not proof that a flow is safe or malicious."
        )

        st.markdown("### Submitted feature values")
        st.dataframe(
            st.session_state.manual_test_input,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Prediction details")
        st.dataframe(
            manual_result,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download manual test result",
            data=manual_result.to_csv(index=False).encode("utf-8"),
            file_name="NIDS_Manual_Test_Result.csv",
            mime="text/csv"
        )

# =========================================================
# 16. MODEL EVALUATION PAGE
# =========================================================

elif page == "Model Evaluation":

    st.markdown("## Model Performance Evaluation")

    if results is None:

        st.info(
            "Upload a labeled validation CSV to calculate performance."
        )

    elif "Target" not in input_data.columns:

        st.warning(
            "Actual Target labels are not available. "
            "Accuracy cannot be calculated for this dataset."
        )

    else:

        actual = pd.to_numeric(
            input_data["Target"],
            errors="coerce"
        )

        valid = actual.notna()

        actual = actual[valid].astype(int)
        predicted = results.loc[valid, "Prediction"]

        accuracy = accuracy_score(actual, predicted)
        precision = precision_score(
            actual, predicted, zero_division=0
        )
        recall = recall_score(
            actual, predicted, zero_division=0
        )
        f1 = f1_score(
            actual, predicted, zero_division=0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
        c2.metric("Precision", f"{precision * 100:.2f}%")
        c3.metric("Recall", f"{recall * 100:.2f}%")
        c4.metric("F1-score", f"{f1 * 100:.2f}%")

        st.markdown("### Confusion Matrix")

        cm = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1]
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual BENIGN", "Actual ATTACK"],
            columns=["Predicted BENIGN", "Predicted ATTACK"]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        st.markdown("### Classification Report")

        report = classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).transpose().round(4),
            use_container_width=True
        )


# =========================================================
# 17. FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    'NIDS SECURITY OPERATIONS CENTER · CIC-IDS2017 · '
    'HIST GRADIENT BOOSTING'
    '</div>',
    unsafe_allow_html=True
)
