from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
from feature_engineering import (
    INPUT_FEATURES, MODEL_FEATURES, add_engineered_features
)
from audit_logger import append_audit_record

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "credit_data.csv"
MODEL_DIR = BASE_DIR / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"
AUDIT_LOG_PATH = BASE_DIR / "logs" / "audit_log.csv"
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
}

st.set_page_config(
    page_title="Credit Scoring Model | CodeAlpha",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background: #f5f8fc; color: #182230; }
    [data-testid="stHeader"] { background: rgba(245,248,252,0.92); }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e1e8f0;
        padding: 16px 18px; border-radius: 14px;
    }
    div[data-testid="stTabs"] button {
        font-weight: 600; font-size: 15px;
    }
    .main .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
    .notice {
        background: #eaf2ff; border-left: 4px solid #2563eb;
        padding: 12px 16px; border-radius: 8px; color: #1e3a5f;
    }
</style>
""", unsafe_allow_html=True)

st.title("💳 Credit Scoring Model")
st.caption("CodeAlpha Machine Learning Internship Project")
st.markdown(
    '<div class="notice"><b>Data note:</b> This project uses synthetic sample data. '
    'It is for learning and demonstration only, not for real credit or lending decisions.</div>',
    unsafe_allow_html=True
)

if not DATA_PATH.exists():
    st.error("credit_data.csv is missing. Put it in the same folder as app.py.")
    st.stop()

data = pd.read_csv(DATA_PATH)
if not METRICS_PATH.exists() or not all((MODEL_DIR / f).exists() for f in MODEL_FILES.values()):
    st.warning("The models have not been trained yet. Use the command shown below.")
    st.code("python train_model.py")
    st.stop()

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)
models = {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}

total_rows = len(data)
positive_rate = data["creditworthy"].mean() * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset records", f"{total_rows:,}")
c2.metric("ML models", "3")
c3.metric("User input fields", "8")
c4.metric("Creditworthy labels", f"{positive_rate:.1f}%")

tabs = st.tabs([
    "Overview", "Credit Prediction", "Model Comparison",
    "Confusion Matrix", "ROC Curve", "Feature Insights", "Dataset", "Audit Log"
])

with tabs[0]:
    st.subheader("Project Overview")
    st.write(
        "A creditworthiness classification project comparing Logistic Regression, "
        "Decision Tree, and Random Forest on a held-out test dataset."
    )
    st.caption(
        "Feature engineering: the model automatically derives a "
        "debt-to-annual-income ratio from total debt and annual income."
    )
    st.markdown("**Dataset label distribution**")
    counts = data["creditworthy"].map({0: "Not creditworthy", 1: "Creditworthy"}).value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=100)
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of records")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

with tabs[1]:
    st.subheader("Creditworthiness Prediction")
    st.write("Enter financial details to generate predictions from the selected model.")
    with st.form("prediction_form"):
        a, b, c = st.columns(3)
        with a:
            age = st.slider("Age", 18, 80, 35)
            annual_income = st.number_input("Annual income", min_value=0, max_value=2_000_000, value=60000, step=1000)
            total_debt = st.number_input("Total debt", min_value=0, max_value=2_000_000, value=15000, step=1000)
        with b:
            payment_history = st.selectbox("Payment history", ["Good", "Average", "Poor"])
            employment_years = st.slider("Years employed", 0.0, 45.0, 5.0, 0.5)
            credit_utilization = st.slider("Credit utilization", 0.0, 1.0, 0.30, 0.01)
        with c:
            existing_loans = st.slider("Existing loans", 0, 10, 1)
            late_payments = st.slider("Late payments", 0, 20, 1)
            chosen_model = st.selectbox("Model for main result", MODEL_NAMES, index=2)
        submitted = st.form_submit_button("Predict creditworthiness", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "age": age, "annual_income": annual_income, "total_debt": total_debt,
            "payment_history": payment_history, "employment_years": employment_years,
            "credit_utilization": credit_utilization, "existing_loans": existing_loans,
            "late_payments": late_payments
        }], columns=INPUT_FEATURES)
        model_row = add_engineered_features(row)[MODEL_FEATURES]
        all_model_outputs = {}
        for name in MODEL_NAMES:
            model_prediction = int(models[name].predict(model_row)[0])
            model_probability = float(models[name].predict_proba(model_row)[0][1])
            all_model_outputs[name] = {
                "predicted_label": "Creditworthy" if model_prediction == 1 else "Not creditworthy",
                "probability_creditworthy": round(model_probability, 6),
            }

        selected_output = all_model_outputs[chosen_model]
        result_label = selected_output["predicted_label"]
        probability = selected_output["probability_creditworthy"]

        try:
            audit_record = append_audit_record(
                AUDIT_LOG_PATH,
                selected_model=chosen_model,
                predicted_label=result_label,
                probability_creditworthy=probability,
                all_model_outputs=all_model_outputs,
            )
            st.success(f"Prediction recorded in the audit log. ID: {audit_record['prediction_id'][:8]}")
        except (OSError, ValueError, TypeError) as exc:
            st.warning(f"The prediction worked, but the audit record could not be saved: {exc}")

        st.markdown(f"### Result: {result_label}")
        st.metric("Model-estimated probability of creditworthy label", f"{probability:.1%}")
        st.markdown("**Compare all three models for this same input**")
        cols = st.columns(3)
        for col, name in zip(cols, MODEL_NAMES):
            output = all_model_outputs[name]
            col.metric(
                name,
                output["predicted_label"],
                f"{output['probability_creditworthy']:.1%} probability"
            )

with tabs[2]:
    st.subheader("Model Comparison")
    rows = []
    for name in MODEL_NAMES:
        m = metrics[name]
        rows.append({
            "Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
            "Recall": m["recall"], "F1-score": m["f1_score"], "ROC-AUC": m["roc_auc"]
        })
    metric_frame = pd.DataFrame(rows).set_index("Model")
    st.dataframe(metric_frame.style.format("{:.3f}"), use_container_width=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    metric_frame.plot(kind="bar", ax=ax, ylim=(0, 1))
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Held-out test-set performance")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Scores are calculated by train_model.py on a held-out test set.")

with tabs[3]:
    st.subheader("Confusion Matrix")
    selected_cm = st.selectbox("Select model", MODEL_NAMES, key="cm_model")
    cm = np.array(metrics[selected_cm]["confusion_matrix"])
    # Keep the confusion matrix compact so the complete chart fits on screen.
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=100)
    image = ax.imshow(cm, interpolation="nearest", cmap="YlGnBu")
    ax.set_title(f"{selected_cm} — Confusion Matrix", pad=12, fontsize=13)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04, shrink=0.82)
    labels = ["Not credit-\nworthy", "Creditworthy"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xlabel("Predicted label", fontsize=10, labelpad=8)
    ax.set_ylabel("Actual label", fontsize=10, labelpad=8)
    midpoint = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > midpoint else "black", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.22, right=0.88, top=0.86, bottom=0.22)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    st.write("Rows represent actual labels; columns represent predicted labels.")

with tabs[4]:
    st.subheader("ROC Curve")
    test_data_path = MODEL_DIR / "test_data.csv"
    if test_data_path.exists():
        test_data = pd.read_csv(test_data_path)
        X_test = test_data[MODEL_FEATURES]
        y_test = test_data["creditworthy"]
        # Medium-sized figure: keep the ROC chart readable without stretching it full-width.
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=100)
        for name in MODEL_NAMES:
            probabilities = models[name].predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probabilities)
            curve_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2.5, label=f"{name} (AUC = {curve_auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random baseline")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Test Set")
        ax.legend(loc="lower right", fontsize=9, frameon=True)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    else:
        st.warning("Test data not found. Run python train_model.py first.")

with tabs[5]:
    st.subheader("Feature Insights")
    st.write("Permutation importance estimates how much a model's test-set score changes when one feature is shuffled.")
    test_data_path = MODEL_DIR / "test_data.csv"
    if test_data_path.exists():
        test_data = pd.read_csv(test_data_path)
        X_test = test_data[MODEL_FEATURES]
        y_test = test_data["creditworthy"]
        selected_imp = st.selectbox("Select model for feature importance", MODEL_NAMES, key="imp_model")
        with st.spinner("Calculating feature importance..."):
            importance = permutation_importance(
                models[selected_imp], X_test, y_test, n_repeats=5,
                random_state=42, scoring="f1"
            )
        display_names = {
            "age": "Age",
            "annual_income": "Annual income",
            "total_debt": "Total debt",
            "employment_years": "Employment years",
            "credit_utilization": "Credit utilization",
            "existing_loans": "Existing loans",
            "late_payments": "Late payments",
            "debt_to_annual_income_ratio": "Debt / annual income ratio",
            "payment_history": "Payment history",
        }
        importance_frame = pd.DataFrame({
            "Feature": [display_names.get(name, name) for name in MODEL_FEATURES],
            "Importance": importance.importances_mean
        }).sort_values("Importance", ascending=True)
        # Keep this chart medium-sized while reserving room for the derived feature label.
        fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=100)
        ax.barh(importance_frame["Feature"], importance_frame["Importance"])
        ax.set_xlabel("Mean decrease in F1-score after shuffling", fontsize=9, labelpad=8)
        ax.set_title(f"Permutation Feature Importance — {selected_imp}", fontsize=12, pad=12)
        ax.tick_params(axis="y", labelsize=9, pad=5)
        ax.tick_params(axis="x", labelsize=9)
        fig.subplots_adjust(left=0.34, right=0.97, top=0.88, bottom=0.16)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    else:
        st.warning("Test data not found. Run python train_model.py first.")

with tabs[6]:
    st.subheader("Dataset Preview")
    st.write("Sample records included with this project.")
    st.caption(
        "The dataset contains 8 original input fields. During training and prediction, "
        "the model also derives debt_to_annual_income_ratio from total_debt and annual_income."
    )
    st.dataframe(data.head(50), use_container_width=True)
    st.write("**Rows and columns:**", data.shape)
    st.write("**Missing values by column**")
    st.dataframe(data.isna().sum().rename("Missing values"), use_container_width=True)
    st.download_button(
        "Download dataset CSV", data=data.to_csv(index=False).encode("utf-8"),
        file_name="credit_data.csv", mime="text/csv"
    )

with tabs[7]:
    st.subheader("Prediction Audit Log")
    st.write(
        "A local history of prediction events, including the selected model, "
        "result, probability, and outputs from all three models."
    )
    st.caption(
        "Privacy note: names and submitted financial input values are not stored. "
        "The log records prediction metadata only and uses UTC timestamps."
    )
    if AUDIT_LOG_PATH.exists() and AUDIT_LOG_PATH.stat().st_size > 0:
        audit_frame = pd.read_csv(AUDIT_LOG_PATH)
        if audit_frame.empty:
            st.info("No prediction events have been recorded yet.")
        else:
            st.metric("Recorded prediction events", f"{len(audit_frame):,}")
            visible_columns = [
                "timestamp_utc", "prediction_id", "selected_model",
                "predicted_label", "probability_creditworthy", "dataset_type"
            ]
            available_columns = [column for column in visible_columns if column in audit_frame.columns]
            st.dataframe(
                audit_frame[available_columns].tail(50).iloc[::-1],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download audit log CSV",
                data=audit_frame.to_csv(index=False).encode("utf-8"),
                file_name="credit_scoring_audit_log.csv",
                mime="text/csv",
            )
            st.caption("The table shows the 50 most recent events. The download contains all recorded events.")
    else:
        st.info("No predictions have been logged yet. Submit a prediction in the Credit Prediction tab to create the first audit record.")

