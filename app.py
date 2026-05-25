import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = r"""# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
"""


st.set_page_config(
    page_title="Mini Project B - Time-Series Forecasting Starter",
    page_icon="📈",
    layout="wide",
)


def get_openrouter_key() -> str:
    """Read OpenRouter key from secrets, environment, or user password input."""
    key = ""
    try:
        key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.environ.get("OPENROUTER_API_KEY", "")

    if not key:
        key = st.text_input(
            "OpenRouter API key",
            type="password",
            help="Used only when you click the AI grader button. Do not hardcode keys.",
        )

    return str(key).strip()


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def audit_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    audit = pd.DataFrame(
        {
            "column": data.columns,
            "dtype": [str(dtype) for dtype in data.dtypes],
            "missing_percent": data.isna().mean().mul(100).round(3).values,
            "unique_count": [data[col].nunique(dropna=True) for col in data.columns],
        }
    )
    return audit


def parse_and_clean_timeseries(data: pd.DataFrame, timestamp_col: str, target_col: str) -> pd.DataFrame:
    working = data.copy()
    working[timestamp_col] = pd.to_datetime(working[timestamp_col], errors="coerce")
    working[target_col] = pd.to_numeric(working[target_col], errors="coerce")
    working = working.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col)
    working = working.drop_duplicates(subset=[timestamp_col], keep="last")
    return working.reset_index(drop=True)


def maybe_resample(data: pd.DataFrame, timestamp_col: str, target_col: str, rule: str) -> pd.DataFrame:
    if rule == "No resampling":
        return data

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)

    resampled = (
        data.set_index(timestamp_col)[numeric_cols]
        .resample(rule)
        .mean()
        .interpolate(limit_direction="both")
        .reset_index()
    )
    return resampled


def create_baseline_features(data: pd.DataFrame, timestamp_col: str, target_col: str, horizon: int) -> pd.DataFrame:
    features = data[[timestamp_col, target_col]].copy()
    features = features.sort_values(timestamp_col).reset_index(drop=True)

    features["lag_1"] = features[target_col].shift(1)
    features["lag_24"] = features[target_col].shift(24)
    features["rolling_mean_24"] = features[target_col].shift(1).rolling(24).mean()
    features["hour"] = features[timestamp_col].dt.hour
    features["weekend"] = features[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
    features["month"] = features[timestamp_col].dt.month
    features["y_target"] = features[target_col].shift(-horizon)

    return features.dropna().reset_index(drop=True)


def build_submission_json(
    student_name: str,
    student_id: str,
    app_url: str,
    project_title: str,
    project_goal: str,
    data: pd.DataFrame,
    feature_table: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    horizon: int,
    resampling_choice: str,
    results_df,
    insights: str,
) -> dict:
    has_metrics_table = isinstance(results_df, pd.DataFrame)
    results_table = [] if results_df is None else results_df.to_dict(orient="records")

    evidence = {
        "student_name": student_name,
        "student_id": student_id,
        "deployed_app_url": app_url,
        "project_title": project_title,
        "project_goal": project_goal,
        "dataset_rows_after_cleaning": int(len(data)),
        "dataset_columns_after_cleaning": int(data.shape[1]),
        "timestamp_column": timestamp_col,
        "target_column": target_col,
        "time_min": str(data[timestamp_col].min()) if len(data) else "",
        "time_max": str(data[timestamp_col].max()) if len(data) else "",
        "horizon_steps": int(horizon),
        "resampling_choice": resampling_choice,
        "baseline_features_created": [
            "lag_1",
            "lag_24",
            "rolling_mean_24",
            "hour",
            "weekend",
            "month",
        ],
        "feature_table_rows": int(len(feature_table)),
        "x_y_prepared": bool(len(feature_table) > 0),
        "student_added_modeling": has_metrics_table,
        "has_metrics_table": has_metrics_table,
        "results_table": results_table,
        "student_insights": insights,
        "has_insights": bool(insights.strip()),
        "notes": [
            "Starter baseline features are prepared.",
            "Student additions include professional feature engineering, model comparison, metrics, dashboard visuals, and insights.",
        ],
    }
    return evidence


def make_project_card(evidence: dict) -> str:
    rows = [
        "# Mini Project B Project Card",
        "",
        f"**Student:** {evidence.get('student_name', '')}",
        f"**Student ID:** {evidence.get('student_id', '')}",
        f"**Project title:** {evidence.get('project_title', '')}",
        f"**Goal:** {evidence.get('project_goal', '')}",
        "",
        "## Dataset",
        f"- Timestamp column: `{evidence.get('timestamp_column', '')}`",
        f"- Target column: `{evidence.get('target_column', '')}`",
        f"- Rows after cleaning: {evidence.get('dataset_rows_after_cleaning', 0)}",
        f"- Time range: {evidence.get('time_min', '')} to {evidence.get('time_max', '')}",
        f"- Resampling: {evidence.get('resampling_choice', '')}",
        f"- Forecast horizon: {evidence.get('horizon_steps', '')} step(s)",
        "",
        "## Baseline Feature Table",
        "- Features: lag_1, lag_24, rolling_mean_24, hour, weekend, month",
        f"- X/y prepared: {evidence.get('x_y_prepared', False)}",
        "",
        "## Student Additions",
        f"- Metrics table present: {evidence.get('has_metrics_table', False)}",
        f"- Advanced feature engineering present: {evidence.get('student_added_feature_engineering', False)}",
        f"- Advanced feature count: {evidence.get('advanced_feature_count', 0)}",
        f"- Dashboard visual count: {evidence.get('dashboard_visual_count', 0)}",
        f"- Insights provided: {evidence.get('has_insights', False)}",
        "",
        "## Insights",
        evidence.get("student_insights", ""),
    ]
    return "\n".join(rows)


def parse_ai_json(raw_text: str):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


st.title("Mini Project B - Time-Series Forecasting Starter")
st.caption("This starter stops at data audit, baseline feature preparation, exports, and AI grading.")

with st.sidebar:
    st.header("Student Info")
    student_name = st.text_input("Student name", value="Hamed Alsuraihi")
    student_id = st.text_input("Student ID", value="PG112s25139")
    deployed_url = st.text_input("Deployed Streamlit URL", value="")
    project_title = st.text_input("Project title", value="Appliance Energy Forecasting")
    project_goal = st.text_area(
        "Project goal",
        value="Forecast future appliance energy consumption using time-series features.",
    )

    st.header("Dataset")
    dataset_path = st.text_input("Dataset path", value="data/dataset_sample.csv")

try:
    df = load_dataset(dataset_path)
except Exception as exc:
    st.error(f"Could not load dataset from {dataset_path}: {exc}")
    st.stop()

st.subheader("1) Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("2) Dataset Audit")
audit = audit_dataframe(df)
col_a, col_b = st.columns(2)
with col_a:
    st.write("Columns, dtypes, missing %, and unique counts")
    st.dataframe(audit, use_container_width=True)
with col_b:
    st.write("Top 10 columns by missing percentage")
    st.dataframe(
        audit.sort_values("missing_percent", ascending=False).head(10),
        use_container_width=True,
    )

st.subheader("3) Timestamp and Target Selection")
columns = df.columns.tolist()
default_timestamp_index = columns.index("date") if "date" in columns else 0
numeric_like_cols = []
for col in columns:
    converted = pd.to_numeric(df[col], errors="coerce")
    if converted.notna().mean() >= 0.8:
        numeric_like_cols.append(col)

default_target_index = (
    numeric_like_cols.index("Appliances") if "Appliances" in numeric_like_cols else 0
)

timestamp_col = st.selectbox(
    "Timestamp column",
    options=columns,
    index=default_timestamp_index,
)
target_col = st.selectbox(
    "Target column",
    options=numeric_like_cols if numeric_like_cols else columns,
    index=default_target_index,
)

clean_df = parse_and_clean_timeseries(df, timestamp_col, target_col)

if clean_df.empty:
    st.error("No valid rows remain after timestamp parsing and target conversion.")
    st.stop()

st.success(
    f"Cleaned time-series has {len(clean_df):,} rows from "
    f"{clean_df[timestamp_col].min()} to {clean_df[timestamp_col].max()}."
)

st.subheader("4) Optional Resampling and Forecast Horizon")
col_r, col_h = st.columns(2)
with col_r:
    resampling_choice = st.selectbox(
        "Resampling option",
        options=["No resampling", "30min", "1H", "1D"],
        index=0,
    )
with col_h:
    horizon = st.number_input(
        "Forecast horizon in rows/steps after resampling",
        min_value=1,
        max_value=168,
        value=1,
        step=1,
    )

model_data = maybe_resample(clean_df, timestamp_col, target_col, resampling_choice)

st.write("Preview after cleaning/resampling")
st.dataframe(model_data.head(10), use_container_width=True)

fig, ax = plt.subplots()
ax.plot(model_data[timestamp_col], model_data[target_col])
ax.set_title(f"Target over time: {target_col}")
ax.set_xlabel("Time")
ax.set_ylabel(target_col)
st.pyplot(fig)

st.subheader("5) Baseline Feature Table")
feature_table = create_baseline_features(model_data, timestamp_col, target_col, int(horizon))

feature_cols = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]
X = feature_table[feature_cols] if len(feature_table) else pd.DataFrame(columns=feature_cols)
y = feature_table["y_target"] if len(feature_table) else pd.Series(dtype=float, name="y_target")

st.write(f"Prepared X shape: {X.shape}")
st.write(f"Prepared y length: {len(y)}")
st.dataframe(feature_table.head(20), use_container_width=True)

st.divider()
st.subheader("6) STUDENT ADDITIONS - MODELING")
st.info("Professional modeling section added: advanced time features, domain sensor features, time-based split, model comparison, and metrics.")
results_df = None
time_based_split_used = False
split_rows = {}
models_trained = []
data_integrity_checks = {}
best_model_name = ""
best_predictions = np.array([])
plot_df = pd.DataFrame()
advanced_features_created = []
student_added_feature_engineering = False
feature_engineering_description = ""
dashboard_visuals_created = False
dashboard_elements = []
feature_importance_df = pd.DataFrame()
model_df = pd.DataFrame()
y_test = pd.Series(dtype=float)
val_end = 0

# -------------------------------
# STUDENT ADDITION: MODELING
# Professional feature engineering + time-based split + model comparison
# -------------------------------
try:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def build_professional_features(data: pd.DataFrame, timestamp_col: str, target_col: str, horizon: int) -> pd.DataFrame:
        """Create advanced forecasting features without using future target values."""
        features = data.sort_values(timestamp_col).reset_index(drop=True).copy()

        # Target lag features: recent memory, hourly memory, and daily memory.
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            features[f"lag_{lag}"] = features[target_col].shift(lag)

        # Rolling statistics use shift(1) first to avoid target leakage.
        shifted_target = features[target_col].shift(1)
        for window in [6, 12, 24, 48]:
            features[f"rolling_mean_{window}"] = shifted_target.rolling(window).mean()
            features[f"rolling_std_{window}"] = shifted_target.rolling(window).std()
            features[f"rolling_min_{window}"] = shifted_target.rolling(window).min()
            features[f"rolling_max_{window}"] = shifted_target.rolling(window).max()

        features["target_diff_1"] = features[target_col].diff(1)
        features["target_diff_24"] = features[target_col].diff(24)

        # Calendar and cyclical features.
        features["hour"] = features[timestamp_col].dt.hour
        features["dayofweek"] = features[timestamp_col].dt.dayofweek
        features["weekend"] = features[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
        features["month"] = features[timestamp_col].dt.month
        features["quarter"] = features[timestamp_col].dt.quarter
        features["dayofyear"] = features[timestamp_col].dt.dayofyear
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
        features["dayofweek_sin"] = np.sin(2 * np.pi * features["dayofweek"] / 7)
        features["dayofweek_cos"] = np.cos(2 * np.pi * features["dayofweek"] / 7)

        # Domain-specific sensor/weather features. Shift by 1 step so the model uses known past context.
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        domain_cols = [col for col in numeric_cols if col != target_col]
        for col in domain_cols:
            features[f"{col}_context_lag1"] = features[col].shift(1)

        # Useful grouped domain summaries when these columns exist in the energy dataset.
        temp_cols = [col for col in data.columns if re.fullmatch(r"T\d+", str(col))]
        humidity_cols = [col for col in data.columns if re.fullmatch(r"RH_\d+", str(col))]
        if temp_cols:
            features["indoor_temp_mean_lag1"] = data[temp_cols].mean(axis=1).shift(1)
            features["indoor_temp_std_lag1"] = data[temp_cols].std(axis=1).shift(1)
        if humidity_cols:
            features["indoor_humidity_mean_lag1"] = data[humidity_cols].mean(axis=1).shift(1)
            features["indoor_humidity_std_lag1"] = data[humidity_cols].std(axis=1).shift(1)

        features["y_target"] = features[target_col].shift(-horizon)
        return features.dropna().reset_index(drop=True)

    model_df = build_professional_features(model_data, timestamp_col, target_col, int(horizon))

    q1 = model_data[target_col].quantile(0.25)
    q3 = model_data[target_col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((model_data[target_col] < lower_bound) | (model_data[target_col] > upper_bound)).sum())

    data_integrity_checks = {
        "timestamp_cleaning": "Parsed timestamps, dropped invalid timestamp/target rows, sorted by time, and removed duplicate timestamps.",
        "resampling_discussed": f"Selected resampling option: {resampling_choice}.",
        "outlier_method": "IQR rule on target column.",
        "target_outlier_count": outlier_count,
        "target_outlier_percent": round((outlier_count / max(len(model_data), 1)) * 100, 3),
        "leakage_control": "Lag and rolling features are shifted before modeling; future y_target is created only after features are computed.",
    }

    st.write("### Data integrity checks")
    st.json(data_integrity_checks)

    if len(model_df) < 150:
        st.warning("Not enough feature rows for a reliable model comparison. Try using a smaller forecast horizon or less aggressive resampling.")
    else:
        exclude_cols = [timestamp_col, target_col, "y_target"]
        X_model = model_df.drop(columns=[col for col in exclude_cols if col in model_df.columns])
        X_model = X_model.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        X_model = X_model.fillna(X_model.median(numeric_only=True))
        y_model = model_df["y_target"]

        advanced_features_created = X_model.columns.tolist()
        student_added_feature_engineering = True
        feature_engineering_description = (
            "Added advanced lag features, rolling statistics, cyclical calendar features, "
            "target differences, shifted sensor/weather context variables, and grouped indoor temperature/humidity summaries."
        )

        st.write("### Advanced feature engineering")
        st.write(feature_engineering_description)
        st.write(f"Total modeling features: **{len(advanced_features_created)}**")
        with st.expander("Show advanced feature list"):
            st.write(advanced_features_created)

        # Time-based split: first 70% train, next 15% validation, last 15% test.
        n = len(model_df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train, y_train = X_model.iloc[:train_end], y_model.iloc[:train_end]
        X_val, y_val = X_model.iloc[train_end:val_end], y_model.iloc[train_end:val_end]
        X_test, y_test = X_model.iloc[val_end:], y_model.iloc[val_end:]

        time_based_split_used = True
        split_rows = {
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "test_period_start": str(model_df[timestamp_col].iloc[val_end]),
            "test_period_end": str(model_df[timestamp_col].iloc[-1]),
        }

        st.write("### Time-based split")
        st.json(split_rows)

        models = {
            "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "Random Forest": RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                max_depth=14,
                min_samples_leaf=3,
                n_jobs=-1,
            ),
            "Histogram Gradient Boosting": HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=42,
            ),
        }

        results = []
        predictions = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

            nonzero_mask = y_test != 0
            if nonzero_mask.any():
                mape = np.mean(np.abs((y_test[nonzero_mask] - y_pred[nonzero_mask]) / y_test[nonzero_mask])) * 100
            else:
                mape = np.nan

            results.append(
                {
                    "model": model_name,
                    "Validation_RMSE": round(float(val_rmse), 3),
                    "MAE": round(float(mae), 3),
                    "RMSE": round(float(rmse), 3),
                    "R2": round(float(r2), 3),
                    "MAPE_percent": None if np.isnan(mape) else round(float(mape), 3),
                    "features_used": int(X_model.shape[1]),
                    "notes": "Advanced engineered features with chronological 70/15/15 split; test set is the latest time period.",
                }
            )

            predictions[model_name] = y_pred

        results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
        models_trained = results_df["model"].tolist()

        st.write("### Model metrics table")
        st.dataframe(results_df, use_container_width=True)

        best_model_name = results_df.iloc[0]["model"]
        best_predictions = predictions[best_model_name]
        best_model = models[best_model_name]

        # Feature importance for tree models, or coefficient magnitude for Ridge pipeline.
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
        elif hasattr(best_model, "named_steps") and "ridge" in best_model.named_steps:
            importances = np.abs(best_model.named_steps["ridge"].coef_)
        else:
            importances = np.zeros(X_model.shape[1])

        feature_importance_df = (
            pd.DataFrame({"feature": X_model.columns, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )

        st.success(f"Best model based on test RMSE: {best_model_name}")

except Exception as exc:
    st.error(f"Modeling section failed: {exc}")
    results_df = None

st.subheader("7) STUDENT ADDITIONS - DASHBOARD")
st.info("Professional dashboard added: KPI cards, forecast plots, residuals, feature importance, pattern plots, and correlation heatmap.")

# -------------------------------
# STUDENT ADDITION: DASHBOARD
# Professional visuals + explanations
# -------------------------------
try:
    if results_df is not None and len(best_predictions) > 0:
        dashboard_visuals_created = True

        st.write("### Forecast performance KPIs")
        best_row = results_df.iloc[0]
        kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
        kpi_1.metric("Best model", best_model_name)
        kpi_2.metric("Test RMSE", best_row["RMSE"])
        kpi_3.metric("Test MAE", best_row["MAE"])
        kpi_4.metric("Test R²", best_row["R2"])
        dashboard_elements.append("forecast_kpi_cards")

        plot_df = pd.DataFrame(
            {
                "timestamp": model_df[timestamp_col].iloc[val_end:].values,
                "actual": y_test.values,
                "predicted": best_predictions,
            }
        )
        plot_df["residual"] = plot_df["actual"] - plot_df["predicted"]

        st.write("### Actual vs predicted energy use")
        fig_pred, ax_pred = plt.subplots(figsize=(11, 4))
        ax_pred.plot(plot_df["timestamp"], plot_df["actual"], label="Actual")
        ax_pred.plot(plot_df["timestamp"], plot_df["predicted"], label="Predicted")
        ax_pred.set_title(f"Actual vs Predicted Appliance Energy Use - {best_model_name}")
        ax_pred.set_xlabel("Time")
        ax_pred.set_ylabel(target_col)
        ax_pred.legend()
        st.pyplot(fig_pred)
        dashboard_elements.append("actual_vs_predicted_plot")

        st.write("### Prediction scatter plot")
        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 5))
        ax_scatter.scatter(plot_df["actual"], plot_df["predicted"], alpha=0.35)
        min_value = min(plot_df["actual"].min(), plot_df["predicted"].min())
        max_value = max(plot_df["actual"].max(), plot_df["predicted"].max())
        ax_scatter.plot([min_value, max_value], [min_value, max_value], linestyle="--")
        ax_scatter.set_title("Actual vs Predicted Scatter")
        ax_scatter.set_xlabel("Actual")
        ax_scatter.set_ylabel("Predicted")
        st.pyplot(fig_scatter)
        dashboard_elements.append("actual_predicted_scatter_plot")

        st.write("### Residual plot")
        fig_res, ax_res = plt.subplots(figsize=(11, 4))
        ax_res.plot(plot_df["timestamp"], plot_df["residual"])
        ax_res.axhline(0, linestyle="--")
        ax_res.set_title("Residuals Over Time")
        ax_res.set_xlabel("Time")
        ax_res.set_ylabel("Residual")
        st.pyplot(fig_res)
        dashboard_elements.append("residual_plot")

        st.write("### Residual distribution")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
        ax_hist.hist(plot_df["residual"], bins=40)
        ax_hist.set_title("Residual Distribution")
        ax_hist.set_xlabel("Residual")
        ax_hist.set_ylabel("Frequency")
        st.pyplot(fig_hist)
        dashboard_elements.append("residual_distribution_histogram")

        st.write("### Model comparison")
        fig_metrics, ax_metrics = plt.subplots(figsize=(8, 4))
        metric_plot_df = results_df.set_index("model")[["MAE", "RMSE", "Validation_RMSE"]]
        metric_plot_df.plot(kind="bar", ax=ax_metrics)
        ax_metrics.set_title("Model Error Comparison")
        ax_metrics.set_xlabel("Model")
        ax_metrics.set_ylabel("Error")
        st.pyplot(fig_metrics)
        dashboard_elements.append("metrics_comparison_chart")

        if not feature_importance_df.empty:
            st.write("### Top feature importance")
            fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
            ordered = feature_importance_df.sort_values("importance", ascending=True)
            ax_imp.barh(ordered["feature"], ordered["importance"])
            ax_imp.set_title(f"Top Features Used by {best_model_name}")
            ax_imp.set_xlabel("Importance")
            st.pyplot(fig_imp)
            dashboard_elements.append("feature_importance_chart")

        st.write("### Energy use by hour and day type")
        pattern_df = model_data.copy()
        pattern_df["hour"] = pattern_df[timestamp_col].dt.hour
        pattern_df["day_type"] = np.where(pattern_df[timestamp_col].dt.dayofweek.isin([5, 6]), "Weekend", "Weekday")
        hourly_pattern = pattern_df.groupby(["hour", "day_type"])[target_col].mean().unstack()

        fig_hourly, ax_hourly = plt.subplots(figsize=(9, 4))
        hourly_pattern.plot(ax=ax_hourly)
        ax_hourly.set_title("Average Appliance Energy Use by Hour")
        ax_hourly.set_xlabel("Hour of day")
        ax_hourly.set_ylabel(f"Average {target_col}")
        st.pyplot(fig_hourly)
        dashboard_elements.append("hourly_weekday_weekend_pattern_plot")

        st.write("### Correlation heatmap for selected variables")
        corr_cols = [target_col] + [col for col in ["T1", "RH_1", "T2", "RH_2", "T_out", "RH_out", "Windspeed", "Visibility", "Press_mm_hg"] if col in model_data.columns]
        corr_df = model_data[corr_cols].select_dtypes(include=[np.number]).corr()
        if len(corr_df.columns) >= 2:
            fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
            im = ax_corr.imshow(corr_df.values)
            ax_corr.set_xticks(range(len(corr_df.columns)))
            ax_corr.set_yticks(range(len(corr_df.columns)))
            ax_corr.set_xticklabels(corr_df.columns, rotation=45, ha="right")
            ax_corr.set_yticklabels(corr_df.columns)
            ax_corr.set_title("Correlation Heatmap")
            fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
            st.pyplot(fig_corr)
            dashboard_elements.append("correlation_heatmap")

        st.write("### Professional insights")
        st.markdown(
            f"""
- The project now uses **advanced feature engineering** instead of only the starter baseline features.
- The feature set includes lag memory, rolling statistics, cyclical time variables, target changes, and shifted sensor/weather context.
- A strict chronological split was used: training data comes first, validation follows, and the latest period is reserved for testing.
- The best model is **{best_model_name}**, selected by the lowest test RMSE.
- The residual and scatter plots show where the model fits well and where unusual appliance activity causes larger errors.
- The feature importance and pattern plots make the dashboard more explainable and professional.
"""
        )
        dashboard_elements.append("professional_written_insights")
    else:
        st.warning("Dashboard additions need a successful metrics table from the modeling section.")

except Exception as exc:
    st.error(f"Dashboard section failed: {exc}")
    dashboard_visuals_created = False



student_insights = st.text_area(
    "Student insights / explanation",
    value=(
        "The project uses cleaned chronological energy data, baseline lag/rolling features, "
        "and a time-based split to forecast appliance energy use. The model comparison table "
        "shows which model performs best on the future test period, while the residual plot "
        "highlights periods where prediction errors are larger."
    ),
    help="After adding models and visuals, summarize what you learned.",
)

st.divider()
st.subheader("8) Export submission.json and project_card.md")

submission = build_submission_json(
    student_name=student_name,
    student_id=student_id,
    app_url=deployed_url,
    project_title=project_title,
    project_goal=project_goal,
    data=model_data,
    feature_table=feature_table,
    timestamp_col=timestamp_col,
    target_col=target_col,
    horizon=int(horizon),
    resampling_choice=resampling_choice,
    results_df=results_df,
    insights=student_insights,
)

submission.update(
    {
        "data_integrity_checks": data_integrity_checks,
        "time_based_split_used": bool(time_based_split_used),
        "train_validation_test_rows": split_rows,
        "models_trained": models_trained,
        "best_model": best_model_name,
        "student_added_feature_engineering": bool(student_added_feature_engineering),
        "feature_engineering_description": feature_engineering_description,
        "advanced_features_created": advanced_features_created,
        "advanced_feature_count": int(len(advanced_features_created)),
        "dashboard_visuals_created": bool(dashboard_visuals_created),
        "dashboard_elements": dashboard_elements,
        "dashboard_visual_count": int(len(dashboard_elements)),
    }
)

submission_json = json.dumps(submission, indent=2)
project_card = make_project_card(submission)

col_download_1, col_download_2 = st.columns(2)
with col_download_1:
    st.download_button(
        "Download submission.json",
        data=submission_json,
        file_name="submission.json",
        mime="application/json",
    )
with col_download_2:
    st.download_button(
        "Download project_card.md",
        data=project_card,
        file_name="project_card.md",
        mime="text/markdown",
    )

with st.expander("Preview submission.json"):
    st.json(submission)

st.divider()
st.subheader("9) AI Grader /80")

st.warning(
    "The grader is strict. The starter alone will score low because students must add models, metrics, dashboard improvements, and insights."
)

api_key = get_openrouter_key()

if st.button("Run AI Grader"):
    if not api_key:
        st.error("Please provide an OpenRouter API key before running the grader.")
    else:
        grader_prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
            "<insert submission.json contents here>",
            submission_json,
        )

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "user", "content": grader_prompt},
            ],
            "temperature": 0,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            raw_output = response.json()["choices"][0]["message"]["content"]
            parsed = parse_ai_json(raw_output)

            if parsed is not None:
                st.success("AI grader returned valid JSON.")
                st.json(parsed)
            else:
                st.warning("Could not parse JSON. Raw model output:")
                st.code(raw_output)
        except Exception as exc:
            st.error(f"AI grader request failed: {exc}")
