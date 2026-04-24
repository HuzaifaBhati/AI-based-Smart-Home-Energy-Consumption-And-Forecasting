import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Smart Home Energy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        "../data/cleaned_hourly.csv", parse_dates=["datetime"], index_col="datetime"
    )
    predictions = pd.read_csv(
        "../data/baseline_predictions.csv",
        parse_dates=["datetime"],
        index_col="datetime",
    )
    cost = pd.read_csv(
        "../data/cost_analysis.csv", parse_dates=["datetime"], index_col="datetime"
    )
    anomaly = pd.read_csv(
        "../data/anomaly_detection.csv", parse_dates=["datetime"], index_col="datetime"
    )
    schedule = pd.read_csv("../data/appliance_schedule.csv")
    rec = pd.read_csv("../data/recommendations.csv")
    return df, predictions, cost, anomaly, schedule, rec


df, predictions, cost, anomaly, schedule, rec = load_data()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=80)
st.sidebar.title("⚡ Smart Home Energy")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "📋 Navigate to",
    [
        "🏠 Overview",
        "📈 Energy Forecasting",
        "💰 Cost Analysis",
        "🚨 Anomaly Detection",
        "🏠 Appliance Scheduler",
        "🧠 Recommendations",
    ],
)

st.sidebar.markdown("---")

# Date filter
st.sidebar.subheader("📅 Date Filter")
min_date = df.index.min().date()
max_date = df.index.max().date()

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2010-01-01").date(),
    min_value=min_date,
    max_value=max_date,
)
end_date = st.sidebar.date_input(
    "End Date", value=max_date, min_value=min_date, max_value=max_date
)

# Filter data
df_filtered = df.loc[str(start_date) : str(end_date)]
cost_filtered = cost.loc[str(start_date) : str(end_date)]
anomaly_filtered = anomaly.loc[str(start_date) : str(end_date)]

st.sidebar.markdown("---")
st.sidebar.markdown("**🔧 Model Used:** XGBoost")
st.sidebar.markdown("**📊 Dataset:** UCI Household Power")
st.sidebar.markdown("**📅 Data Range:** 2006–2010")

# ─────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────
if page == "🏠 Overview":
    st.title("⚡ Smart Home Energy Dashboard")
    st.markdown("### AI-Based Energy Consumption Forecasting & Optimization")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    avg_power = df_filtered["Global_active_power"].mean()
    total_kwh = df_filtered["Global_active_power"].sum()
    total_cost = cost_filtered["hourly_cost"].sum()
    total_savings = cost_filtered["savings"].sum()

    col1.metric("⚡ Avg Power", f"{avg_power:.2f} kW", "Hourly Average")
    col2.metric("🔋 Total Usage", f"{total_kwh:,.0f} kWh", "Selected Period")
    col3.metric("💰 Total Cost", f"₹{total_cost:,.0f}", "Electricity Bill")
    col4.metric("💚 Total Savings", f"₹{total_savings:,.0f}", "From Optimization")

    st.markdown("---")

    # Energy consumption over time
    st.subheader("📈 Energy Consumption Over Time")
    fig = px.line(
        df_filtered,
        y="Global_active_power",
        title="Hourly Global Active Power Consumption",
        labels={"Global_active_power": "Power (kW)", "datetime": "Date"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig.update_layout(height=350, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Average Usage by Hour")
        hourly_avg = df_filtered.groupby(df_filtered.index.hour)[
            "Global_active_power"
        ].mean()
        peak_hours = [19, 20, 21]
        colors = ["red" if h in peak_hours else "#1f77b4" for h in hourly_avg.index]
        fig2 = go.Figure(
            go.Bar(
                x=hourly_avg.index,
                y=hourly_avg.values,
                marker_color=colors,
                hovertemplate="Hour %{x}: %{y:.2f} kW<extra></extra>",
            )
        )
        fig2.update_layout(
            title="Avg Power by Hour (Red = Peak)",
            xaxis_title="Hour",
            yaxis_title="kW",
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("📊 Average Usage by Month")
        monthly_avg = df_filtered.groupby(df_filtered.index.month)[
            "Global_active_power"
        ].mean()
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        fig3 = go.Figure(
            go.Bar(
                x=[months[m - 1] for m in monthly_avg.index],
                y=monthly_avg.values,
                marker_color="mediumseagreen",
                hovertemplate="%{x}: %{y:.2f} kW<extra></extra>",
            )
        )
        fig3.update_layout(
            title="Avg Power by Month",
            xaxis_title="Month",
            yaxis_title="kW",
            height=350,
        )
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 2 — ENERGY FORECASTING
# ─────────────────────────────────────────
elif page == "📈 Energy Forecasting":
    st.title("📈 Energy Consumption Forecasting")
    st.markdown("---")

    # Model comparison metrics
    st.subheader("🏆 Model Performance Comparison")
    metrics_df = pd.DataFrame(
        {
            "Model": ["Random Forest", "XGBoost", "LSTM v1", "LSTM v2"],
            "MAE": [0.3236, 0.3199, 0.3354, 0.3238],
            "RMSE": [0.4723, 0.4657, 0.4954, 0.4789],
            "R²": [0.5821, 0.5937, 0.5404, 0.5704],
            "MAPE %": [43.69, 42.71, 43.58, 39.84],
        }
    )
    st.dataframe(
        metrics_df.style.highlight_max(subset=["R²"], color="lightgreen").highlight_min(
            subset=["MAE", "RMSE", "MAPE %"], color="lightgreen"
        ),
        use_container_width=True,
    )

    st.markdown("---")

    # Predictions vs Actual
    st.subheader("📊 XGBoost — Actual vs Predicted")

    pred_filtered = predictions.loc[str(start_date) : str(end_date)]

    fig4 = go.Figure()
    fig4.add_trace(
        go.Scatter(
            x=pred_filtered.index,
            y=pred_filtered["actual"],
            name="Actual",
            line=dict(color="steelblue", width=1),
        )
    )
    fig4.add_trace(
        go.Scatter(
            x=pred_filtered.index,
            y=pred_filtered["xgb_predicted"],
            name="XGBoost Predicted",
            line=dict(color="red", width=1),
            opacity=0.7,
        )
    )
    fig4.update_layout(
        title="Actual vs Predicted Energy Consumption",
        xaxis_title="Date",
        yaxis_title="Power (kW)",
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Scatter plot
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Prediction Accuracy")
        fig5 = px.scatter(
            x=pred_filtered["actual"],
            y=pred_filtered["xgb_predicted"],
            labels={"x": "Actual (kW)", "y": "Predicted (kW)"},
            title="Actual vs Predicted Scatter Plot",
            opacity=0.4,
            color_discrete_sequence=["steelblue"],
        )
        # Perfect prediction line
        max_val = pred_filtered["actual"].max()
        fig5.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", dash="dash"),
            )
        )
        fig5.update_layout(height=400)
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader("📉 Prediction Error Distribution")
        errors = pred_filtered["actual"] - pred_filtered["xgb_predicted"]
        fig6 = px.histogram(
            errors,
            nbins=50,
            title="Prediction Error Distribution",
            labels={"value": "Error (kW)", "count": "Frequency"},
            color_discrete_sequence=["coral"],
        )
        fig6.update_layout(height=400)
        st.plotly_chart(fig6, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 3 — COST ANALYSIS
# ─────────────────────────────────────────
elif page == "💰 Cost Analysis":
    st.title("💰 Electricity Cost Analysis")
    st.markdown("---")

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Cost", f"₹{cost_filtered['hourly_cost'].sum():,.2f}")
    col2.metric("💚 Optimized Cost", f"₹{cost_filtered['optimized_cost'].sum():,.2f}")
    col3.metric(
        "🎯 Total Savings",
        f"₹{cost_filtered['savings'].sum():,.2f}",
        f"{(cost_filtered['savings'].sum()/cost_filtered['hourly_cost'].sum())*100:.1f}% saved",
    )

    st.markdown("---")

    # Monthly cost comparison
    st.subheader("📊 Monthly Cost — Actual vs Optimized")
    cost_filtered["year_month"] = cost_filtered.index.to_period("M").astype(str)
    monthly_cost = (
        cost_filtered.groupby("year_month")
        .agg(
            actual_cost=("hourly_cost", "sum"),
            optimized_cost=("optimized_cost", "sum"),
            savings=("savings", "sum"),
        )
        .reset_index()
    )

    fig7 = go.Figure()
    fig7.add_trace(
        go.Bar(
            x=monthly_cost["year_month"],
            y=monthly_cost["actual_cost"],
            name="Actual Cost",
            marker_color="coral",
        )
    )
    fig7.add_trace(
        go.Bar(
            x=monthly_cost["year_month"],
            y=monthly_cost["optimized_cost"],
            name="Optimized Cost",
            marker_color="mediumseagreen",
        )
    )
    fig7.update_layout(
        barmode="group",
        height=400,
        xaxis_title="Month",
        yaxis_title="Cost (₹)",
        hovermode="x unified",
    )
    st.plotly_chart(fig7, use_container_width=True)

    # Savings over time
    st.subheader("💚 Monthly Savings from Optimization")
    fig8 = px.bar(
        monthly_cost,
        x="year_month",
        y="savings",
        title="Monthly Savings",
        labels={"year_month": "Month", "savings": "Savings (₹)"},
        color_discrete_sequence=["steelblue"],
    )
    fig8.update_layout(height=350)
    st.plotly_chart(fig8, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 4 — ANOMALY DETECTION
# ─────────────────────────────────────────
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection")
    st.markdown("---")

    # KPIs
    col1, col2, col3 = st.columns(3)
    total_anomalies = anomaly_filtered["is_anomaly"].sum()
    anomaly_pct = anomaly_filtered["is_anomaly"].mean() * 100
    max_spike = anomaly_filtered[anomaly_filtered["is_anomaly"] == 1][
        "Global_active_power"
    ].max()

    col1.metric("🚨 Anomalies Found", f"{total_anomalies:,}")
    col2.metric("📊 Anomaly Rate", f"{anomaly_pct:.2f}%")
    col3.metric("⚡ Max Spike", f"{max_spike:.2f} kW")

    st.markdown("---")

    # Anomaly plot
    st.subheader("📈 Energy Consumption with Anomalies Highlighted")
    normal = anomaly_filtered[anomaly_filtered["is_anomaly"] == 0]
    spikes = anomaly_filtered[anomaly_filtered["is_anomaly"] == 1]

    fig9 = go.Figure()
    fig9.add_trace(
        go.Scatter(
            x=normal.index,
            y=normal["Global_active_power"],
            mode="lines",
            name="Normal",
            line=dict(color="steelblue", width=0.8),
        )
    )
    fig9.add_trace(
        go.Scatter(
            x=spikes.index,
            y=spikes["Global_active_power"],
            mode="markers",
            name="Anomaly",
            marker=dict(color="red", size=6, symbol="x"),
        )
    )
    fig9.update_layout(
        title="Anomaly Detection — Energy Consumption",
        xaxis_title="Date",
        yaxis_title="Power (kW)",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig9, use_container_width=True)

    # Anomaly table
    st.subheader("📋 Anomaly Details")
    anomaly_table = spikes[["Global_active_power", "z_score"]].copy()
    anomaly_table.columns = ["Power (kW)", "Z-Score"]
    anomaly_table = anomaly_table.sort_values("Z-Score", ascending=False)
    st.dataframe(anomaly_table.head(20), use_container_width=True)

# ─────────────────────────────────────────
# PAGE 5 — APPLIANCE SCHEDULER
# ─────────────────────────────────────────
elif page == "🏠 Appliance Scheduler":
    st.title("🏠 Smart Appliance Scheduler")
    st.markdown("---")

    st.info("💡 Schedule your appliances during off-peak hours to save money!")

    # Peak hours info
    col1, col2 = st.columns(2)
    with col1:
        st.error(
            "🔴 **Peak Hours (Expensive):** 19:00 – 22:00\n\nAvoid running heavy appliances!"
        )
    with col2:
        st.success(
            "🟢 **Best Hours (Cheap):** 10:00 – 14:00 & 23:00 – 05:00\n\nIdeal for heavy appliances!"
        )

    st.markdown("---")

    # Appliance schedule table
    st.subheader("📋 Appliance Scheduling Recommendations")
    st.dataframe(
        schedule.style.highlight_max(subset=["Savings (₹)"], color="lightgreen"),
        use_container_width=True,
    )

    st.markdown("---")

    # Savings chart
    st.subheader("💰 Daily Savings per Appliance")
    fig10 = px.bar(
        schedule,
        x="Appliance",
        y="Savings (₹)",
        title="Potential Daily Savings by Scheduling Appliances Optimally",
        color="Savings (₹)",
        color_continuous_scale="Greens",
        labels={"Savings (₹)": "Daily Savings (₹)"},
    )
    fig10.update_layout(height=400)
    st.plotly_chart(fig10, use_container_width=True)

    # Total savings
    st.success(
        f"💰 **Total Daily Savings if all appliances scheduled optimally: "
        f"₹{schedule['Savings (₹)'].sum():.2f}**"
    )

# ─────────────────────────────────────────
# PAGE 6 — RECOMMENDATIONS
# ─────────────────────────────────────────
elif page == "🧠 Recommendations":
    st.title("🧠 Smart Energy Recommendations")
    st.markdown("---")

    st.info(
        "💡 These recommendations are generated by AI based on your energy usage patterns!"
    )

    for _, row in rec.iterrows():
        with st.expander(f"{row['Category']} — Click to expand", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.warning(f"⚠️ **Issue:** {row['Issue']}")
                st.success(f"✅ **Recommendation:** {row['Recommendation']}")
            with col2:
                st.metric("💰 Est. Monthly Savings", row["Est. Savings"])

    st.markdown("---")
    st.subheader("📊 Estimated Savings Summary")

    savings_data = pd.DataFrame(
        {
            "Category": rec["Category"].values,
            "Min Savings": [200, 100, 150, 100, 200, 500][: len(rec)],
            "Max Savings": [400, 200, 300, 250, 500, 1000][: len(rec)],
        }
    )

    total_min = sum([200, 100, 150, 100, 200, 500][: len(rec)])
    total_max = sum([400, 200, 300, 250, 500, 1000][: len(rec)])

    fig11 = go.Figure()
    fig11.add_trace(
        go.Bar(
            x=savings_data["Category"],
            y=savings_data["Max Savings"],
            name="Max Savings",
            marker_color="mediumseagreen",
        )
    )
    fig11.add_trace(
        go.Bar(
            x=savings_data["Category"],
            y=savings_data["Min Savings"],
            name="Min Savings",
            marker_color="lightgreen",
        )
    )
    fig11.update_layout(
        barmode="overlay",
        title="Estimated Monthly Savings by Category (₹)",
        xaxis_title="Category",
        yaxis_title="Savings (₹)",
        height=400,
    )
    st.plotly_chart(fig11, use_container_width=True)

    total_min = sum([200, 100, 150, 100, 200, 500])
    total_max = sum([400, 200, 300, 250, 500, 1000])
    st.success(
        f"🎯 **Total Potential Monthly Savings: ₹{total_min:,} – ₹{total_max:,}**"
    )
