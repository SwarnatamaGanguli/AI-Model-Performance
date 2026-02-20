import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="AI Performance Review", layout="wide")

# -------------------------------------------------
# CLEAN CONSULTING STYLE
# -------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: white;
}
h1 {
    color: #111111;
    font-weight: 700;
}
h2 {
    color: #111111;
    margin-top: 40px;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("ai_models_performance.csv")
df.columns = df.columns.str.strip()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
model_column = df.columns[0]

# Composite score
df["Composite_Score"] = df[numeric_cols].mean(axis=1)
df = df.sort_values("Composite_Score", ascending=False)
df["Rank"] = range(1, len(df)+1)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("AI Model Performance – Executive Review")

# -------------------------------------------------
# KEY METRICS
# -------------------------------------------------
st.markdown("### Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Top Model", df.iloc[0][model_column])
col2.metric("Best Score", round(df["Composite_Score"].max(),2))
col3.metric("Portfolio Average", round(df["Composite_Score"].mean(),2))
col4.metric("Total Models", len(df))

# -------------------------------------------------
# INSIGHT 1: PERFORMANCE CONCENTRATION
# -------------------------------------------------
st.markdown("## Insight 1: Performance is concentrated among top models")

top10 = df.head(10)
share = (top10["Composite_Score"].sum() / df["Composite_Score"].sum())*100

fig_rank = px.bar(
    top10,
    x=model_column,
    y="Composite_Score",
)

fig_rank.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font_color="#111111",
    showlegend=False
)

st.plotly_chart(fig_rank, use_container_width=True)

st.markdown(f"**Takeaway:** Top 10 models contribute {share:.1f}% of total performance strength, indicating strong dominance at the top tier.")

# -------------------------------------------------
# INSIGHT 2: COMPETITIVE GAP
# -------------------------------------------------
st.markdown("## Insight 2: Noticeable performance drop after top ranks")

fig_gap = px.line(
    df,
    y="Composite_Score",
    markers=True
)

fig_gap.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font_color="#111111"
)

st.plotly_chart(fig_gap, use_container_width=True)

gap = df.iloc[0]["Composite_Score"] - df.iloc[5]["Composite_Score"]

st.markdown(f"**Takeaway:** There is a {gap:.2f} point gap between rank 1 and rank 6, suggesting measurable competitive separation.")

# -------------------------------------------------
# INSIGHT 3: PERFORMANCE TRADE-OFF
# -------------------------------------------------
if len(numeric_cols) >= 2:
    st.markdown("## Insight 3: Speed vs Quality Trade-off")

    x_axis = numeric_cols[0]
    y_axis = numeric_cols[1]

    fig_quad = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        size="Composite_Score",
        hover_name=model_column
    )

    fig_quad.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#111111"
    )

    st.plotly_chart(fig_quad, use_container_width=True)

    st.markdown("**Takeaway:** Models vary significantly in performance balance, indicating optimization opportunities between efficiency and quality.")

# -------------------------------------------------
# INSIGHT 4: FORECASTING TREND
# -------------------------------------------------
st.markdown("## Insight 4: Expected Performance Trend")

X = np.array(df.index).reshape(-1,1)
y = df["Composite_Score"].values

model = LinearRegression()
model.fit(X,y)

future = np.array(range(len(df), len(df)+5)).reshape(-1,1)
prediction = model.predict(future)

forecast_df = pd.DataFrame({
    "Future Rank": range(len(df)+1, len(df)+6),
    "Predicted Score": prediction
})

fig_forecast = px.line(
    forecast_df,
    x="Future Rank",
    y="Predicted Score",
    markers=True
)

fig_forecast.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font_color="#111111"
)

st.plotly_chart(fig_forecast, use_container_width=True)

st.markdown("**Takeaway:** Performance trajectory suggests incremental improvement, but breakthrough gains appear limited without structural innovation.")

# -------------------------------------------------
# EXECUTIVE SUMMARY
# -------------------------------------------------
st.markdown("## Executive Summary")

leader = df.iloc[0][model_column]
median_score = df["Composite_Score"].median()
volatility = df["Composite_Score"].std()

st.markdown(f"""
The AI model portfolio demonstrates clear tier differentiation.  
**{leader}** leads the competitive landscape with a measurable margin.

Performance is concentrated among the top performers, while mid-tier models
display optimization potential. Competitive separation is evident after rank five,
suggesting opportunity for strategic consolidation.

Future gains are expected to be incremental unless innovation shifts the performance frontier.
""")

# -------------------------------------------------
# PDF REPORT
# -------------------------------------------------
st.markdown("## Download Executive Report")

def create_pdf():
    filename = "AI_Executive_Report.pdf"
    doc = SimpleDocTemplate(filename)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("AI Performance Executive Report", styles["Title"]))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(f"Top Model: {leader}", styles["Normal"]))
    elements.append(Paragraph(f"Best Score: {df['Composite_Score'].max():.2f}", styles["Normal"]))
    elements.append(Paragraph(f"Portfolio Average: {df['Composite_Score'].mean():.2f}", styles["Normal"]))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("""
Conclusion:
Performance concentration at the top suggests competitive dominance,
while mid-tier optimization represents the largest opportunity area.
""", styles["Normal"]))

    doc.build(elements)
    return filename

if st.button("Generate PDF"):
    file = create_pdf()
    with open(file, "rb") as f:
        st.download_button("Download PDF", f, file_name=file)