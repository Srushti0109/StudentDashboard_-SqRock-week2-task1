import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Student AI Dashboard", layout="wide")

# --- IMPROVED CSS FOR UNIFORM CARDS ---
st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: #f8f9fb;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #eceef1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        align-items: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Student Performance Analytics & AI Prediction")
st.markdown("---")

@st.cache_resource
def load_data():
    # Priority 1: Try to load your real file
    try:
        df = pd.read_csv('StudentsPerformance.csv', encoding='latin1')
        df.columns = [c.replace('"', '').strip().lower() for c in df.columns]
        df = df.rename(columns={'math score': 'math', 'reading score': 'reading', 'writing score': 'writing'})
    except:
        # Priority 2: Fallback data
        data = {
            'gender': ['female', 'female', 'male', 'male', 'female', 'male'] * 50,
            'prep': ['none', 'completed', 'none', 'completed', 'none', 'completed'] * 50,
            'math': [72, 69, 90, 47, 76, 85] * 50,
            'reading': [72, 90, 95, 57, 78, 88] * 50,
            'writing': [74, 88, 93, 44, 75, 82] * 50
        }
        df = pd.DataFrame(data)

    for c in ['math', 'reading', 'writing']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna()

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Dashboard Filters")
gen_list = df['gender'].unique()
selected_gen = st.sidebar.multiselect("Select Gender", gen_list, default=gen_list)
data = df[df['gender'].isin(selected_gen)]

# --- TOP METRICS (Standardized) ---
m1, m2, m3 = st.columns(3)
m1.metric("Avg Math Score", f"{data['math'].mean():.1f}")
m2.metric("Avg Reading Score", f"{data['reading'].mean():.1f}")
m3.metric("Avg Writing Score", f"{data['writing'].mean():.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# --- VISUAL ANALYTICS ---
tab1, tab2 = st.tabs(["📊 Distribution & Correlation", "🌡️ Heatmap Analysis"])

with tab1:
    left, right = st.columns(2)
    with left:
        fig_hist = px.histogram(data, x="math", color="gender", nbins=20, 
                               title="Math Score Distribution", 
                               color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_hist, use_container_width=True)
    with right:
        fig_scatter = px.scatter(data, x="reading", y="writing", color="prep",
                                title="Reading vs Writing Correlation",
                                trendline="ols") 
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("Subject Correlation Matrix")
    corr = data[['math', 'reading', 'writing']].corr()
    fig_heat = px.imshow(corr, text_auto=True, aspect="auto", 
                        color_continuous_scale='RdBu_r',
                        labels=dict(color="Correlation"))
    st.plotly_chart(fig_heat, use_container_width=True)

# --- AI PREDICTION SECTION ---
st.markdown("---")
st.subheader("🔮 Research Insight: AI Math Predictor")
st.info("This section uses a Linear Regression model to estimate Math potential based on verbal/literacy skills.")

p_col1, p_col2 = st.columns(2)
with p_col1:
    r_val = st.slider("Reading Score Input", 0, 100, 70)
with p_col2:
    w_val = st.slider("Writing Score Input", 0, 100, 70)

if len(data) > 1:
    model = LinearRegression().fit(data[['reading', 'writing']], data['math'])
    prediction = model.predict([[r_val, w_val]])[0]
    final_score = max(0, min(100, prediction))
    
    st.success(f"### Predicted Math Competency: **{final_score:.2f}%**")
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = final_score,
        title = {'text': "Confidence Level"},
        gauge = {'axis': {'range': [None, 100]},
                 'bar': {'color': "#2ecc71"},
                 'steps' : [
                     {'range': [0, 40], 'color': "#ff7675"},
                     {'range': [40, 70], 'color': "#ffeaa7"},
                     {'range': [70, 100], 'color': "#55efc4"}]}))
    st.plotly_chart(fig_gauge)