# ============================================================
# STAGE 6: Streamlit Web App
#
# A professional multi-page app with:
#   Page 1 → Home (project overview + dataset stats)
#   Page 2 → Predict (upload CSV → get tumor predictions)
#   Page 3 → Biomarkers (view discovered genes + charts)
#   Page 4 → About (methodology explanation)
#
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# PAGE CONFIG — must be the VERY FIRST streamlit command
# Sets the browser tab title, icon, and layout
# layout="wide" uses full screen width instead of narrow center
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------
# CUSTOM CSS
# Streamlit allows injecting CSS via st.markdown with unsafe_allow_html
# This styles our cards, badges, and headings
# ---------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Metric cards */
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3148;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Prediction result box */
    .prediction-box {
        background: #1a2e1a;
        border: 2px solid #16a34a;
        border-radius: 10px;
        padding: 16px 24px;
        margin: 8px 0;
    }
    .prediction-text {
        color: #86efac;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        color: #a78bfa;
        font-size: 1.4rem;
        font-weight: 700;
        border-bottom: 2px solid #2d3148;
        padding-bottom: 8px;
        margin: 20px 0 15px 0;
    }

    /* Info badge */
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        background: #3b0764;
        color: #d8b4fe;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# LOAD MODELS — cached so they load only ONCE
#
# @st.cache_resource tells Streamlit:
# "Load this once and keep it in memory — don't reload on every rerun"
# Without this, every button click would reload the model (~2 seconds)
# ---------------------------------------------------------------
@st.cache_resource
def load_all_models():
    """Load all saved models and encoders from disk."""
    try:
        model         = pickle.load(open("model/model.pkl",         "rb"))
        pca           = pickle.load(open("model/pca.pkl",           "rb"))
        encoder       = pickle.load(open("model/encoder.pkl",       "rb"))
        scaler        = pickle.load(open("model/scaler.pkl",        "rb"))
        feature_names = pickle.load(open("model/feature_names.pkl", "rb"))
        return model, pca, encoder, scaler, feature_names, None
    except FileNotFoundError as e:
        # Return None values if models not found
        # The app will show a friendly error instead of crashing
        return None, None, None, None, None, str(e)

model, pca, encoder, scaler, feature_names, load_error = load_all_models()


# ---------------------------------------------------------------
# SIDEBAR NAVIGATION
#
# st.sidebar puts content in the left panel
# st.radio creates a selection list — user picks one option
# The selected page name drives which content we show below
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Brain Tumor AI")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔬 Predict", "🧬 Biomarkers", "📖 About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Dataset:** GSE50161")
    st.markdown("**Platform:** Affymetrix HG-U133+2")
    st.markdown("**Samples:** 130")
    st.markdown("**Genes:** 54,613")

    # Show model status
    st.markdown("---")
    if load_error:
        st.error("⚠️ Models not loaded")
        st.caption("Run train scripts first")
    else:
        st.success("✅ Models loaded")


# ================================================================
# PAGE 1: HOME
# ================================================================
if page == "🏠 Home":

    st.title("🧠 Brain Tumor Classification System")
    st.markdown("#### Using Gene Expression Data & Machine Learning")
    st.markdown("---")

    # Project description
    st.markdown("""
    This system predicts brain tumor subtypes from **gene expression microarray data**
    using a machine learning pipeline built on the **GSE50161** dataset from NCBI GEO.
    """)

    # Key stats in metric cards (4 columns)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">130</div>
            <div class="metric-label">Total Samples</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">54,613</div>
            <div class="metric-label">Gene Features</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">5</div>
            <div class="metric-label">Tumor Classes</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">50</div>
            <div class="metric-label">PCA Components</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Two-column layout: class info + pipeline
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">🏷️ Tumor Types</div>', unsafe_allow_html=True)

        tumor_info = {
            "Ependymoma":             {"count": 46,  "color": "🟣", "desc": "Arises from ependymal cells lining brain ventricles"},
            "Glioblastoma":           {"count": 34,  "color": "🔵", "desc": "Most aggressive malignant brain tumor (Grade IV)"},
            "Medulloblastoma":        {"count": 22,  "color": "🟢", "desc": "Most common malignant pediatric brain tumor"},
            "Pilocytic Astrocytoma":  {"count": 15,  "color": "🟡", "desc": "Slow-growing, usually benign (Grade I)"},
            "Normal":                 {"count": 13,  "color": "⚪", "desc": "Healthy brain tissue (control group)"},
        }

        for name, info in tumor_info.items():
            st.markdown(f"""
            **{info['color']} {name}** — {info['count']} samples
            <br><small style='color:#94a3b8'>{info['desc']}</small>
            """, unsafe_allow_html=True)
            st.markdown("")

    with col_right:
        st.markdown('<div class="section-header">⚙️ ML Pipeline</div>', unsafe_allow_html=True)

        steps = [
            ("1", "Load GSE50161 dataset", "130 samples × 54,613 genes"),
            ("2", "Remove AFFX control probes", "62 probes removed"),
            ("3", "Label encoding", "Text → numbers (0–4)"),
            ("4", "Standard scaling", "Mean=0, Std=1 per gene"),
            ("5", "Train/test split (80/20)", "Stratified by tumor type"),
            ("6", "PCA dimensionality reduction", "54,613 → 50 components"),
            ("7", "Random Forest (200 trees)", "class_weight='balanced'"),
            ("8", "Evaluation + Biomarker Discovery", "ANOVA + RF importance"),
        ]

        for num, step, detail in steps:
            st.markdown(f"""
            <div style='display:flex; gap:10px; margin-bottom:8px; align-items:flex-start'>
                <div style='background:#6d28d9; color:white; border-radius:50%;
                            width:24px; height:24px; display:flex; align-items:center;
                            justify-content:center; font-size:0.75rem; flex-shrink:0;
                            margin-top:2px'>{num}</div>
                <div>
                    <div style='color:#e2e8f0; font-size:0.9rem'>{step}</div>
                    <div style='color:#64748b; font-size:0.78rem'>{detail}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Show class distribution chart if outputs exist
    if os.path.exists("outputs/class_distribution.png"):
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
        st.image("outputs/class_distribution.png", use_column_width=True)


# ================================================================
# PAGE 2: PREDICT
# ================================================================
elif page == "🔬 Predict":

    st.title("🔬 Tumor Type Prediction")
    st.markdown("Upload gene expression data to predict brain tumor subtype.")
    st.markdown("---")

    if load_error:
        st.error(f"❌ Models not found. Please run the training scripts first.")
        st.code("python 2_preprocess.py\npython 3_train_model.py")
        st.stop()

    # Two tabs: Demo and Upload
    # st.tabs creates a tabbed interface
    tab1, tab2 = st.tabs(["🎯 Try Demo Data", "📁 Upload Your CSV"])

    # ---- TAB 1: DEMO ----
    with tab1:
        st.markdown("### Test with sample data from the dataset")
        st.info("This uses the first 5 rows of the original dataset as a quick demo.")

        n_samples = st.slider(
            "How many samples to predict?",
            min_value=1, max_value=10, value=3
        )
        # st.slider creates a draggable slider
        # min_value, max_value set the range
        # value sets the default position

        if st.button("▶️ Run Demo Prediction", type="primary"):
            try:
                # Load original data
                df_demo = pd.read_csv("data/Brain_GSE50161.csv")
                true_labels = df_demo['type'].values[:n_samples]

                # Preprocess same way as training
                df_demo = df_demo.drop(columns=['samples', 'type'])
                df_demo = df_demo.loc[:, ~df_demo.columns.str.startswith("AFFX")]
                df_demo = df_demo.iloc[:n_samples]

                # Scale → PCA → Predict
                # This is the EXACT same pipeline as training
                X_scaled = scaler.transform(df_demo)
                X_pca    = pca.transform(X_scaled)
                preds    = model.predict(X_pca)
                probs    = model.predict_proba(X_pca)   # confidence scores
                labels   = encoder.inverse_transform(preds)

                st.success("✅ Prediction Complete!")
                st.markdown("---")

                # Show results
                for i in range(n_samples):
                    confidence = probs[i].max() * 100  # highest probability class

                    # Color based on confidence
                    conf_color = "#86efac" if confidence > 85 else "#fde68a" if confidence > 60 else "#fca5a5"

                    col_a, col_b, col_c = st.columns([1, 2, 2])
                    with col_a:
                        st.markdown(f"**Sample {i+1}**")
                    with col_b:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-text">🧬 {labels[i]}</div>
                        </div>""", unsafe_allow_html=True)
                    with col_c:
                        st.markdown(f"""
                        <div style='padding:12px; background:#1e2130; border-radius:8px;'>
                            <span style='color:{conf_color}; font-weight:600'>
                                {confidence:.1f}% confidence
                            </span>
                            <br><small style='color:#94a3b8'>
                                True label: {true_labels[i]}
                            </small>
                        </div>""", unsafe_allow_html=True)

                # Probability breakdown for last sample
                st.markdown("---")
                st.markdown("#### Probability Breakdown (Sample 1)")
                st.caption("How confident the model is for each tumor type:")

                prob_df = pd.DataFrame({
                    'Tumor Type': encoder.classes_,
                    'Probability': probs[0]
                }).sort_values('Probability', ascending=True)

                fig, ax = plt.subplots(figsize=(7, 3))
                colors_bar = ['#dc2626' if p == probs[0].max() else '#2d3148'
                              for p in prob_df['Probability']]
                ax.barh(prob_df['Tumor Type'], prob_df['Probability'],
                        color=colors_bar, edgecolor='none')
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.set_facecolor('#0f1117')
                fig.patch.set_facecolor('#0f1117')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.error(f"Error: {e}")

    # ---- TAB 2: UPLOAD ----
    with tab2:
        st.markdown("### Upload your own gene expression CSV")

        # Show format instructions
        with st.expander("📋 Required CSV Format"):
            st.markdown("""
            Your CSV must have **54,613 gene probe columns** matching the Affymetrix HG-U133+2 platform.

            - Each **row** = one patient/sample
            - Each **column** = one gene probe (e.g., `1007_s_at`, `1053_at`)
            - Values should be **log2-normalized** expression values
            - Optional columns `samples` and `type` will be automatically removed

            **Tip:** Files from NCBI GEO dataset GSE50161 work directly.
            """)

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload gene expression data in the format described above"
        )

        if uploaded_file is not None:
            # Read uploaded file
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: {df_upload.shape[0]} samples, {df_upload.shape[1]} columns")
            st.markdown("**Preview (first 3 rows, first 6 columns):**")
            st.dataframe(df_upload.iloc[:3, :6])

            # Remove metadata columns if present
            df_upload = df_upload.drop(columns=['samples', 'type'], errors='ignore')

            # Remove AFFX columns
            df_upload = df_upload.loc[:, ~df_upload.columns.str.startswith("AFFX")]

            # Check column compatibility
            upload_cols = set(df_upload.columns)
            model_cols  = set(feature_names)
            overlap     = upload_cols & model_cols
            missing     = model_cols - upload_cols

            st.info(f"Matching gene columns: {len(overlap)} / {len(model_cols)}")

            if len(missing) > 0:
                st.warning(f"⚠️ {len(missing)} gene columns missing. They will be filled with 0.")
                # Add missing columns as 0
                for col in missing:
                    df_upload[col] = 0

            # Reorder columns to match training order
            df_upload = df_upload.reindex(columns=feature_names, fill_value=0)

            if st.button("▶️ Predict All Samples", type="primary"):
                try:
                    with st.spinner("Running predictions..."):
                        X_scaled = scaler.transform(df_upload)
                        X_pca    = pca.transform(X_scaled)
                        preds    = model.predict(X_pca)
                        probs    = model.predict_proba(X_pca)
                        labels   = encoder.inverse_transform(preds)

                    st.success(f"✅ Predicted {len(labels)} samples!")

                    # Results table
                    results_df = pd.DataFrame({
                        'Sample'    : [f"Sample {i+1}" for i in range(len(labels))],
                        'Prediction': labels,
                        'Confidence': [f"{p.max()*100:.1f}%" for p in probs]
                    })
                    st.dataframe(results_df, use_container_width=True)

                    # Download button — lets user save results as CSV
                    csv_out = results_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=csv_out,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Prediction error: {e}")


# ================================================================
# PAGE 3: BIOMARKERS
# ================================================================
elif page == "🧬 Biomarkers":

    st.title("🧬 Biomarker Discovery")
    st.markdown("Genes identified as key discriminators of brain tumor subtypes.")
    st.markdown("---")

    # Sub-navigation with tabs
    btab1, btab2, btab3, btab4 = st.tabs([
        "🏆 Top Genes (RF)",
        "📊 ANOVA Results",
        "🌋 Volcano Plots",
        "🗺️ Heatmap"
    ])

    with btab1:
        st.markdown("### Top Genes — Random Forest + PCA Loading Method")
        st.markdown("""
        These genes were identified by combining **Random Forest feature importance**
        with **PCA component loadings**. The model found these genes most useful
        for making correct predictions across all 200 decision trees.
        """)

        if os.path.exists("outputs/biomarkers/top100_rf_importance.csv"):
            rf_df = pd.read_csv("outputs/biomarkers/top100_rf_importance.csv")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Top 20 Genes Table:**")
                st.dataframe(rf_df.head(20), use_container_width=True)
            with col2:
                if os.path.exists("outputs/biomarkers/top20_rf_genes.png"):
                    st.image("outputs/biomarkers/top20_rf_genes.png", use_column_width=True)

            st.download_button(
                "⬇️ Download Full Top-100 List",
                rf_df.to_csv(index=False),
                "top100_rf_biomarkers.csv",
                "text/csv"
            )
        else:
            st.warning("Run 4_biomarker_discovery.py first to generate these results.")

    with btab2:
        st.markdown("### ANOVA F-Test Results")
        st.markdown("""
        **ANOVA (Analysis of Variance)** statistically tests whether each gene's
        expression is significantly different across the 5 tumor groups.

        - **F-score**: Higher = more discriminative between groups
        - **p-value**: Lower = more statistically significant (threshold: 0.05)
        """)

        if os.path.exists("outputs/biomarkers/top100_anova.csv"):
            anova_df = pd.read_csv("outputs/biomarkers/top100_anova.csv")

            # Filter controls
            sig_count = (anova_df['p_value'] < 0.001).sum()
            st.metric("Highly Significant Genes (p < 0.001)", sig_count)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(anova_df.head(20), use_container_width=True)
            with col2:
                if os.path.exists("outputs/biomarkers/top20_anova_genes.png"):
                    st.image("outputs/biomarkers/top20_anova_genes.png", use_column_width=True)

        # Consensus biomarkers
        if os.path.exists("outputs/biomarkers/consensus_biomarkers.csv"):
            st.markdown("---")
            st.markdown("### ⭐ Consensus Biomarkers (RF + ANOVA Overlap)")
            st.markdown("""
            These genes appear in **both** the RF top-200 and ANOVA top-200.
            Agreement between two independent methods makes these the **strongest
            biomarker candidates** for further biological validation.
            """)
            consensus_df = pd.read_csv("outputs/biomarkers/consensus_biomarkers.csv")
            st.dataframe(consensus_df.head(20), use_container_width=True)
            st.download_button(
                "⬇️ Download Consensus Biomarkers",
                consensus_df.to_csv(index=False),
                "consensus_biomarkers.csv",
                "text/csv"
            )

    with btab3:
        st.markdown("### Volcano Plots — Differential Expression")
        st.markdown("""
        Volcano plots show **all 54,613 genes** simultaneously for each tumor type vs normal:
        - 🔴 **Red (top-right)**: Up-regulated — more active in tumor than normal
        - 🔵 **Blue (top-left)**: Down-regulated — less active in tumor than normal
        - **X-axis**: log2 Fold Change (how much expression changed)
        - **Y-axis**: -log10(p-value) (how statistically significant)
        """)

        if os.path.exists("outputs/biomarkers/volcano_plots.png"):
            st.image("outputs/biomarkers/volcano_plots.png", use_column_width=True)
        else:
            st.warning("Run 4_biomarker_discovery.py first.")

        # Show DEG tables per class
        st.markdown("---")
        st.markdown("### Differentially Expressed Genes per Tumor Type")

        tumor_options = ['ependymoma', 'glioblastoma', 'medulloblastoma', 'pilocytic_astrocytoma']
        selected_tumor = st.selectbox("Select tumor type:", tumor_options)
        # st.selectbox creates a dropdown menu

        deg_path = f"outputs/biomarkers/DEG_{selected_tumor}.csv"
        if os.path.exists(deg_path):
            deg_df = pd.read_csv(deg_path)

            up_count   = (deg_df['regulation'] == 'upregulated').sum()
            down_count = (deg_df['regulation'] == 'downregulated').sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 Up-regulated",   up_count)
            col2.metric("🔵 Down-regulated", down_count)
            col3.metric("Total Genes",        len(deg_df))

            # Filter by regulation type
            reg_filter = st.radio(
                "Show:",
                ["All", "Upregulated only", "Downregulated only"],
                horizontal=True
            )

            if reg_filter == "Upregulated only":
                show_df = deg_df[deg_df['regulation'] == 'upregulated']
            elif reg_filter == "Downregulated only":
                show_df = deg_df[deg_df['regulation'] == 'downregulated']
            else:
                show_df = deg_df

            st.dataframe(
                show_df[['gene', 'log2FC', 'p_value', 'regulation']].head(50),
                use_container_width=True
            )

    with btab4:
        st.markdown("### Biomarker Expression Heatmap")
        st.markdown("""
        Shows expression levels of the **top 25 biomarker genes** across all 130 samples.
        - Samples are sorted by tumor type
        - 🔴 **Red** = high expression, 🔵 **Blue** = low expression
        - Distinct color blocks per tumor type confirm these genes truly discriminate
        """)

        if os.path.exists("outputs/biomarkers/biomarker_heatmap.png"):
            st.image("outputs/biomarkers/biomarker_heatmap.png", use_column_width=True)
        else:
            st.warning("Run 4_biomarker_discovery.py first.")


# ================================================================
# PAGE 4: ABOUT
# ================================================================
elif page == "📖 About":

    st.title("📖 About This Project")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Project Goal")
        st.markdown("""
        This project classifies brain tumor subtypes using gene expression
        microarray data and identifies biomarker genes that distinguish
        each tumor type from normal brain tissue.

        **Clinical Relevance:** Accurate tumor subtype classification guides
        treatment decisions. Glioblastoma (Grade IV) requires aggressive
        treatment, while pilocytic astrocytoma (Grade I) may only need
        surgical removal.
        """)

        st.markdown("### 📦 Technologies")
        tech_data = {
            "Library": ["pandas", "numpy", "scikit-learn", "scipy", "matplotlib", "seaborn", "streamlit"],
            "Purpose": [
                "Data loading & manipulation",
                "Numerical computations",
                "PCA, Random Forest, scaling",
                "ANOVA F-test statistics",
                "Chart generation",
                "Statistical visualizations",
                "Web application"
            ]
        }
        st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 🔬 Methodology")

        methods = [
            ("Data Source",       "NCBI GEO — GSE50161 dataset (Affymetrix HG-U133+2)"),
            ("Preprocessing",     "AFFX probe removal, label encoding, StandardScaler"),
            ("Dimensionality",    "PCA: 54,613 genes → 50 principal components (~93% variance)"),
            ("Model",             "Random Forest (200 trees, balanced class weights)"),
            ("Validation",        "Stratified 5-fold cross-validation + held-out test set"),
            ("Biomarker Method 1","RF feature importance mapped through PCA loadings"),
            ("Biomarker Method 2","ANOVA F-test across 5 tumor groups"),
            ("Biomarker Method 3","Differential expression: log2FC + t-test vs normal"),
        ]

        for title, desc in methods:
            st.markdown(f"""
            <div style='margin-bottom:10px; padding:10px 14px;
                        background:#1e2130; border-radius:8px;
                        border-left:3px solid #7c3aed'>
                <div style='color:#e2e8f0; font-weight:600; font-size:0.9rem'>{title}</div>
                <div style='color:#94a3b8; font-size:0.82rem; margin-top:3px'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📁 Project File Structure")
    st.code("""
BrainTumorProject/
├── data/
│   ├── Brain_GSE50161.csv         ← Original dataset
│   └── processed/
│       ├── X_train.npy, X_test.npy
│       └── y_train.npy, y_test.npy
├── model/
│   ├── model.pkl                  ← Trained Random Forest
│   ├── pca.pkl                    ← Fitted PCA transformer
│   ├── scaler.pkl                 ← StandardScaler
│   ├── encoder.pkl                ← LabelEncoder
│   └── feature_names.pkl          ← Gene names list
├── outputs/
│   ├── class_distribution.png
│   ├── pca_variance_curve.png
│   ├── confusion_matrix.png
│   └── biomarkers/
│       ├── top100_rf_importance.csv
│       ├── top100_anova.csv
│       ├── consensus_biomarkers.csv
│       ├── volcano_plots.png
│       └── biomarker_heatmap.png
├── 1_explore_data.py
├── 2_preprocess.py
├── 3_train_model.py
├── 4_biomarker_discovery.py
├── app.py
└── requirements.txt
    """, language="")