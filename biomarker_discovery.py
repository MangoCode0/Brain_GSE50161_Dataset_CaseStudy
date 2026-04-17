# ============================================================
# STAGE 5: Biomarker Discovery
#
# Goal: Find which specific GENES are most important for
#       distinguishing brain tumor types.
#
# This is the core research contribution of your project.
# Three methods used:
#   Method 1 → Random Forest Feature Importance (via PCA loadings)
#   Method 2 → ANOVA F-test (statistical significance)
#   Method 3 → Differential Expression per tumor type
#
# Output:
#   - Top biomarker genes overall
#   - Top biomarker genes per tumor class
#   - Up/down regulated genes vs Normal
#   - All results saved as CSV (publishable format)
# ============================================================

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats   # for ANOVA F-test statistical calculations

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/biomarkers", exist_ok=True)


# ---------------------------------------------------------------
# STEP 1: Load everything we need
#
# We need:
#   model        → to extract feature importances
#   pca          → to map importances back to original genes
#   feature_names→ actual gene names (54,613 of them)
#   encoder      → to convert class numbers back to names
#   full dataset → for differential expression analysis
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading Model, PCA, and Data")
print("=" * 60)

model         = pickle.load(open("model/model.pkl",          "rb"))
pca           = pickle.load(open("model/pca.pkl",            "rb"))
encoder       = pickle.load(open("model/encoder.pkl",        "rb"))
scaler        = pickle.load(open("model/scaler.pkl",         "rb"))
feature_names = pickle.load(open("model/feature_names.pkl",  "rb"))

# Load full original dataset for differential expression
df_raw = pd.read_csv("data/Brain_GSE50161.csv")

# Gene expression matrix (only gene columns, no metadata)
# We keep the original unscaled values for interpretability
df_genes = df_raw.drop(columns=['samples', 'type'])

# Remove AFFX control probes (same as preprocessing)
df_genes = df_genes.loc[:, ~df_genes.columns.str.startswith("AFFX")]

class_names = list(encoder.classes_)
print(f"Gene features    : {len(feature_names)}")
print(f"Classes          : {class_names}")
print(f"Model trees      : {model.n_estimators}")


# ===============================================================
# METHOD 1: Random Forest Feature Importance via PCA Loadings
#
# The challenge: our model was trained on PCA components,
# not original genes. So model.feature_importances_ gives us
# importance of PC1, PC2, PC3... — not actual gene names.
#
# To get gene importances, we use this math:
#
#   PCA components are LINEAR COMBINATIONS of genes:
#   PC1 = (0.003 × Gene_A) + (0.001 × Gene_B) + (-0.002 × Gene_C) + ...
#
#   pca.components_ is a matrix of shape (50 components × 54613 genes)
#   Each row = one component, each value = how much that gene contributes
#
#   Gene importance = SUM over all components of:
#                     (component importance × |gene loading in that component|)
#
#   This "unrolls" PCA back to individual gene contributions
# ===============================================================
print("\n" + "=" * 60)
print("METHOD 1: Random Forest Importance via PCA Loadings")
print("=" * 60)

# model.feature_importances_ = array of length 50 (one per PCA component)
# Values sum to 1.0, higher = more important component
rf_importances = model.feature_importances_
print(f"RF importances shape: {rf_importances.shape} (one per PCA component)")

# pca.components_ = matrix of shape (50, 54613)
# pca.components_[i, j] = how much gene j contributes to component i
pca_loadings = np.abs(pca.components_)   # absolute values (direction doesn't matter)
print(f"PCA loadings shape : {pca_loadings.shape} (components × genes)")

# Matrix multiplication to get gene-level importance:
# rf_importances (50,) × pca_loadings (50, 54613) → gene_scores (54613,)
# For each gene: sum up (component_importance × gene_loading_in_component)
gene_scores = rf_importances @ pca_loadings
# '@' is matrix multiplication operator in Python/numpy

print(f"Gene scores shape  : {gene_scores.shape} (one score per gene)")

# Build a DataFrame with gene names and their importance scores
rf_gene_importance = pd.DataFrame({
    'gene'      : feature_names,
    'importance': gene_scores
})

# Sort by importance (highest first)
rf_gene_importance = rf_gene_importance.sort_values(
    'importance', ascending=False
).reset_index(drop=True)

print("\nTop 15 Biomarker Genes (Random Forest Method):")
print("-" * 45)
print(rf_gene_importance.head(15).to_string(index=False))

# Save top 100 to CSV
rf_gene_importance.head(100).to_csv(
    "outputs/biomarkers/top100_rf_importance.csv", index=False
)
print("\nSaved: outputs/biomarkers/top100_rf_importance.csv")

# Plot top 20 genes
plt.figure(figsize=(10, 6))
top20_rf = rf_gene_importance.head(20)
bars = plt.barh(
    top20_rf['gene'][::-1],          # reverse so highest is at top
    top20_rf['importance'][::-1],
    color='#7c3aed',
    edgecolor='white',
    linewidth=0.5
)
plt.xlabel("Importance Score", fontsize=12)
plt.title("Top 20 Biomarker Genes\n(Random Forest + PCA Loading Method)",
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/biomarkers/top20_rf_genes.png", dpi=150)
plt.show()
print("Chart saved: outputs/biomarkers/top20_rf_genes.png")


# ===============================================================
# METHOD 2: ANOVA F-test
#
# ANOVA = Analysis of Variance
# It tests: "Is this gene's expression significantly different
#            across the 5 tumor groups?"
#
# For each gene, we calculate an F-statistic and p-value:
#   High F + Low p-value → gene is a strong discriminator
#
# p-value < 0.05 = statistically significant (standard threshold)
# p-value < 0.001 = highly significant (research paper standard)
#
# This is purely statistical — independent of our ML model
# ===============================================================
print("\n" + "=" * 60)
print("METHOD 2: ANOVA F-Test (Statistical Biomarker Detection)")
print("=" * 60)
print("Calculating F-statistics for all genes... (may take 1-2 min)")

# For ANOVA we need gene expression grouped by tumor type
# We use the original (unscaled) values for interpretability

f_scores  = []
p_values  = []
gene_list = []

# Loop through each gene column
# For each gene, we collect expression values per class
# then run one-way ANOVA: f_oneway(class0_values, class1_values, ...)
for gene in df_genes.columns:
    # Group expression values by tumor type
    groups = [
        df_genes.loc[df_raw['type'] == cls, gene].values
        for cls in class_names
    ]
    # Run ANOVA
    # f_stat: how different are the group means relative to within-group variance?
    # p_val : probability that this difference happened by random chance
    f_stat, p_val = stats.f_oneway(*groups)  # *groups unpacks the list

    f_scores.append(f_stat)
    p_values.append(p_val)
    gene_list.append(gene)

# Build results DataFrame
anova_results = pd.DataFrame({
    'gene'   : gene_list,
    'f_score': f_scores,
    'p_value': p_values
})

# Sort by F-score (highest = most discriminative)
anova_results = anova_results.sort_values('f_score', ascending=False).reset_index(drop=True)

# Count significant genes
sig_001 = (anova_results['p_value'] < 0.001).sum()
sig_005 = (anova_results['p_value'] < 0.05).sum()
print(f"\nGenes significant at p < 0.001 : {sig_001}")
print(f"Genes significant at p < 0.05  : {sig_005}")
print(f"Total genes tested             : {len(anova_results)}")

print("\nTop 15 Biomarker Genes (ANOVA Method):")
print("-" * 55)
print(anova_results.head(15)[['gene','f_score','p_value']].to_string(index=False))

# Save full results
anova_results.to_csv("outputs/biomarkers/anova_all_genes.csv", index=False)
anova_results.head(100).to_csv("outputs/biomarkers/top100_anova.csv", index=False)
print("\nSaved: outputs/biomarkers/anova_all_genes.csv")
print("Saved: outputs/biomarkers/top100_anova.csv")

# Plot top 20 ANOVA genes
plt.figure(figsize=(10, 6))
top20_anova = anova_results.head(20)
plt.barh(
    top20_anova['gene'][::-1],
    top20_anova['f_score'][::-1],
    color='#2563eb',
    edgecolor='white',
    linewidth=0.5
)
plt.xlabel("ANOVA F-Score", fontsize=12)
plt.title("Top 20 Biomarker Genes\n(ANOVA F-Test Method)",
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/biomarkers/top20_anova_genes.png", dpi=150)
plt.show()
print("Chart saved: outputs/biomarkers/top20_anova_genes.png")


# ===============================================================
# METHOD 3: Differential Expression (Per Tumor Type vs Normal)
#
# For each tumor type, we compare its gene expression to normal tissue.
# This finds genes that are:
#   UP-regulated   : much more active in tumor than normal
#   DOWN-regulated : much less active in tumor than normal
#
# We use log2 Fold Change (log2FC):
#   log2FC = log2(tumor_mean / normal_mean)
#
#   log2FC > 1.0  → gene is 2x more expressed in tumor (up-regulated)
#   log2FC < -1.0 → gene is 2x less expressed in tumor (down-regulated)
#
# Combined with p-value, this is the standard method in
# published bioinformatics papers (volcano plot analysis)
# ===============================================================
print("\n" + "=" * 60)
print("METHOD 3: Differential Expression (Tumor vs Normal)")
print("=" * 60)

# Get normal tissue samples
normal_mask   = df_raw['type'] == 'normal'
normal_values = df_genes[normal_mask]
normal_mean   = normal_values.mean()  # mean expression of each gene in normal

print(f"Normal samples used as baseline: {normal_mask.sum()}")

all_degs = {}  # dictionary to store results per tumor type

for tumor_type in class_names:
    if tumor_type == 'normal':
        continue  # skip normal vs normal

    # Get tumor samples
    tumor_mask   = df_raw['type'] == tumor_type
    tumor_values = df_genes[tumor_mask]
    tumor_mean   = tumor_values.mean()

    # Log2 Fold Change
    # Add small value (1e-6) to avoid log(0) errors
    log2fc = np.log2(
        (tumor_mean + 1e-6) / (normal_mean + 1e-6)
    )

    # T-test for each gene (tumor group vs normal group)
    p_vals = []
    for gene in df_genes.columns:
        t_stat, p_val = stats.ttest_ind(
            tumor_values[gene].values,
            normal_values[gene].values
        )
        p_vals.append(p_val)

    p_vals = np.array(p_vals)

    # Build results
    deg_df = pd.DataFrame({
        'gene'   : df_genes.columns,
        'log2FC' : log2fc.values,
        'p_value': p_vals
    })

    # Classify each gene:
    #   Upregulated   : log2FC > 1 AND p < 0.05
    #   Downregulated : log2FC < -1 AND p < 0.05
    #   Not significant: everything else
    deg_df['regulation'] = 'not_significant'
    deg_df.loc[(deg_df['log2FC'] >  1.0) & (deg_df['p_value'] < 0.05), 'regulation'] = 'upregulated'
    deg_df.loc[(deg_df['log2FC'] < -1.0) & (deg_df['p_value'] < 0.05), 'regulation'] = 'downregulated'

    # Sort by absolute fold change (largest change first)
    deg_df['abs_log2FC'] = deg_df['log2FC'].abs()
    deg_df = deg_df.sort_values('abs_log2FC', ascending=False).reset_index(drop=True)

    up_count   = (deg_df['regulation'] == 'upregulated').sum()
    down_count = (deg_df['regulation'] == 'downregulated').sum()
    print(f"\n{tumor_type}:")
    print(f"  Up-regulated genes   : {up_count}")
    print(f"  Down-regulated genes : {down_count}")
    print(f"  Top 5 up-regulated   : {list(deg_df[deg_df['regulation']=='upregulated']['gene'].head(5))}")

    # Save per-tumor DEG results
    filename = f"outputs/biomarkers/DEG_{tumor_type}.csv"
    deg_df.to_csv(filename, index=False)

    all_degs[tumor_type] = deg_df


# ---------------------------------------------------------------
# STEP: Volcano Plots (one per tumor type)
#
# A volcano plot visualizes ALL genes at once:
#   X-axis = log2 Fold Change (how much expression changed)
#   Y-axis = -log10(p-value) (how statistically significant)
#
# Up-regulated genes (red)   → top-right corner
# Down-regulated genes (blue)→ top-left corner
# Not significant (grey)     → bottom/middle
#
# This is one of the most common plots in published papers
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Generating Volcano Plots")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()  # flatten 2x2 grid to list of 4

tumor_types_plot = [t for t in class_names if t != 'normal']

for idx, tumor_type in enumerate(tumor_types_plot):
    deg_df = all_degs[tumor_type]
    ax = axes[idx]

    # Separate into 3 groups for coloring
    not_sig = deg_df[deg_df['regulation'] == 'not_significant']
    up      = deg_df[deg_df['regulation'] == 'upregulated']
    down    = deg_df[deg_df['regulation'] == 'downregulated']

    # -log10(p) → larger value = more significant
    # We use clip to avoid infinite values from p=0
    def neg_log10(p):
        return -np.log10(np.clip(p, 1e-300, 1))

    # Plot each group with different color
    ax.scatter(not_sig['log2FC'], neg_log10(not_sig['p_value']),
               alpha=0.3, s=3, color='grey',   label='Not significant')
    ax.scatter(up['log2FC'],      neg_log10(up['p_value']),
               alpha=0.6, s=8, color='#dc2626', label=f'Up ({len(up)})')
    ax.scatter(down['log2FC'],    neg_log10(down['p_value']),
               alpha=0.6, s=8, color='#2563eb', label=f'Down ({len(down)})')

    # Reference lines
    ax.axvline(x= 1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=-1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=neg_log10(0.05), color='orange', linestyle='--', linewidth=0.8, alpha=0.5)

    # Label top 5 most significant up-regulated genes
    top5 = up.head(5)
    for _, row in top5.iterrows():
        ax.annotate(
            row['gene'],
            xy=(row['log2FC'], neg_log10(row['p_value'])),
            fontsize=6,
            color='#7f1d1d'
        )

    ax.set_title(f"{tumor_type} vs Normal", fontsize=11, fontweight='bold')
    ax.set_xlabel("log2 Fold Change", fontsize=9)
    ax.set_ylabel("-log10(p-value)", fontsize=9)
    ax.legend(fontsize=7, markerscale=2)

plt.suptitle("Volcano Plots: Differential Gene Expression\n(Tumor vs Normal Brain Tissue)",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("outputs/biomarkers/volcano_plots.png", dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved: outputs/biomarkers/volcano_plots.png")


# ---------------------------------------------------------------
# STEP: Find Consensus Biomarkers
#
# The most reliable biomarkers appear in BOTH:
#   - RF importance top genes (model-based)
#   - ANOVA top genes (statistics-based)
#
# Overlap between two independent methods = stronger evidence
# This is the kind of result you'd highlight in a paper
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("CONSENSUS BIOMARKERS (Overlap of RF + ANOVA Methods)")
print("=" * 60)

# Take top 200 genes from each method
top200_rf    = set(rf_gene_importance.head(200)['gene'])
top200_anova = set(anova_results.head(200)['gene'])

consensus = top200_rf & top200_anova   # intersection (genes in BOTH)

print(f"Top 200 RF genes     : {len(top200_rf)}")
print(f"Top 200 ANOVA genes  : {len(top200_anova)}")
print(f"Consensus (overlap)  : {len(consensus)} genes")
print("\nThese genes appear important in BOTH methods — strongest biomarker candidates:")

# Get their ANOVA scores for ranking
consensus_df = anova_results[anova_results['gene'].isin(consensus)].head(30)
print(consensus_df[['gene','f_score','p_value']].head(20).to_string(index=False))

consensus_df.to_csv("outputs/biomarkers/consensus_biomarkers.csv", index=False)
print("\nSaved: outputs/biomarkers/consensus_biomarkers.csv")


# ---------------------------------------------------------------
# STEP: Heatmap of Top Biomarkers
#
# A heatmap shows expression levels of the top genes
# across all samples, colored by value.
# Samples are grouped by tumor type.
# This visually confirms whether the genes truly discriminate.
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Generating Biomarker Heatmap")
print("=" * 60)

# Use top 25 ANOVA genes for the heatmap
top25_genes = list(anova_results.head(25)['gene'])

# Prepare data: select those genes + add type for grouping
heatmap_data = df_genes[top25_genes].copy()
heatmap_data['type'] = df_raw['type'].values

# Sort by tumor type so same classes are together
heatmap_data = heatmap_data.sort_values('type')
labels = heatmap_data['type'].values
heatmap_matrix = heatmap_data[top25_genes].values  # shape: (130, 25)

# Normalize each gene (column) for display
# So we can see relative patterns, not absolute values
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
heatmap_normalized = mm.fit_transform(heatmap_matrix)

plt.figure(figsize=(14, 8))
ax = sns.heatmap(
    heatmap_normalized.T,    # transpose: genes as rows, samples as columns
    cmap='RdBu_r',           # Red=high, Blue=low expression
    xticklabels=False,       # too many samples to label
    yticklabels=top25_genes,
    cbar_kws={'label': 'Normalized Expression'}
)

plt.title("Top 25 Biomarker Genes Expression Heatmap\n(Samples sorted by Tumor Type)",
          fontsize=13, fontweight='bold')
plt.xlabel("Samples (grouped by tumor type)", fontsize=11)
plt.ylabel("Biomarker Genes", fontsize=11)

# Add color bar at top showing which tumor type each sample is
# Create a color strip manually
type_colors = {
    'ependymoma':'#7c3aed', 'glioblastoma':'#2563eb',
    'medulloblastoma':'#059669', 'normal':'#d97706',
    'pilocytic_astrocytoma':'#dc2626'
}
color_strip = [type_colors[t] for t in labels]

# Add a row of colored rectangles at bottom as class indicator
for i, color in enumerate(color_strip):
    plt.gca().add_patch(
        plt.Rectangle((i, -1.5), 1, 1, color=color, transform=plt.gca().transData)
    )

plt.tight_layout()
plt.savefig("outputs/biomarkers/biomarker_heatmap.png", dpi=150)
plt.show()
print("Chart saved: outputs/biomarkers/biomarker_heatmap.png")


# Final summary
print("\n" + "=" * 60)
print("✅ BIOMARKER DISCOVERY COMPLETE — SUMMARY")
print("=" * 60)
print(f"Method 1 (RF)    : Top genes saved → top100_rf_importance.csv")
print(f"Method 2 (ANOVA) : {sig_001} highly significant genes (p<0.001)")
print(f"Method 3 (DEG)   : Volcano plots + per-class DEG CSVs saved")
print(f"Consensus        : {len(consensus)} strong biomarker candidates")
print("\nAll results in: outputs/biomarkers/")
print("\n👉 Next: Run app.py to build the Streamlit web app")