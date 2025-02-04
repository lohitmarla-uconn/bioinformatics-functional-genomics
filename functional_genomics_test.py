import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# 1️⃣ Load and Simulate Gene Expression Data
np.random.seed(42)
genes = [f"Gene_{i}" for i in range(1, 101)]
conditions = ["Control_1", "Control_2", "Treated_1", "Treated_2"]

# Generate random expression values for each gene in each condition
expression_data = np.random.normal(loc=10, scale=2, size=(100, 4))
df = pd.DataFrame(expression_data, index=genes, columns=conditions)

# Introduce differential expression in some genes
df.loc["Gene_5":"Gene_15", ["Treated_1", "Treated_2"]] += 5  # Upregulated
df.loc["Gene_20":"Gene_30", ["Treated_1", "Treated_2"]] -= 5  # Downregulated

# 2️⃣ Compute Log2 Fold Change (log2FC) and P-values
df["log2FC"] = np.log2(df[["Treated_1", "Treated_2"]].mean(axis=1) / df[["Control_1", "Control_2"]].mean(axis=1))

# Perform t-tests to compute p-values
p_values = []
for gene in df.index:
    t_stat, p_val = ttest_ind(df.loc[gene, ["Treated_1", "Treated_2"]],
                              df.loc[gene, ["Control_1", "Control_2"]], equal_var=False)
    p_values.append(p_val)

df["p_value"] = p_values

# Apply False Discovery Rate (FDR) correction instead of Bonferroni
df["adjusted_p"] = multipletests(df["p_value"], method="fdr_bh")[1]

# 3️⃣ Identify Differentially Expressed Genes (DEGs)
deg_df = df[(df["log2FC"].abs() > 0.8) & (df["adjusted_p"] < 0.05)]
print(f"Number of differentially expressed genes: {len(deg_df)}")

# Debugging: Print significant genes count at different thresholds
print(f"Genes with adjusted p < 0.05: {sum(df['adjusted_p'] < 0.05)}")
print(f"Genes with adjusted p < 0.10: {sum(df['adjusted_p'] < 0.10)}")

if len(deg_df) > 0:
    # 4️⃣ Visualization - Heatmap of DEGs
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.loc[deg_df.index, conditions], cmap="coolwarm", annot=True)
    plt.title("Differentially Expressed Genes Heatmap")
    plt.show()
else:
    print("No DEGs found. Heatmap will not be plotted.")

# 5️⃣ Visualization - Volcano Plot
plt.figure(figsize=(8, 6))
plt.scatter(df["log2FC"], -np.log10(df["p_value"]), color='gray', alpha=0.7)

# Highlight DEGs
if len(deg_df) > 0:
    plt.scatter(deg_df["log2FC"], -np.log10(deg_df["p_value"]), color='red', label="DEGs")

plt.axhline(-np.log10(0.05), color='blue', linestyle="dashed", label="p=0.05 cutoff")
plt.axvline(1, color='black', linestyle="dashed", label="log2FC = ±1")
plt.axvline(-1, color='black', linestyle="dashed")

plt.xlabel("log2 Fold Change")
plt.ylabel("-log10(p-value)")
plt.title("Volcano Plot of Gene Expression")
plt.legend()
plt.show()

# 6️⃣ Save the Results
deg_df.to_csv("differentially_expressed_genes.csv")
print("DEG results saved successfully!")
