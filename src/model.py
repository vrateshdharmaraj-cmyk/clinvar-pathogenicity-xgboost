
# 1. IMPORTS

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score



# 2. LOAD DATA
print("Loading dataset...")

df = pd.read_csv("../data/variant_summary.txt", sep="\t", nrows=50000)

print("Data loaded:", df.shape)



# 3. CLEAN DATA

df = df[[
    "ClinicalSignificance",
    "Type",
    "Name",
    "GeneSymbol"
]]

# only clean labels
df = df[df["ClinicalSignificance"].isin(["Pathogenic", "Benign"])]

print(df["ClinicalSignificance"].value_counts())



# 4. LABEL ENCODING

df["label"] = df["ClinicalSignificance"].map({
    "Pathogenic": 1,
    "Benign": 0
})



# 5. FEATURE ENGINEERING


# Variant type
df["Type_encoded"] = df["Type"].astype("category").cat.codes

# Gene frequency
gene_counts = df["GeneSymbol"].value_counts()
df["Gene_freq"] = df["GeneSymbol"].map(gene_counts)

# Mutation patterns
df["is_substitution"] = df["Name"].str.contains(">", na=False).astype(int)
df["is_deletion"] = df["Name"].str.contains("del", na=False).astype(int)
df["is_insertion"] = df["Name"].str.contains("ins", na=False).astype(int)



# 6. PREPARE FEATURES

features = [
    "Type_encoded",
    "Gene_freq",
    "is_substitution",
    "is_deletion",
    "is_insertion"
]

X = df[features]
y = df["label"]

X = df[features + ["GeneSymbol"]]
y = df["label"]



# 7. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#  gene pathogenic score ONLY from training data
temp = X_train.copy()
temp["label"] = y_train

gene_patho_ratio = temp.groupby("GeneSymbol")["label"].mean()

# Map to train and test
X_train["Gene_patho_score"] = X_train["GeneSymbol"].map(gene_patho_ratio)
X_test["Gene_patho_score"] = X_test["GeneSymbol"].map(gene_patho_ratio)

# Fill missing
X_train["Gene_patho_score"] = X_train["Gene_patho_score"].fillna(0.5)
X_test["Gene_patho_score"] = X_test["Gene_patho_score"].fillna(0.5)

# Drop GeneSymbol
X_train = X_train.drop(columns=["GeneSymbol"])
X_test = X_test.drop(columns=["GeneSymbol"])





# 8. TRAIN MODEL

scale = (len(y_train) - sum(y_train)) / sum(y_train)

model = XGBClassifier(scale_pos_weight=scale)
model.fit(X_train, y_train)



# 9. EVALUATE MODEL

pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("\n=== RESULTS ===")
print("Accuracy:", accuracy_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, probs))





# 10. SHAP EXPLAINABILITY
import shap
import matplotlib.pyplot as plt

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Create plot WITHOUT showing
shap.summary_plot(shap_values, X_test, show=False)

# Save correctly
plt.savefig("../results/shap_plot.png", bbox_inches='tight')
plt.show()





