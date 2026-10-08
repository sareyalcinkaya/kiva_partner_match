import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# Load and join the four files
loans = pd.read_csv("data/kiva_loans.csv")
themes = pd.read_csv("data/loan_themes_by_region.csv")
theme_ids = pd.read_csv("data/loan_theme_ids.csv")
mpi = pd.read_csv("data/kiva_mpi_region_locations.csv")

# Clean MPI — drop placeholder geocodes
mpi = mpi[mpi["lat"] != 1000.0]

# Filter partners: keep partners with 100+ loans
partner_counts = loans.groupby("partner_id").size()
active_partners = partner_counts[partner_counts >= 100].index
themes = themes[themes["Partner ID"].isin(active_partners)]

# Build partner feature matrix from loan_themes_by_region
# Features: sector, country, Loan Theme Type
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
feature_cols = (themes[["sector", "country", "Loan Theme Type"]]
                .fillna("Unknown")
                .astype(str))
partner_features = encoder.fit_transform(feature_cols)

# Compute cosine similarity matrix (partners x partners)
sim_matrix = cosine_similarity(partner_features)

# Save
os.makedirs("model", exist_ok=True)
with open("model/similarity_matrix.pkl", "wb") as f:
    pickle.dump({
        "matrix": sim_matrix,
        "partners": themes.reset_index(drop=True),
        "encoder": encoder
    }, f)

print("Done — similarity matrix saved.")