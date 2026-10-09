
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

os.makedirs("outputs", exist_ok=True)

# 1. Explore dataset
df = pd.read_csv("Mall_Customers.csv")
print(df.head(), "\n", df.describe(), "\n", df.isnull().sum())

features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = StandardScaler().fit_transform(df[features])

# 2. Try different K values
scores = {}
for k in range(2, 11):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    scores[k] = silhouette_score(X, labels)

print("\nSilhouette Scores:", scores)
plt.plot(list(scores.keys()), list(scores.values()), marker="o")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.savefig("outputs/k_selection.png")
plt.close()

# 3. Select best K and cluster customers
best_k = max(scores, key=scores.get)
df["Cluster"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)
print("\nSelected K:", best_k)

# 4. Describe customer groups
profile = df.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Age=("Age", "mean"),
    Income=("Annual Income (k$)", "mean"),
    Spending=("Spending Score (1-100)", "mean")
).round(2)
print("\nCluster Profiles:\n", profile)

sns.scatterplot(data=df, x=features[1], y=features[2], hue="Cluster")
plt.savefig("outputs/kmeans_clusters.png")
plt.close()

# 5. t-SNE visualization
Z = TSNE(n_components=2, random_state=42, init="pca",
         learning_rate="auto").fit_transform(X)
sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=df["Cluster"].astype(str))
plt.savefig("outputs/tsne_clusters.png")
plt.close()

# 6. Recommend campaigns
income_mid = df[features[1]].median()
spending_mid = df[features[2]].median()

def campaign(r):
    if r.Income >= income_mid and r.Spending >= spending_mid:
        return "A - Premium Loyalty"
    if r.Spending < spending_mid:
        return "B - Discounts and Coupons"
    return "C - Customer Engagement"

profile["Campaign"] = profile.apply(campaign, axis=1)
print("\nCampaign Recommendations:\n", profile)

df.to_csv("outputs/customer_clusters.csv", index=False)
profile.to_csv("outputs/campaign_recommendations.csv")
print("\nProject completed! Check the outputs folder.")
