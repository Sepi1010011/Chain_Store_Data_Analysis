import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from typing import Tuple


class CustomerSegmentation:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.rfm_scaled, self.rfm = self.rfm_calculation()
        
    def rfm_calculation(self):
        try:
            self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"], errors='coerce')
            self.df['Amount'] = self.df['Quantity'] * self.df['UnitPrice']

            reference_date = self.df['InvoiceDate'].max() + datetime.timedelta(days=1)

            rfm = self.df.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (reference_date - x.max()).days,
                'InvoiceNo': 'nunique',
                'Amount': 'sum'
            }).reset_index()

            rfm.rename(columns={
                'InvoiceDate': 'Recency',
                'InvoiceNo': 'Frequency',
                'Amount': 'Monetary'
            }, inplace=True)
            rfm["CustomerID"] = rfm["CustomerID"].astype(str)

            rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].copy()

            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_features)

            return rfm_scaled, rfm
        
        except Exception as e:
            print(f"Error in RFM calculation: {e}")
            return None, None
        
    def find_best_k(self) -> int:
        try:
            silhouette_scores = {}
            for k in range(2, 11):  # Avoid k=1 as silhouette score is undefined
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(self.rfm_scaled)
                score = silhouette_score(self.rfm_scaled, labels)
                silhouette_scores[k] = score

            best_k = max(silhouette_scores, key=silhouette_scores.get)
            return best_k
        except Exception as e:
            print(f"Error finding best k: {e}")
            return 3
        
    def clustering(self, best_k: int) -> pd.DataFrame:
        try:
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            self.rfm['Cluster'] = kmeans.fit_predict(self.rfm_scaled)
            return self.rfm
        except Exception as e:
            print(f"Error in clustering: {e}")
            
    def show_plot(self) -> plt.Figure:
        try:
            pca = PCA(n_components=2)
            rfm_pca = pca.fit_transform(self.rfm_scaled)

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=self.rfm['Cluster'], cmap='viridis', alpha=0.7)
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_title("Customer Segmentation using KMeans (PCA Projection)")
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster')

            return fig  

        except Exception as e:
            print(f"Error in plot visualization: {e}")
            return None
    
    def sample(self, cluster):
        pass

    def run(self) -> pd.DataFrame:
        if self.rfm_scaled is None or self.rfm is None:
            print("Error: Invalid RFM data. Check dataset.")
            return
        
        best_k = self.find_best_k()
        
        return self.clustering(best_k).sample(20), self.show_plot(), best_k


