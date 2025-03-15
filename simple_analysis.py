import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SimpleAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def top_branch_seller(self, head_num: int = 10):
        num_branch = len(self.df["section"])
        top_seller_branch = self.df.groupby(["department", "section"])["quantity"].sum().reset_index()

        top_branch_order = top_seller_branch.sort_values(by="quantity", ascending=False).iloc[0:num_branch]

        
        # Find the most sold items in this branch
        top_branch = top_seller_branch.sort_values(by="quantity", ascending=False).iloc[0]

        top_items = self.df[
            (self.df["department"] == top_branch["department"]) &
            (self.df["section"] == top_branch["section"])
        ].groupby("item_description")["quantity"].sum().reset_index().sort_values(by="quantity", ascending=False)
        
        return top_branch_order, top_items.head(head_num)
    
    def check_popularity(self, popular_item):
        # Why Some Items Are Unpopular/popular?
        popular_analysis = self.df[self.df["item_description"].isin(popular_item["item_description"])]
        price_analysis = popular_analysis.groupby("item_description")[["purshase_price", "selling_price"]].mean()
        price_analysis["profit_margin"] = price_analysis["selling_price"] - price_analysis["purshase_price"]

        return price_analysis

    def popular_and_unpopular(self, head_num: int = 10):
        item_sales = self.df.groupby("item_description")["quantity"].sum().reset_index()

        popular_item = item_sales.sort_values(by="quantity", ascending=False).head(head_num)

        unpopular_item = item_sales.sort_values(by="quantity", ascending=True).head(head_num)

        return popular_item, unpopular_item
    
    @staticmethod
    def predicting_top_seller(df: pd.DataFrame, classifier_model):
        # Create target variable (1 = top seller, 0 = unpopular)
        df['top_seller'] = df['quantity'].apply(lambda x: 1 if x > df['quantity'].median() else 0)

        # Select features
        features = ['purshase_price', 'selling_price', 'quantity']
        X = df[features]
        y = df['top_seller']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = classifier_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        return model
    
    def top_season_seller(self):
        pass
    
    def most_overdated(self):
        pass
    
    def forecasting_selling(self):
        pass
    
    def run(self):
        print("Running analysis...")
        # Top branch seller
        
        branch_order, top_item_top_branch = self.top_branch_seller()
        
        print("The Order Of Top Branches:", branch_order)
        print("The Top Items In the Top Branch: ", top_item_top_branch)
        
        # Identify popular and unpopular items
        popular_items, unpopular_items = self.popular_and_unpopular()
        print("Popular Items:")
        print(popular_items)
        print("Unpopular Items:")
        print(unpopular_items)
        
        # Check popularity analysis
        popularity_analysis = self.check_popularity(popular_items)
        print("Popularity Analysis:")
        print(popularity_analysis)
        
        # Predict top sellers using Logistic Regression
        print("Training top seller predictor...")
        model = self.predicting_top_seller(self.df, LogisticRegression)
        
        print("Analysis complete.")
