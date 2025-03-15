import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules


class MarketBasketAnalysis:
    """A class for Market Basket Analysis using Apriori Algorithm."""

    def __init__(self, df: pd.DataFrame):
        self.myretaildata = df
        self.basket = self.basketdata()
        self.association_rule = pd.DataFrame()

    def basketdata(self, country: str = "Germany") -> pd.DataFrame:
        """Creates a basket format dataset for association rule mining."""
        try:
        
            if self.myretaildata.empty:
                raise ValueError("Dataset is empty. Cannot proceed with basket data creation.")

            required_columns = {"Invoice", "Description", "Country", "Quantity"}
        
            if not required_columns.issubset(self.myretaildata.columns):
                raise KeyError(f"Missing required columns: {required_columns - set(self.myretaildata.columns)}")

            # Data Cleaning
            self.myretaildata['Description'] = self.myretaildata['Description'].str.strip()
            self.myretaildata.dropna(subset=['Invoice'], inplace=True)
            self.myretaildata['Invoice'] = self.myretaildata['Invoice'].astype(str)
            self.myretaildata = self.myretaildata[~self.myretaildata['Invoice'].str.contains('C')]
            self.myretaildata.drop_duplicates(inplace=True)

            return (self.myretaildata[self.myretaildata['Country'] == country]
                    .groupby(['Invoice', 'Description'])['Quantity']
                    .sum().unstack().fillna(0)
                    .reset_index().set_index('Invoice'))

        except Exception as e:
            print(f"Error in basket data frame: {e}")
            return pd.DataFrame()

    @staticmethod
    def my_encode_units(x: int) -> int:
        """Encodes product quantities into binary values."""
        return 1 if x >= 1 else 0

    def pattern_recognition(self, my_basket_sets, min_support=0.07, metric="lift", min_threshold=1, lift_threshold=3, confidence_threshold=0.3) -> pd.DataFrame:
        """Performs frequent pattern recognition using Apriori Algorithm."""
        try:
            frequent_items = apriori(my_basket_sets, min_support=min_support, use_colnames=True)
            self.association_rule = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)
            
            # Filter based on lift and confidence
            filtered_rules = self.association_rule[
                (self.association_rule['lift'] >= lift_threshold) &
                (self.association_rule['confidence'] >= confidence_threshold)
            ]
            return filtered_rules

        except Exception as e:
            print(f"Error in pattern recognition: {e}")
            return pd.DataFrame()

    def strongest_relation_lift(self, head_num=6) -> pd.DataFrame:
        """Finds the strongest relationships based on lift."""
        return self.association_rule.nlargest(head_num, 'lift')

    def next_purchase_prediction(self, head_num=6) -> pd.DataFrame:
        """Finds the most reliable purchase predictions based on confidence."""
        return self.association_rule.nlargest(head_num, 'confidence')

    def most_freq_occurring(self, head_num=6) -> pd.DataFrame:
        """Finds the most frequently occurring association rules based on support."""
        return self.association_rule.nlargest(head_num, 'support')

    def most_important_rules(self, head_num=6) -> pd.DataFrame:
        """Finds the most important association rules based on lift and confidence."""
        return self.association_rule.nlargest(head_num, ['lift', 'confidence'])

    def extracting_order_rules(self, head_num=5):
        """Extracts sequential purchase order rules."""
        try:
            if self.association_rule.empty:
                raise ValueError("No association rules available. Run pattern recognition first.")

            self.association_rule['antecedents'] = self.association_rule['antecedents'].apply(list)
            self.association_rule['consequents'] = self.association_rule['consequents'].apply(list)
            self.association_rule['full_rule'] = self.association_rule['antecedents'] + self.association_rule['consequents']

            # Most likely purchase order
            top_rule_sequence = self.association_rule.nlargest(1, ['confidence', 'lift'])['full_rule'].values[0]

            # Top 5 most common sequences
            top_sequences = self.association_rule.nlargest(head_num, ['confidence', 'lift'])
            ordered_rules = [
                {"Rule": row['full_rule'], "Confidence": row['confidence'], "Lift": row['lift']}
                for _, row in top_sequences.iterrows()
            ]

            # Longest order sequence
            self.association_rule['rule_length'] = self.association_rule['full_rule'].apply(len)
            longest_rule = self.association_rule.nlargest(1, ['rule_length', 'confidence'])['full_rule'].values[0]

            return {
                "most_likely_purchase_order": top_rule_sequence,
                "top_sequences": ordered_rules,
                "longest_sequence": longest_rule
            }

        except Exception as e:
            print(f"Error extracting ordered rules: {e}")
            return {}

    def assoc_rule_analysis(self, srl=False, npp=False, mfo=False, mir=False, exr=False):
        """Performs various association rule analyses based on selected parameters."""
        results = {}

        if srl:
            results["strongest_relation_lift"] = self.strongest_relation_lift()
        if npp:
            results["next_purchase_prediction"] = self.next_purchase_prediction()
        if mfo:
            results["most_freq_occurring"] = self.most_freq_occurring()
        if mir:
            results["most_important_rules"] = self.most_important_rules()
        if exr:
            results["extracting_order_rules"] = self.extracting_order_rules()

        return results if results else "No analysis selected."

    def run(self, srl=True, npp=True, mfo=True, mir=True, exr=True):
        """Runs the entire market basket analysis pipeline."""
        try:
            if self.basket.empty:
                raise ValueError("Basket data is empty. Check dataset or preprocessing steps.")

            my_basket_sets = self.basket.applymap(self.my_encode_units)
            if "POSTAGE" in my_basket_sets.columns:
                my_basket_sets.drop('POSTAGE', axis=1, inplace=True)

            self.association_rule = self.pattern_recognition(my_basket_sets)

            return self.assoc_rule_analysis(srl, npp, mfo, mir, exr)

        except Exception as e:
            print(f"Error in execution: {e}")
            return {}
