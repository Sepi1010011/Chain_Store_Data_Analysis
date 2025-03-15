import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from simple_analysis import SimpleAnalysis
from customer_segmentation import CustomerSegmentation
from mba import MarketBasketAnalysis
from file_handling import DatasetLoader


# Streamlit App Title
st.title("🛒 Chain Store Analysis")
st.header("A system to analyze your data and find the pattern to gain your profit")
st.divider()

def append_to_csv(report_data, file_name="D:\\Projects\\Food Barcode Object Detection\\Deployments\\flagged\\reports.csv"):
    try:
        new_row = pd.DataFrame([report_data])
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
        else:
            df = pd.DataFrame(columns=report_data.keys())
            
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(file_name, index=False)
        
    except Exception as e:
        st.error(f"⚠️ Error storing the report: {e}")
        

def toggle_report_form():
    st.session_state.show_report_form = not st.session_state.show_report_form

def reset_report_form():
    st.session_state.report_data = {
        "name": None,
        "email": None,
        "feedback": None,
        "report_choice": None,
        "report_text": None,
        "date_of_report": datetime.datetime.now()
    }
    

if 'show_report_form' not in st.session_state:
    st.session_state.show_report_form = False
    if "report_data" not in st.session_state:
        st.session_state.report_data = {
        "name": None,
        "email": None,
        "feedback": None,
        "report_choice": None,
        "report_text": None,
        "date_of_report": datetime.datetime.now()
        }
        
    reset_report_form()

def upload_dataset_report():
    
    st.caption("A quick report form:")
    if st.session_state.show_report_form:
        
        with st.form(key="quick_form"):
            st.subheader("Your Name:")
            st.session_state.report_data["name"] = st.text_input("Enter your full name", value=st.session_state.report_data["name"])
            st.session_state.report_data["feedback"] = st.text_area("Provide your feedback", value=st.session_state.report_data["feedback"])
            
            st.subheader("write your email:")
            st.session_state.report_data["email"] = st.text_input("Enter your email:", value=st.session_state.report_data["email"])
            
            st.session_state.report_data["date_of_report"] = datetime.datetime.now()
            st.subheader("Select Your Problem:")
            st.session_state.report_data["report_choice"] = st.radio(
                "Choose an option",
                ["I upload the dataset but it will give me nothing",
                "Your system is too slow",
                "I can't upload an file",
                "It can't analyze very well",
                "my columns are different"], index=None, key="report_choice_radio")
            
            
            st.session_state.report_data["report_choice"] = "Was Not A Option"
                
            st.caption("If you can't see your problem in above list write it here:")
            st.session_state.report_data["report_text"] = st.text_input("Write your report!", value=st.session_state.report_data["report_text"])
            
            submit_button = st.form_submit_button(label="Submit")
            if submit_button:
                if not all (st.session_state.report_data.values()):
                    st.warning("Please fill in all of the fields!!")
                
                else:   
                    append_to_csv(st.session_state.report_data)
                    st.balloons()
                    st.success("✅ Your report has been submitted. Thank you for your feedback!")
                    
                    st.session_state.show_report_form = False
                    reset_report_form()
                    
        col1, col2 = st.columns([0.2, 0.2])
        with col1:
            st.button("🔄 New Form", on_click=reset_report_form)
            
        with col2:
            st.button("❌ Cancel Form", on_click=toggle_report_form)
            
            
def dataset_upload(key):
    uploaded_file = st.file_uploader("Upload your dataset in formats:", type=["csv", "xlsx", "xls", "json", "parquet"], key=key)

    if uploaded_file:
        dataset_loader = DatasetLoader(uploaded_file)
        df = dataset_loader.get_dataframe()

        if df.empty:
            st.error("Failed to load dataset. Please check the file format.")
        
        else:
            data_tab = st.tabs(["Dataset Info", "Dataset Description"])
            with data_tab[0]:
                st.header("Dataset Preview:")    
                dataset_info = dataset_loader.dataset_info()
                # Dataset Information
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.write("Shape:", dataset_info["Shape"])
                    st.write("Columns:", dataset_info["Columns"])
    
                with col2:
                    st.write("Data Types:")
                    st.write(dataset_info["Data Types"])
    
                with col3:
                    st.write("Missing Values:")
                    st.write(dataset_info["Missing Values"])
    
        
            with data_tab[1]:
                st.header("Dataset Description:")    
                dataset_info = dataset_loader.dataset_adv_info()
                # Dataset Information
                
                st.write("Head:", dataset_info["Head"])
    
                st.write("Description:")
                st.write(dataset_info["Description"])
                
                
            if 'from_date' in df.columns:
                df['from_date'] = pd.to_datetime(df['from_date'], errors='coerce')
    
            return df

# Home page
home_tabs = st.tabs(["Simple Analysis", "Customer Segmentation", "Market Basket Analysis"])

with home_tabs[0]:
    st.write("Upload your dataset for simple analysis")
    
    sa_df = dataset_upload("SA_Dataset_Uploading")
    
    col1, col2 = st.columns([0.2, 1.4])
    with col1:
        sa_button = st.button("Analyze", key="SA_Button")
        
    with col2:
        report_button = st.button("Report", key="SA_Button_Report", on_click=toggle_report_form)
        
        if st.session_state.show_report_form:
            upload_dataset_report()
        
    st.divider()
    
    if sa_button:
        # Create tabs instead of checkboxes
        tabs = st.tabs([
            "Top Sellers & Items", "Popular vs Unpopular Items", "Top Seller Classification",
            "Seasonal Trends & Overdated Items"
        ])
        
        # Top Sellers & Items Analysis
        with tabs[0]:
            st.subheader("Top Selling Branches")
            simple_analysis = SimpleAnalysis(sa_df)
            top_branches, top_items = simple_analysis.top_branch_seller()
            
            top_branches = top_branches.set_index("section")
            
            st.bar_chart(top_branches["quantity"], x_label="Section", y_label="Quantity")
            
            top_items = top_items.set_index("item_description")
            
            st.bar_chart(top_items["quantity"], x_label="Items", y_label="Quantity", horizontal=True)
            
        # Popular & Unpopular Items
        with tabs[1]:
            st.subheader("Popular vs Unpopular Items")
            popular_unpopular = SimpleAnalysis(sa_df)
            popular_item, unpopular_item = popular_unpopular.popular_and_unpopular()
            
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("Top 10 Popular Items:")
                st.write(popular_item)
            with col2:
                st.write("Bottom 10 Unpopular Items:")
                st.write(unpopular_item)
            
            popular_analysis = popular_unpopular.check_popularity(popular_item)
            unpopular_analysis = popular_unpopular.check_popularity(unpopular_item)
            
            
            st.write("Popular Analysis:")
            st.write(popular_analysis)
        
            st.write("Unpopular Analysis:")
            st.write(unpopular_analysis)
        
        # Classification of Top Seller Items
        with tabs[2]:
            st.subheader("Classify Top Seller Items")
            # top_classify = SimpleAnalysis(sa_df)
            # model = top_classify()

        # Seasonal Trends & Overdated Items
        with tabs[3]:
            st.subheader("Seasonal Trends & Old Stock Items")
            sa_df['month'] = sa_df['from_date'].dt.month
            monthly_sales = sa_df.groupby('month')['quantity'].sum()
            st.write("Sales by Month")
            st.bar_chart(monthly_sales)
            overdated_items = sa_df.groupby('item_description')['quantity'].sum().sort_values()
            st.write("Most Overdated Items:")
            st.write(overdated_items.head(10))


with home_tabs[1]:
    st.write("Upload your dataset for Customer Segmentation")
    cs_df = dataset_upload("CS_Dataset_Uploading")
    col1, col2 = st.columns([0.2, 1.4])
    with col1:
        cs_button = st.button("Analyze", key="CS_Button")
    
    with col2:
        report_button = st.button("Report", key="CS_Button_Report", on_click=toggle_report_form)
        if st.session_state.show_report_form:
            upload_dataset_report()
            
    st.divider()
    
    if cs_button:
        # Customer Segmentation
        st.subheader("Customer Segmentation")
        cs = CustomerSegmentation(cs_df)
        result, fig, best_k = cs.run()
        st.write("The number of best k clustering is: {best_k}")
        st.write(f"By Considering the {best_k} cluster")
        st.divider()
        st.write(result)
        st.pyplot(fig)
        
        st.write("""
                Cluster 1: High-value customers who purchase frequently, spend a lot, and have made recent purchases.\n
                Cluster 2: Medium-value or loyal customers with moderate purchase frequency and spend.\n
                Cluster 3: Low-value customers with infrequent and low spending, potentially at risk of churn.\n
                Cluster 4: New or occasional customers who haven't made many purchases yet or who purchased long ago
                 """)
        
     
     
with home_tabs[2]:
    st.write("Upload your dataset for Market Basket Analysis")
    mba_df = dataset_upload("MBA_Dataset_Uploading")
    
    col1, col2 = st.columns([0.2, 1.4])
    with col1:
        mba_button = st.button("Analyze", key="MBA_Button")
    with col2:
        report_button = st.button("Report", key="MBA_Button_Report", on_click=toggle_report_form)
        if st.session_state.show_report_form:
            upload_dataset_report()
            
    st.divider()

    if mba_button:
        st.subheader("Market Basket Analysis")
        mba = MarketBasketAnalysis(mba_df)

        mba_analysis = mba.run()

        srl = mba_analysis["strongest_relation_lift"]  
        npp = mba_analysis["next_purchase_prediction"]
        mfo = mba_analysis["most_freq_occurring"]
        mir = mba_analysis["most_important_rules"]
        exr = mba_analysis["extracting_order_rules"]

        st.write("Finding Strongest Relation Lift")
        st.write(srl)

        st.write("Finding Next Purchase Prediction")
        st.write(npp)

        st.write("Finding Most Frequent Occurring")
        st.write(mfo)

        st.write("Finding Most Important Rules")
        st.write(mir)

        st.write("Finding Extracting Order Rules")
        st.write(exr)