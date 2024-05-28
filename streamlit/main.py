import streamlit as st
import pandas as pd
import numpy as np

from customrec_engine import RecommendationAbstract
from customrec_engine import engines, PRODUCT_DATAS

# Load data
# products_filepath = "../data/products_books_v1.csv"
# transactions_filepath = "../data/transactions_books_v1.csv"

DATASET_AVAILABLE = {
    "games": {
        "product_data": PRODUCT_DATAS[0],
        "index": 0,
        "sets": {
            "Souls Player": [374320, 1245620, 335300],
            "RPG Player": [],
            "Strategy Player": []
        }
    },
    "books": {
        "product_data": PRODUCT_DATAS[1],
        "index": 1,
        "sets": {
            "Harry Potter Fan": [],
            "Sci Fi Fan": []
        }
    },
    "movies": {
        "product_data": PRODUCT_DATAS[2],
        "index": 2,
        "sets": {
            "Action Movie Fan": [],
            "Romantic Movie Fan": []
        }
    }
}

dataset_selected = 0
product_data = PRODUCT_DATAS[0]
products_filepath = product_data["product_filepath"]
transactions_filepath = product_data["transactions_filepath"]
# Select product type
products_type = "books"  # Support more types in the future


products_df = pd.read_csv(products_filepath)
training_transactions_df = pd.read_csv(transactions_filepath)

# Define the available product cases
product_cases = {
    "books": {
        "Harry Potter Fan": ["Book 1", "Book 2", "Book 3"],
        "Sci Fi Fan": ["Book 4", "Book 5", "Book 6"]
    }
}



# Select the dataset in product datas.
selected_product_dataset = st.selectbox("Select the dataset:", list(DATASET_AVAILABLE.keys()))
if selected_product_dataset:
    product_data = DATASET_AVAILABLE[selected_product_dataset]["product_data"]
    products_filepath = product_data["product_filepath"]
    transactions_filepath = product_data["transactions_filepath"]
    products_df = pd.read_csv(products_filepath)
    training_transactions_df = pd.read_csv(transactions_filepath)
    products_type = selected_product_dataset


# Define the recommendation engines
class CosineSimilarity:
    @staticmethod
    def recommend_from_past(transactions):
        # Dummy implementation, replace with actual recommendation logic
        return ["Product A", "Product B", "Product C"]

# Function to generate base cases based on product type
def base_cases(product_type):
    return ["empty_case"] + list(DATASET_AVAILABLE[product_type]['sets'].keys())

def get_product_id(product_name):
    return products_df.loc[products_df["product_title"] == product_name, "id"].iloc[0]

if "selected_product" not in st.session_state:
    st.session_state.selected_product = []
    
if "transactions_list" not in st.session_state:
    st.session_state.transactions_list = []


st.title("Product Recommendation Demo")

recommend_single_engine = st.selectbox("Select the recommendation engine for single product:", list(engines.keys()))
recommend_past_engine = st.selectbox("Select the recommendation engine for past transactions:", list(engines.keys()))


# Select base case
base_case = st.selectbox("Select base case:", base_cases(products_type))
if base_case != "empty_case":
    print('product_type', products_type)
    # reset the cart transactions using the new dataset
    transactions_list = []
    
    for product_id in DATASET_AVAILABLE[selected_product_dataset]["sets"][base_case]:
        # print("Product ID", product_id)
        try:
            product_title = products_df.loc[products_df["product_id"] == product_id, 'product_title']
            transactions_list.append({
                "id": product_id,
                "user_id": 1,
                "product_id": product_id,
                "product_title": product_title.iloc[0],
                "rate": 5                
            })
        except Exception as e:
            print("Base transactions || Error at", e)
    st.session_state.transactions_list = transactions_list
            

st.write("Selected products:", st.session_state.selected_product)


# Search and select products
selected_product = st.selectbox("Select products to add to cart:", products_df["product_title"])
st.session_state.selected_product = selected_product
print("Selected Product", st.session_state.selected_product)
# Buy button
if st.button("Buy"):
    transactions = st.session_state.transactions_list
    product = st.session_state.selected_product
    product_id = get_product_id(product)
    transaction = {
        "id": product_id,
        "user_id": 1,
        "product_id": product_id,
        'product_title': product, 
        "rate": 5
    }
    transactions.append(transaction)
    st.session_state.transactions_list = transactions
    
def to_arr_df(transactions, columns = ["product_id", "product_title", "product_price", "prediction"]):
    arr = []
    for dftrans, pred  in transactions:
        dicttrans = dftrans
        dicttrans["prediction"] = pred
        arr.append(dicttrans)
    try:
        return pd.DataFrame(arr).loc[:, columns]
    except Exception as e:
        print("Error at", e)
        return pd.DataFrame(arr)

# Get recommendations
if st.button("Get Recommendations"):
    if st.session_state.selected_product:
        
        selected_engine: RecommendationAbstract = engines[recommend_single_engine]["engine"](products=products_df, product_data=product_data, transactions=training_transactions_df)
        product = st.session_state.selected_product
        selected_engine.load()
        product_id = get_product_id(product)
        print("Product ID", product_id, selected_engine.strategy_name)
        recommendation = selected_engine.recommend_from_single(product_id=product_id)
        st.write("Recommendations based on selected products:")
        st.dataframe(to_arr_df(recommendation))
    else:
        st.warning("Please select some products first!")

    transactions = st.session_state.transactions_list
    # sort by product appearisons 
    if not len(transactions) == 0:
        # print("recommend_past_engine", recommend_past_engine)
        recommend_past_engine_obj: RecommendationAbstract = engines[recommend_past_engine]["engine"](products=products_df, product_data=product_data)
        recommend_past_engine_obj.load()
        
        transaction_books_ids = []
        for transaction in transactions:
            transaction_books_ids.append(transaction["product_id"])
        recommend_past =recommend_past_engine_obj.recommend_from_past(transaction_books_ids)
        st.write("Recommendations based on past transaction:")
        st.dataframe(to_arr_df(recommend_past))

# Show transactions
st.write("Transaction History:")
# Show st.session_state.transactions_list
transactions_df = pd.DataFrame(st.session_state.transactions_list)
st.dataframe(transactions_df)
    
