import streamlit as st
import pandas as pd

from customrec_engine import CosineSimilarityRecommender, WordVecBodyRecommender, TitleWordVecTitleyRecommender, RecommendationAbstract, TitleWordVecTitleyRecommenderV2


# Load data
# products_filepath = "../data/products_books_v1.csv"
# transactions_filepath = "../data/transactions_books_v1.csv"

product_data = {
    "data_context": "books",
    "product_filepath": "data/products_books_v1_10_10.csv",
    "transactions_filepath": "data/transactions_books_v1_10_10.csv",
    "features": ["id", "product_title", "product_image", "product_soup", "product_images"],
    "version": "1.0",
    "unique_name": "_books_v1_10_10",
}
products_filepath = product_data["product_filepath"]
transactions_filepath = product_data["transactions_filepath"]

products_df = pd.read_csv(products_filepath)
training_transactions_df = pd.read_csv(transactions_filepath)

cosineSimilarRecommender = CosineSimilarityRecommender(products=products_df, product_data=product_data)
# cosineSimilarRecommender.load()

wordVecBodyRecommender = WordVecBodyRecommender(products=products_df, product_data=product_data)
# wordVecBodyRecommender.load()

titleWordVecTitleRecommender = TitleWordVecTitleyRecommender(products=products_df, product_data=product_data)
# titleWordVecTitleRecommender.load()

titleWordVectRecommender2 = TitleWordVecTitleyRecommenderV2(products=products_df, product_data=product_data)


engines = {
    "cosine_similarity": {
        "title": "Cosine Similarity",
        "engine": cosineSimilarRecommender
    },
    "wordvec_body": {
        "title": "WordVec Body",
        "engine": wordVecBodyRecommender
    },
    "wordvec_title": {
        "title": "WordVec Title",
        "engine": titleWordVecTitleRecommender
    },
    "word_vec_title_v2": {
        "title": "WordVec Title V2",
        "engine": titleWordVectRecommender2
    
    },
    "collaborative_filtering": {
        "title": "Collaborative Filtering",
        "engine": cosineSimilarRecommender
    },
    "KNN Basic": {
        "title": "KNN Basic",
        "engine": cosineSimilarRecommender
    },
    
    "matrix_factorization SVD": {
        "title": "Base Algorithm",
        "engine": cosineSimilarRecommender
    },
    "matrix_factorization SVD++": {
        "title": "Base Algorithm",
        "engine": cosineSimilarRecommender
    },
    "matrix_factorization KNNWithMeans": {
        "title": "Base KNNWithMeans",
        "engine": cosineSimilarRecommender
    },
    
    "matrix_factorization KNNWithZscore": {
        "title": "Base KNNWithMeans",
        "engine": cosineSimilarRecommender
    },
    
}

# Define the available product cases
PRODUCT_CASES = {
    "books": {
        "Harry Potter Fan": ["Book 1", "Book 2", "Book 3"],
        "Sci Fi Fan": ["Book 4", "Book 5", "Book 6"]
    }
}
# Define the recommendation engines
class CosineSimilarity:
    @staticmethod
    def recommend_from_past(transactions):
        # Dummy implementation, replace with actual recommendation logic
        return ["Product A", "Product B", "Product C"]

# Function to generate base cases based on product type
def base_cases(product_type):
    return ["empty_case"] + list(PRODUCT_CASES[product_type].keys())

def get_product_id(product_name):
    return products_df.loc[products_df["product_title"] == product_name, "id"].iloc[0]




if "selected_product" not in st.session_state:
    st.session_state.selected_product = []
    
if "transactions_list" not in st.session_state:
    st.session_state.transactions_list = []


st.title("Product Recommendation Demo")

recommend_single_engine = st.selectbox("Select the recommendation engine for single product:", list(engines.keys()))
recommend_past_engine = st.selectbox("Select the recommendation engine for past transactions:", list(engines.keys()))

# Select product type
products_type = "books"  # Support more types in the future

# Select base case
base_case = st.selectbox("Select base case:", base_cases(products_type))
if base_case != "empty_case":
    st.session_state.selected_product = PRODUCT_CASES[products_type][base_case]

st.write("Selected products:", st.session_state.selected_product)

# Display products
# st.write("Product Information:")
# st.write(products_df)

# Search and select products
selected_product = st.selectbox("Select products to add to cart:", products_df["product_title"])
st.session_state.selected_product = selected_product
print(st.session_state.selected_product)
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
    

# Get recommendations
if st.button("Get Recommendations"):
    if st.session_state.selected_product:
        
        selected_engine: RecommendationAbstract = engines[recommend_single_engine]["engine"]
        product = st.session_state.selected_product
        selected_engine.load()
        product_id = get_product_id(product)
        # recommend_single = engines[recommend_single_engine].recommend_from_past(st.session_state.selected_product)
        # selected_engine = 
        recommendation = selected_engine.recommend_from_single(product_id=product_id)
        st.write("Recommendations based on selected products:", recommendation)
    else:
        st.warning("Please select some products first!")

    if not st.session_state.transaction_list.empty:
        # last_transaction = transactions_df.iloc[-1]
        # recommend_past = engines[recommend_past_engine].recommend_from_past([last_transaction])
        # st.write("Recommendations based on past transaction:", recommend_past)
        recommend_past = engines[recommend_past_engine]["engine"].recommend_from_past(st.session_state.transactions_list)
        st.write("Recommendations based on past transaction:", recommend_past)

# Show transactions
st.write("Transaction History:")
# Show st.session_state.transactions_list
transactions_df = pd.DataFrame(st.session_state.transactions_list)
st.write(transactions_df)
    
