import streamlit as st
import pandas as pd

from customrec_engine import CosineSimilarityRecommender, WordVecBodyRecommender, TitleWordVecTitleyRecommender


# Load data
products_filepath = "../data/products_books_v1.csv"
transactions_filepath = "../data/transactions_books_v1.csv"
products_df = pd.read_csv(products_filepath)
training_transactions_df = pd.read_csv(transactions_filepath)

product_data = {
    "data_context": "books",
    "product_filepath": "data/products_books_v1_10_10.csv",
    "transactions_filepath": "data/transactions_books_v1_10_10.csv",
    "features": ["id", "product_title", "product_image", "product_soup", "product_images"],
    "version": "1.0",
    "unique_name": "_books_v1_10_10",
}
cosineSimilarRecommender = CosineSimilarityRecommender(products=products_df, transactions=training_transactions_df)
cosineSimilarRecommender.load()

wordVecBodyRecommender = WordVecBodyRecommender(products=products_df, transactions=training_transactions_df)
wordVecBodyRecommender.load()

titleWordVecTitleRecommender = TitleWordVecTitleyRecommender(products=products_df, transactions=training_transactions_df)
titleWordVecTitleRecommender.load()



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
        "engine": titleWordVecTitleRecommender
    }
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


# Streamlit app
def main():
    # Initialize session state
    
    # user transactions
    transactions_df = pd.DataFrame(columns=["id", "user_id", "product_id", "rate"])
    
    if "selected_products" not in st.session_state:
        st.session_state.selected_products = []

    st.title("Product Recommendation Demo")

    # Select recommendation engines
    engines = {
        "Cosine Similarity": CosineSimilarity,
    }
    recommend_single_engine = st.selectbox("Select the recommendation engine for single product:", list(engines.keys()))
    recommend_past_engine = st.selectbox("Select the recommendation engine for past transactions:", list(engines.keys()))

    # Select product type
    products_type = "books"  # Support more types in the future

    # Select base case
    base_case = st.selectbox("Select base case:", base_cases(products_type))
    if base_case != "empty_case":
        st.session_state.selected_products = PRODUCT_CASES[products_type][base_case]

    st.write("Selected products:", st.session_state.selected_products)

    # Display products
    st.write("Product Information:")
    st.write(products_df)

    # Search and select products
    selected_products = st.multiselect("Select products to add to cart:", products_df["product_title"], default=st.session_state.selected_products)
    st.session_state.selected_products = selected_products

    # Buy button
    if st.button("Buy"):
        transactions = []
        for product in st.session_state.selected_products:
            product_id = get_product_id(product)
            transaction = {
                "id": len(transactions_df) + 1,
                "user_id": 1,
                "product_id": product_id,
                'product_title': product, 
                "rate": 1
            }
            transactions.append(transaction)
        transactions_df = pd.concat([transactions_df, pd.DataFrame(transactions)], ignore_index=True)
        st.write("Transaction:", transaction)

    # Get recommendations
    if st.button("Get Recommendations"):
        if st.session_state.selected_products:
            
            selected_engine = engines[recommend_single_engine]["engine"]
            
            # recommend_single = engines[recommend_single_engine].recommend_from_past(st.session_state.selected_products)
            # selected_engine = 
            # st.write("Recommendations based on selected products:", recommend_single)
        else:
            st.warning("Please select some products first!")

        if not transactions_df.empty:
            last_transaction = transactions_df.iloc[-1]
            recommend_past = engines[recommend_past_engine].recommend_from_past([last_transaction])
            st.write("Recommendations based on past transaction:", recommend_past)

    # Show transactions
    st.write("Transaction History:")
    st.write(transactions_df)

if __name__ == "__main__":
    main()
