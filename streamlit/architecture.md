
## Implementation Streamlit Pseudocode

Lets implement so that we can have the following:

```python

engines = {
    "CosineSimilarity": CosineSimilarity,
    ...
}

recommend_single_engine = st.dropdown("Select the recommendation engine", engines.keys(), default="CosineSimilarity")
recommend_past_engine = st.dropdown("Select the recommendation engine", engines.keys(), default="CosineSimilarity")
transactions = []

products_type = "books" # Support more in the future
PRODUCT_CASES = {
    "books": {"Harry Potter Fan": [{product_id: xxxx, rate: 5}], "Sci Fi Fan": {...}, ...}
}

def base_cases(products_type):
    return ["empty_case"] + PRODUCT_CASES[products_type]

if(st.dropdown(base_cases(products_type))):
    transactions = base_cases(products_type)


st.write("Recommendation Engine: ", recommend_single_engine)
st.write("select prodcuts")

st.search_df(products)

st.multi_select("Select the products", products)

if (st.button("buy")):
    transaction = {
        "user_id": "user1",
        "product_id": st.session_state.selected_products,
        "rate": 5}
    transactions.extend(transaction)
    recommend = engines[recommend_single_engine].recommend_from_past(transactions)


# Show the recommendations
st.carousel(recommend)

# Show the user transaction historial
st.df(transactions)


```


Implemented Algorithms:

- CosineSimilarityRecommender
- SoupVecRecommender
- TitleVecRecommender