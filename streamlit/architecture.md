
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


```python
import io  # noqa

from surprise import Dataset, get_dataset_dir, KNNBaseline


def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and return two
    mappings to convert raw ids into movie names and movie names into raw ids.
    """

    file_name = get_dataset_dir() + "/ml-100k/ml-100k/u.item"
    rid_to_name = {}
    name_to_rid = {}
    with open(file_name, encoding="ISO-8859-1") as f:
        for line in f:
            line = line.split("|")
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid


# First, train the algorithm to compute the similarities between items
data = Dataset.load_builtin("ml-100k")
trainset = data.build_full_trainset()
sim_options = {"name": "pearson_baseline", "user_based": False}
algo = KNNBaseline(sim_options=sim_options)
algo.fit(trainset)

# Read the mappings raw id <-> movie name
rid_to_name, name_to_rid = read_item_names()

# Retrieve inner id of the movie Toy Story
toy_story_raw_id = name_to_rid["Toy Story (1995)"]
toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)

# Retrieve inner ids of the nearest neighbors of Toy Story.
toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)

# Convert inner ids of the neighbors into names.
toy_story_neighbors = (
    algo.trainset.to_raw_iid(inner_id) for inner_id in toy_story_neighbors
)
toy_story_neighbors = (rid_to_name[rid] for rid in toy_story_neighbors)

print()
print("The 10 nearest neighbors of Toy Story are:")
for movie in toy_story_neighbors:
    print(movie)

```


- [ ] I am looking here, and I should absolutely get the similitude trying not from books but here the `m1-100k` What other datasets are available there?



```python
BUILTIN_DATASETS = {
    "ml-100k": BuiltinDataset(
        url="https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        path=join(get_dataset_dir(), "ml-100k/ml-100k/u.data"),
        reader_params=dict(
            line_format="user item rating timestamp", rating_scale=(1, 5), sep="\t"
        ),
    ),
    "ml-1m": BuiltinDataset(
        url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        path=join(get_dataset_dir(), "ml-1m/ml-1m/ratings.dat"),
        reader_params=dict(
            line_format="user item rating timestamp", rating_scale=(1, 5), sep="::"
        ),
    ),
    "jester": BuiltinDataset(
        url="https://eigentaste.berkeley.edu/dataset/archive/jester_dataset_2.zip",
        path=join(get_dataset_dir(), "jester/jester_ratings.dat"),
        reader_params=dict(line_format="user item rating", rating_scale=(-10, 10)),
    ),
}

```

