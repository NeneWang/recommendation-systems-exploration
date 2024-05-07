# Developmenet of the Recommendation Engine.

There are 3 main steps to develop the recommendation engine:

1. Data Reformating.
2. Building a Common Recommendation Engine Abstract Class
3. Using the common class to report on the accuracy of each model


## Data Reformating. Data to be reformated and saved as following structure:

transactions:

| name       | type | description                          |
| ---------- | ---- | ------------------------------------ |
| id         | str  | Unique identifier of the transaction |
| user_id    | str  | Unique identifier of the user        |
| product_id | str  | Unique identifier of the product     |
| rate       | int  | Positive association rating          |

Note:
**Rate**: The rate is a value between 1 and 5. 1 being the lowest and 5 being the highest. 
In some cases if the calification is using buy/view/etc will be codified as buy: 5, cart: 3, view: 1. etc.

products:

| name           | type    | description                               |
| -------------- | ------- | ----------------------------------------- |
| id             | str     | Unique identifier of the product          |
| product_title  | str     | Title of the product                      |
| product_image  | str\nan | Image of the product                      |
| product_price  | int\nan | Price of the product                      |
| product_soup   | str     | All Aggregated Description of the product |
| product_images | str\nan | List of images of the product             |
| product_tags   | str\nan | List of tags of the product, sep by comma |

**Products Metadata/Structure Organization**

```json
[
    {
        "data_context": "books",
        "products_filepath": "data/products_books_v1.csv",
        "transactions_filepath": "data/transactions_books_v1.csv",
        "features": ["product_title", "product_image", "product_soup", "product_images"],
        "version": "1.0",
        "cleanup_tags": ["products remove products where under 10 interactions", "remove users under 4 valid transactions"],
        "unique_name": "books_v1",
    }
]
```

users:

Users demographics will not be considered. Thus the user will be a simple id.



## Building an recommendation engine that allows for all recommmendations/

Run all them.


Develop the pseudocode for each:


```python

class RecommendationAbstract:
    strategy_name:str = ...
    version:str = ...
    details:str = ...
    link:str = ...
    supports_single_recommendation:bool = ...
    supports_past_recommendation:bool = ...

    def __init__(self, products):
        self.products = products

    def loadModel(self, model_code):
        """
        Load the data
        """
        self.model = model_code
        

    def train(verbose=False, transactions_train, users_train):
        """
        Train the model
        """
        ... do trainning

        self.model = trained_model

        pass

    def saveModel(self, model_code):
        """
        Save the model
        """
        # ... saves in the path, common method shared.

    def id_to_productDetail(str) -> Dict[str]:
        return self.id_to_products

    def ids_to_products(str) -> List[str] -> List[Dict[str]]:
        for id in ids:
            self.id_to_productDetail(id)

    def like(str) -> List[str]:
        """
        Return a list of products that are similar to the given string.
        """
        return [product for product in self.products if str in product.product_title]

    def recommned_from_single(product_id, n=5) -> List[tuple[str, float, dict]]:
        """Overwrite or default implementation
        
        """
        target_name = self.id_to_productDetail(product_id).product_title
        keywords = target_name.split(" ")
        recommendations = []
        for keyword in keywords:
            recommendations.extends(self.like(keyword))
        
        random.shuffle(recommendations)
        return recommendations[:, n]

    def recommend_from_past(user_transactions, n=10) -> List[tuple[str, float, dict]]:
        
        rec = []
        for transaction in user_transactions:
            rec.extend(self.recommend_from_single(transaction.product_id))
        random.shuffle(rec)
        return rec[:n]

class CosineSimilarity(RecommendationAbstract):
    # Complete Cosine Similarity Details

    def __init__(self, products):
        super().__init__(products)
        similarity_score = cosine_similarity(products)


    def recommend_from_single(str, n=5):
        index = np.where(self.products == str)[0][0]
        similar_products = sorted(list(enumerate(self.similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]

        rec = []
        for similar_product in similar_products:
            rec.append(self.products[similar_product[0]])
        return rec


    def recommend_from_past(transactions, n=10):
        rec = []
        for transaction in transactions:
            rec.extend(self.recommend_from_single(transaction.product_id))
        random.shuffle(rec)
        return rec[:n]


from gensim.models import Word2Vec
import numpy as np
import random

class Word2VecRecommender(RecommendationAbstract):
    def __init__(self, products, embedding_size=100, window=5, min_count=1):
        super().__init__(products)
        # Train Word2Vec model
        self.model = Word2Vec(sentences=products, vector_size=embedding_size, window=window, min_count=min_count)
        # Build product embeddings
        self.product_embeddings = {product: self.model.wv[product] for product in self.products}

    def recommend_from_single(self, product, n=5):
        similar_products = self.model.wv.most_similar(product, topn=n)
        rec = [(product[0], product[1]) for product in similar_products]
        return rec

    def recommend_from_past(self, transactions, n=10):
        rec = []
        for transaction in transactions:
            rec.extend(self.recommend_from_single(transaction.product_id))
        # Sort recommendations based on similarity scores
        rec.sort(key=lambda x: x[1], reverse=True)
        return rec[:n]



```

## Using the common class to report on the accuracy of each model

- [ ] Designing input and Output
- [ ] Design basic tests.
- [ ] Input should be the item selected



Designing Performance Tests.

Follwing are the tests to run. There are two cases of performances to run:

- [ ] Products Recommendations

This works by understanding the following:
- This is more subjective. Does the recommendation of this product make sense? (wont be evaluated, just runned through common sense) And see automatically the difference. Perhaps the best commute recommendation is by the products that all share in common.

- [ ] User Transaction Recommendations

- This can sort of be evaluated. But since they are different by each comparison, almost aomporing apple with oranges. But the idea is to see if the recommendation makes sense.


Pseudocode
```python


# Initialize each recommender with the appropriate products data.
list_of_rec_engines[RecommendationAbstract] = [...]
# For each recommedation engine, start trainning the model.
for rec_engine in list_of_rec_engines:
    rec_engine.train(transactions, users)

# Data clean up
transactions_cleanup = [clean_up(transaction) for transaction in transactions]
remove_users_with_less_than_6_transactions(users, transactions)

userA, userB = split(users)
user_transactions_a, user_transactions_b = split_transaction_by_users(transactions)
"""
user_transactions_1: 
{
    "user_id": [transaction1: {user, product, rate}, transaction2: {user, product, rate}]
}
"""


# Train the models
for rec_engine in list_of_rec_engines:
    rec_engine.train(transactions_a, userA)


"""
scores = {
    "rec_engine1": 0.5,
    "rec_engine2": 0.6,
}
"""
scores = {}
# Using hit, miss, estimate the score of the model: accuracy, precision, recall, f1-score
# accuracy: (hit + miss) / total
# precision: hit / (hit + miss)
# recall: hit / (hit + miss)
# f1-score: 2 * (precision * recall) / (precision + recall)

# Test the models
for rec_engine in list_of_rec_engines:
    
    score = 0
    for transaction in transactions:
        past_transactions: List[id],  predict_partition:List[id] = extract_xy_transaction(transaction)
        recommendations = rec_engine.recommend_from_past(past_transactions)
        
        hit = 0
        miss = 0
        # If any of the recommendations are 
        for recommnedation in recommnedations:
            if recommendation in predict_partition:
                accurate += 1
            else:
                miss -= 1
    
    # Store the score of the rec_engine
    scores[rec_engine] = score

```

Item selected shoudl have things



Lets have those at the 



### word2vec

Here few questiosn:

- Does a different intiiation make a difference?

```python
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
```

```
import gensim.downloader

# Show all available models in gensim-data

print(list(gensim.downloader.info()['models'].keys()))
['fasttext-wiki-news-subwords-300',
 'conceptnet-numberbatch-17-06-300',
 'word2vec-ruscorpora-300',
 'word2vec-google-news-300',
 'glove-wiki-gigaword-50',
 'glove-wiki-gigaword-100',
 'glove-wiki-gigaword-200',
 'glove-wiki-gigaword-300',
 'glove-twitter-25',
 'glove-twitter-50',
 'glove-twitter-100',
 'glove-twitter-200',
 '__testing_word2vec-matrix-synopsis']
>>>

# Download the "glove-twitter-25" embeddings

glove_vectors = gensim.downloader.load('glove-twitter-25')
>>>

# Use the downloaded vectors as usual:

glove_vectors.most_similar('twitter')
[('facebook', 0.948005199432373),
 ('tweet', 0.9403423070907593),
 ('fb', 0.9342358708381653),
 ('instagram', 0.9104824066162109),
 ('chat', 0.8964964747428894),
 ('hashtag', 0.8885937333106995),
 ('tweets', 0.8878158330917358),
 ('tl', 0.8778461217880249),
 ('link', 0.8778210878372192),
 ('internet', 0.8753897547721863)]
```

For some reason, when combined the words, it takes otehr words, and I think, it would be better if it knew which word to prioritize otherwise the combinated sentences seems to not be able to find anything useful


```python

# Test the updated function Kings the same book title
recommend_books_updated("Pale Kings and Prince")

['Clara Callan',
 'Cunt: A Declaration of Independence (Live Girls)',
 'Callander Square',
 'Cunt: A Declaration of Independence (Live Girls Series)']
['New Vegetarian: Bold and Beautiful Recipes for Every Occasion',
 'Prague : A Novel',
 'Seabiscuit: An American Legend',
 'Pigs in Heaven',
 'This Year It Will Be Different: And Other Stories']



# Test the updated function Kings the same book title
recommend_books_updated("Kings")
['Pigs in Heaven',
 'Poisonwood Bible Edition Uk',
 'The Game of Kings (Lymond Chronicles, 1)',
 'The Bean Trees',
 'Homeland and Other Stories']


 
# Test the updated function Kings the same book title
recommend_books_updated("Pale")
['Pale Fire: A Novel (Vintage International)',
 'On a Pale Horse',
 'Pale Horse Coming',
 'Pale Kings and Princes',
 'On a Pale Horse (Incarnations of Immortality, Bk. 1)']

```


Here the problem is tha the searcher assigns equa values to all words of the sentece when searching. But clearly there are words with more meaning. Like King, and Prince.

TextRank: This algorithm is based on Google's PageRank algorithm and treats words in the sentence as nodes in a graph. TextRank calculates the importance of each word based on its connectivity to other words in the sentence. Words with many connections to other words are considered more important. You can rank the words in the sentence using TextRank and select the top-ranked words as the most impactful.


https://summanlp.github.io/textrank/


```python
pip install summa

from summa import keywords

# Input sentence
sentence = "Pale Kings and Prince"

# Extract keywords using TextRank algorithm
keywords_list = keywords.keywords(sentence)

print("Keywords:", keywords_list)

```

https://summanlp.github.io/textrank/


```
>>> from gensim.summarization.summarizer import summarize
>>> text = '''Rice Pudding - Poem by Alan Alexander Milne
... What is the matter with Mary Jane?
... She's crying with all her might and main,
... And she won't eat her dinner - rice pudding again -
... What is the matter with Mary Jane?
... What is the matter with Mary Jane?
... I've promised her dolls and a daisy-chain,
... And a book about animals - all in vain -
... What is the matter with Mary Jane?
... What is the matter with Mary Jane?
... She's perfectly well, and she hasn't a pain;
... But, look at her, now she's beginning again! -
... What is the matter with Mary Jane?
... What is the matter with Mary Jane?
... I've promised her sweets and a ride in the train,
... And I've begged her to stop for a bit and explain -
... What is the matter with Mary Jane?
... What is the matter with Mary Jane?
... She's perfectly well and she hasn't a pain,
... And it's lovely rice pudding for dinner again!
... What is the matter with Mary Jane?'''
>>> print(summarize(text))
And she won't eat her dinner - rice pudding again -
I've promised her dolls and a daisy-chain,
I've promised her sweets and a ride in the train,
And it's lovely rice pudding for dinner again!
```


```python
>>> from gensim.summarization import keywords
>>> text = '''Challenges in natural language processing frequently involve
... speech recognition, natural language understanding, natural language
... generation (frequently from formal, machine-readable logical forms),
... connecting language and machine perception, dialog systems, or some
... combination thereof.'''
>>> keywords(text).split('\n')
[u'natural language', u'machine', u'frequently']
```

It was removed.

3.8.3 best pre-4.0 version to use, if you must. But Gensim's summarization code was removed because it only offered a weak form of 'extractive' summarization – guessing important sentences in text – in way that wasn't very performant, consistent w/ other Gensim code, or customizable/maintainable/improvable. I've never seen a good demo showing it work well, & while a few rare use cases in English may have found it helpful, I believe most users who tried it left feeling it was a frustrating waste of time. Newer techniques, especially LLM-based, can do far better 'abstractive' summarization. – 
gojomo
May 20, 2023 at 

- Neither Summa nor Summarization keywords seems to work in these cases.

It seems that it requieres of a longer description.

```
from gensim.summarization import keywords

# Example text
text = "Pale Kings and Princes is a Spenser novel by Robert B. Parker. The title is taken from John Keats's poem La Belle Dame sans Merci: A Ballad. Following the murder of a reporter, Spenser is hired by a newspaper to investigate drug smuggling around the area of Wheaton, Massachusetts."""

# Getting keywords
key_words = keywords(text)

print(key_words)

```


TO work instead with short titles:

```python
import spacy

# Load the English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Your text
text = "Pale Kings and Prince"

# Process the text
doc = nlp(text)

# Extract nouns
nouns = [token.text for token in doc if token.pos_ == "NOUN"]

print("Nouns:", nouns)

```


Here a code for noun > verb > adjective search


```python
import spacy

# Load the English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Your text
text = "Pale Kings and Prince"

# Process the text
doc = nlp(text)

# Try extracting nouns first
keywords = [token.text for token in doc if token.pos_ == "NOUN"]

# If no nouns, fall back to verbs
if not keywords:
    keywords = [token.text for token in doc if token.pos_ == "VERB"]

# If still no keywords, consider adjectives
if not keywords:
    keywords = [token.text for token in doc if token.pos_ == "ADJ"]

print("Keywords:", keywords)
```






