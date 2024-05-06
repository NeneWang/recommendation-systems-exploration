
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import random
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import string
import re


product_data = {
    "data_context": "books",
    "product_filepath": "data/products_books_v1_10_10.csv",
    "transactions_filepath": "data/transactions_books_v1_10_10.csv",
    "features": ["id", "product_title", "product_image", "product_soup", "product_images"],
    "version": "1.0",
    "unique_name": "_books_v1_10_10",
}



class RecommendationAbstract():
    strategy_name: str = "REQUIRES IMPLEMENTATION"
    version: str = "REQUIRES IMPLEMENTATION"
    details: str = "REQUIRES IMPLEMENTATION"
    link: str = "REQUIRES IMPLEMENTATION"
    supports_single_recommendation: bool = "REQUIRES IMPLEMENTATION"
    supports_past_recommendation: bool = "REQUIRES IMPLEMENTATION"

    def __init__(self, products, product_data):
        self.products = products
        self.product_data = product_data
        self.model = None

    def loadModel(self, model_code):
        """
        Load the model
        """
        self.model = model_code

    def train(self, verbose=False, transactions_train=None, users_train=None):
        """
        Train the model
        """
        # ... do training
        # self.model = trained_model

    def saveModel(self, model_code):
        """
        Save the model
        """
        # ... saves the model

    def id_to_productDetail(self, product_id: str) -> Dict[str, str]:
        """
        Return product details based on product id.
        """
        return self.id_to_products.get(product_id)

    def ids_to_products(self, ids: List[str]) -> List[Dict[str, str]]:
        """
        Return product details for a list of product ids.
        """
        return [self.id_to_productDetail(id) for id in ids]

    def like(self, keyword: str) -> List[str]:
        """
        Return a list of products that contain the given keyword in their title.
        """
        return [product for product in self.products if keyword in product['product_title']]
    def recommend_from_single(self, product_id: str, n=5) -> List[str]:
        """
        Return recommendations based on a single product.
        """
        target_name = self.id_to_productDetail(product_id)['product_title']
        keywords = target_name.split(" ")
        recommendations = []
        for keyword in keywords:
            recommendations.extend(self.like(keyword))
        
        random.shuffle(recommendations)
        return recommendations[:n]

    def recommend_from_past(self, user_transactions, n=10) -> List[str]:
        """
        Return recommendations based on past user transactions.
        """
        rec = []
        for transaction in user_transactions:
            rec.extend(self.recommend_from_single(transaction['product_id']))
        random.shuffle(rec)
        return rec[:n]

class CosineSimilarityRecommender(RecommendationAbstract):
    strategy_name: str = "Cosine Similarity"
    slug_name: str = "cosine_similarity"
    version: str = "v1"
    details: str = "REQUIRES IMPLEMENTATION"
    link: str = "REQUIRES IMPLEMENTATION"
    supports_single_recommendation: bool = True
    supports_past_recommendation: bool = True
    
    def __init__(self, products, product_data):
        super().__init__(products, product_data)
        self.products = products
        self.pt = []
        self.sim_score = None
        
    def train(self, transactions, auto_save=True):
        self.pt = transactions.pivot_table(index="product_id", columns="user_id", values="rate")
        self.pt.fillna(0, inplace=True)
        self.sim_score = cosine_similarity(self.pt)
        if auto_save:
            self.save()
        
        
    def get_filename(self, filename_prepend):
        return  "models/" + self.product_data["unique_name"] + ".pik"
    
    def save(self):
        # Store self.pt
        filename = self.get_filename()
        file_simscr = open(filename, 'wb')
        pickle.dump(self.sim_score, file_simscr)
        file_simscr.close()
        
    def load(self):
        filename = self.get_filename()
        file_simscr = open(filename, 'rb')
        self.sim_score = pickle.load(file_simscr)
        file_simscr.close()
        

    def recommend_from_single(self, product_id, n=5):
        index = np.where(self.products == product_id)[0][0]
        similar_products = sorted(list(enumerate(self.similarity_matrix[index])), key=lambda x: x[1], reverse=True)[1:n+1]
        rec = [self.products[similar_product[0]] for similar_product in similar_products]
        return rec

    def recommend_from_past(self, transactions, n=10):
        rec = []
        for transaction in transactions:
            rec.extend(self.recommend_from_single(transaction['product_id']))
        random.shuffle(rec)
        return rec[:n]

class WordVecBodyRecommender(RecommendationAbstract):
    
    strategy_name: str = "WordVec"
    slug_name: str = "wordvec"
    version: str = "v1"
    details: str = "REQUIRES IMPLEMENTATION"
    link: str = "REQUIRES IMPLEMENTATION"
    supports_single_recommendation: bool = True
    supports_past_recommendation: bool = True
    
    def __init__(self, products, product_data):
        """
        Initialize the recommender with a pre-trained Word2Vec model and a dataframe of books.
        """
        super().__init__(products, product_data)
        self.products_df = products
        self.model = None
        print('id_to_products length', len(self.id_to_products))
        self.train()


    def train(self, auto_save=False):
        """
        Train the Word2Vec model on the book titles.
        """
        sentences = [title.lower().split() for title in self.products_df['product_soup']]
        self.model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
        if auto_save:
            self.save()
            

    def recommend_books_updated(self, input_text, top_n=5):
        """
        Return recommendations based on input text.
        """
        input_text = input_text.lower().split()
        vector = self.model.wv[input_text].mean(axis=0)
        # Compute cosine similarity between the input vector and all product vectors
        similarities = cosine_similarity([vector], self.model.wv.vectors)
        # Get indices of top similar products
        top_indices = similarities.argsort()[0][-top_n:]
        recommended_products = []
        for index in reversed(top_indices):  # Reversed to get top similarities first
            product_title = self.products_df.iloc[index]['product_title']
            confidence = similarities[0][index]
            recommended_products.append((self.id_to_productDetail(self.products_df.iloc[index]['id']), confidence))
        return recommended_products

        
    def recommend_from_single(self, product_id, n=5):
        """
        Return recommendations based on a single product.
        """
        # Get the product_soup for the given product_id
        product_soup = self.products.loc[self.products['id'] == product_id, 'product_soup'].values[0]
        # Use the recommend_books_updated function to recommend books based on the product_soup
        recommendations = self.recommend_books_updated(product_soup, top_n=n)
        return recommendations

    def recommend_from_past(self, transactions, n=10):
        """
        Return recommendations based on past transactions.
        """
        # Concatenate product_soup from past transactions
        past_text = ' '.join(self.products.loc[self.products['id'].isin(transactions), 'product_soup'])
        # Use the self.recommend_books_updated function to recommend books based on past transactions
        recommendations = self.recommend_books_updated(past_text, top_n=n)
        return recommendations
        
        
    def get_filename(self):
        return  "models/" +  self.slug_name + self.product_data["unique_name"] + ".model"

    def save(self):
        """
        Save the computed book vectors to a file.
        """
        filename = self.get_filename()
        filemodel = open(filename, 'wb')
        pickle.dump(self.model, filemodel)
        filemodel.close()

    def load(self):
        """
        Load the book vectors from a file.
        """
        
        filename = self.get_filename()
        filemodel = open(filename, 'rb')
        self.model = pickle.load(filemodel)
        filemodel.close()

class TitleWordVecTitleyRecommender(RecommendationAbstract):
    
    strategy_name: str = "TitleWordVec"
    slug_name: str = "title_word_vec"
    version: str = "v1"
    details: str = "REQUIRES IMPLEMENTATION"
    link: str = "REQUIRES IMPLEMENTATION"
    supports_single_recommendation: bool = True
    supports_past_recommendation: bool = True
    
    def __init__(self, products, product_data):
        """
        Initialize the recommender with a pre-trained Word2Vec model and a dataframe of books.
        """
        super().__init__(products, product_data)
        self.products_df = products
        self.model = None
        self.train()

    def train(self, auto_save=False):
        """
        Train the Word2Vec model on the book titles.
        """
        sentences = [title.lower().split() for title in self.products_df['product_title']]
        self.model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
        if auto_save:
            self.save()

    def recommend_from_single(self, product_id, n=5):
        """
        Return recommendations based on a single product.
        """
        # Get the product_title for the given product_id
        product_title = self.products_df.loc[self.products_df['id'] == product_id, 'product_title'].values[0]
        # Use the recommend_books_updated function to recommend books based on the product_title
        recommendations = self.recommend_books_updated(product_title, top_n=n)
        return recommendations
    
    def recommend_from_past(self, transactions, n=10):
        """
        Return recommendations based on past transactions.
        """
        # Concatenate product_titles from past transactions
        past_titles = self.products_df.loc[self.products_df['id'].isin(transactions), 'product_title']
        # Use the self.recommend_books_updated function to recommend books based on past transactions
        recommendations = self.recommend_books_updated(' '.join(past_titles), top_n=n)
        return recommendations

        

    def recommend_books_updated(self, input_text, top_n=5):
        """
        Return recommendations based on input text.
        """
        input_text = input_text.lower().split()
        vector = self.model.wv[input_text].mean(axis=0)
        # Compute cosine similarity between the input vector and all product vectors
        similarities = cosine_similarity([vector], self.model.wv.vectors)
        # Get indices of top similar products
        top_indices = similarities.argsort()[0][-top_n:]
        recommended_products = []
        for index in reversed(top_indices):  # Reversed to get top similarities first
            product_title = self.products_df.iloc[index]['product_title']
            confidence = similarities[0][index]
            recommended_products.append((self.id_to_productDetail(self.products_df.iloc[index]['id']), confidence))
        return recommended_products

    def save(self):
        """
        Save the computed book vectors to a file.
        """
        filename = self.get_filename()
        with open(filename, 'wb') as filemodel:
            pickle.dump(self.model, filemodel)

    def load(self):
        """
        Load the book vectors from a file.
        """
        filename = self.get_filename()
        with open(filename, 'rb') as filemodel:
            self.model = pickle.load(filemodel)

    def get_filename(self):
        """
        Get the filename for saving/loading the model.
        """
        return "models/" + self.slug_name + self.product_data["unique_name"] + ".model"


