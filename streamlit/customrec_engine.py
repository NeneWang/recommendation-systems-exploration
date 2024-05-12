
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
import spacy
import Tuple
from surprise import KNNBasic, KNNWithZScore, KNNBaseline, KNNWithMeans

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

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
        # populate id_to_products
        self.id_to_products = {}
        for product in self.products.to_dict(orient='records'):
            self.id_to_products[product['id']] = product


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

    # Implementation of the class using cosine Similarity.

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
        
        
    def get_filename(self):
        return "models/" + self.slug_name + self.product_data["unique_name"] + ".pik"
    
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
        

    def recommend_from_single(self, product_id, n=5) -> List[tuple[dict, float]]:
        # Find the index of the product_id in the DataFrame
        # print('product_id', product_id)
        index = np.where(self.products['id'] == product_id)[0][0]
        
        similar_products = []
        # Get similarity scores for the product at the found index
        try:
            similar_products = sorted(enumerate(self.sim_score[index]), key=lambda x: x[1], reverse=True)[1:n+1]
        except Exception as e:
            print('checl - self sim score', self.sim_score)
            print('checl - index', index)
            print('Error', e)
        
        # Retrieve the similar products using their indices and return them
        recommendations_list = []
        for similar_product in similar_products:
            product_index, score = similar_product
            product_dict = self.products.iloc[product_index].to_dict()
            recommendations_list.append((product_dict, score))
        
        return recommendations_list


    def recommend_from_past(self, transactions, n=10):
        print('transactions received', transactions)
        rec: List[tuple[dict, float]] = []
        for transaction in transactions:
            rec.extend(self.recommend_from_single(transaction))
        
        # Sort by the confidence (second parameter of tuple)
        sorted_rec: List[tuple[dict, float]] = sorted(rec, key=lambda x: x[1], reverse=True)
        return sorted_rec[:n]
    

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


    def train(self, auto_save=True):
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

        
    def recommend_from_single(self, product_id, n=5) -> List[tuple[dict, float]]:
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
        return  "models/" + self.slug_name + self.product_data["unique_name"] + ".model"

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
    
class TitleWordVecTitleyRecommenderV2(RecommendationAbstract):
    """
    Key Changes:
    - Using nlp to search first nouns>verbs>adjectives
    - Past Transactions search instead of the aggregated titles, makes individual search with prioritization with eah title
    """
    strategy_name: str = "TitleWordVec"
    slug_name: str = "title_word_vec"
    version: str = "v2"
    details: str = "REQUIRES IMPLEMENTATION"
    link: str = "REQUIRES IMPLEMENTATION"
    supports_single_recommendation: bool = True
    supports_past_recommendation: bool = True
    
    def __init__(self, products, product_data, useKeyword=True):
        """
        Initialize the recommender with a pre-trained Word2Vec model and a dataframe of books.
        """
        super().__init__(products, product_data)
        self.products_df = products
        self.model = None
        self.train()
        self.nlp = spacy.load("en_core_web_sm")
        self.useKeyword = useKeyword # Otherwise uses the list of keywords concatenated found.

    def train(self, auto_save=False):
        """
        Train the Word2Vec model on the book titles.
        """
        # Preprocess text data
        self.products_df['processed_soup'] = self.products_df['product_title'].str.lower().str.translate(str.maketrans('', '', string.punctuation))

        # Prepare self.products_df for Word2Vec model
        sentences = [row.split() for row in self.products_df['processed_soup'].dropna()]
        self.model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
        if auto_save:
            self.save()


    def getMostSignificantKeyword(self, text, default_to_text = True, k=3) -> tuple[str, str]:
        """
        Extracts keywords from the provided text, prioritizing nouns, then verbs, then adjectives.
        Continues until the combined length of keywords is at least 3 characters.
        
        This algorithm is designed to work for short range titles.
        - default_to_
        
        Returns a tuple of (most_relevant_keyword, an_concatenated_list_of_keywords)
        """
        doc = self.nlp(text)
        priorityKey = ""
        top_keywords = []
        fullsearch = ""

        # Define the order of part of speech tags to search based on their relevance
        pos_priority = ["NOUN", "VERB", "ADJ", "PROPN", "ADV", "PRON", "ADP", "CCONJ", "SCONJ", "DET", "AUX", "NUM",
                        "PART", "INTJ", "SYM", "PUNCT", "X"]

        # Iterate over each part of speech in priority order
        for pos in pos_priority:
            keywords = [token.text for token in doc if token.pos_ == pos]
            for key in keywords:
                fullsearch += f" {key}"  # Append all found keywords to fullsearch
                if len(key) > len(priorityKey):
                    priorityKey = key  # Update priorityKey if a longer keyword is found
                if len(top_keywords) < k:
                    top_keywords.append(key)  # Add the keyword to top_keywords if there are less than k
                if len(priorityKey) >= k and len(top_keywords) == k:
                    break  # Break both loops if condition is met
            if len(priorityKey) >= k and len(top_keywords) == k:
                break  # Break outer loop if condition is met
        # If there are more than k words, and not enough top words, to fulfill k. Add as many words as possible to get to k.
        if len(top_keywords) < k:
            set_keywords = set(top_keywords)
            for word in text.split():
                if word not in set_keywords:
                    set_keywords.add(word)
                if len(top_keywords) == k:
                    break
            set_keywords = list(set_keywords)
        
        # print(priorityKey, ' '.join(top_keywords), fullsearch)
        return (' '.join(top_keywords), fullsearch)
        

    def recommend_from_single(self, product_id, n=5, verbose=True, greedy_attempt=3) -> List[tuple[str, float]]:
        """
        Return recommendations based on a single product.
        """
        # Get the product_title for the given product_id
        product_title: str = self.products_df.loc[self.products_df['id'] == product_id, 'product_title'].values[0]
        # Use the recommend_books_updated function to recommend books based on the product_title
        recommendations = []
        search_term = ""
        seen = set(product_title.lower())
        for k in range(greedy_attempt):
            keyword, keywords_concat = self.getMostSignificantKeyword(product_title, k=k+1)
            if(self.useKeyword):
                search_term = keyword
            else:
                search_term = keywords_concat
            if verbose:
                print(f"Searching for '{search_term}' from '{product_title}'")
                # print(recommendations)
        
            rec = self.recommend_books_updated(search_term, top_n=n)
            for rec_item, confidence_rate in rec:
                # print(rec_item)
                if rec_item['product_title'].lower() not in seen:
                    seen.add(rec_item['product_title'].lower())
                    recommendations.append((rec_item, confidence_rate))
                else:
                    continue
            if len(recommendations) >= greedy_attempt:
                break
        return recommendations
    
    def recommend_from_past(self, transactions, n=10):
        """
        Return recommendations based on past transactions.
        Does the following:
        
        @param transactions: List[id] = List of transactions
        
        - Per each transaction uses recommend_from_single, to find relevant books. around 5 recommendations.
        - ensures that the recommendations are unique.
        - Sorts by confidence.
        - limits to n.
        """
    
        for transaction in transactions:
            rec: List[tuple[dict, int]] = self.recommend_from_single(transaction)
            rec.extend(rec) 
        # Because there could be repeated rec[i]['product_title'] we need to remove duplicates.
        seen_titles = set()
        unique_rec = []
        for rec_item, confidence_rate in rec:
            if rec_item['product_title'] not in seen_titles:
                seen_titles.add(rec_item['product_title'])
                unique_rec.append((rec_item, confidence_rate))
                    
        
        # Sort by confidence second parameter
        unique_rec.sort(key=lambda x: x[1])
        return unique_rec[:n]
        

    def recommend_books_updated(self, input_text, top_n=5):
        """
        Return recommendations based on input text.
        """
        input_text = input_text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        vector = self.model.wv[input_text].mean(axis=0)
        similar_vectors = self.model.wv.similar_by_vector(vector, topn=top_n + 10)  # Retrieve more results to filter unique titles
        recommended_titles = []
        for book_vector in similar_vectors:
            similar_title = self.products_df.loc[self.products_df['processed_soup'].apply(lambda x: any(word in x for word in input_text)), 'processed_soup'].unique()
            for title in similar_title:
                if title not in recommended_titles and len(recommended_titles) < top_n:
                    product = self.products_df.loc[self.products_df['processed_soup'] == title].iloc[0].to_dict()
                    recommended_titles.append((product, book_vector[1]))
        return recommended_titles

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
    

class KNNBasicRecommender(RecommendationAbstract):
    strategy_name: str = "KNN Basic"
    slug_name: str = "knn_basic"
    version: str = "v1"
    details: str = "REQUIRES IMPLEMENTATION"
    link: str = "REQUIRES IMPLEMENTATION"
    supports_single_recommendation: bool = True
    supports_past_recommendation: bool = True
    
    def __init__(self, products: pd.DataFrame, product_data: dict, transactions = None):
        super().__init__(products, product_data)
        self.products = products
        self.model = None
        
        # Get the product ids and store them.
        self.product_ids = self.products['id'].unique()
        self.all_transactions_df = transactions
        
    def train(self, transactions, auto_save=True, dont_save_self_state=False) :
        
        sim_options = {"name": "pearson_baseline", "user_based": False}
        model = KNNBasic(sim_options=sim_options)
        
        reader = Reader(rating_scale=(1, 5))
        
        data = Dataset.load_from_df(transactions[['user_id', 'product_id', 'rate']], reader)
        
        model.fit(data.build_full_trainset())
        
        if dont_save_self_state:
            return model
        
        self.model = model
        self.all_transactions_df = transactions
        # self.accuracy = accuracy.rmse(model.test(data.build_full_trainset().build_testset()), verbose=True)
        
        if auto_save:
            self.save()
            
        return model
        
        
    def get_filename(self):
        return "models/" + self.slug_name + self.product_data["unique_name"] + ".pik"
    
    def save(self):
        # Store self.pt
        filename = self.get_filename()
        model_file = open(filename, 'wb')
        pickle.dump(self.model, model_file)
        model_file.close()
        
    def load(self, auto_create=True):
        
        filename = self.get_filename()
        try:
            model_file = open(filename, 'rb')
            self.model = pickle.load(model_file)
            model_file.close()
        except:
            self.save()
            

    def recommend_from_single(self, product_id: str, n=5) -> List[Tuple[dict, float]]:
        """
        
        # Retrieve inner ids of the nearest neighbors of Toy Story.
        toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)
        """
        recommendation_list: List[tuple[dict, float]] = []
        product_inner_id = self.model.trainset.to_inner_iid(product_id)
        neighbors = self.model.get_neighbors(product_inner_id, k=n*2)
        
        # for each neighbor, try to predict and prioritize given a user in all_transactions_that shared that book as well.
        for neighbor_book_inner_id in neighbors:
            # get user_id that top rated the product sort the relevant_transactions
            if neighbor_book_inner_id == product_inner_id:
                continue
            
            product_serie = self.products.iloc[neighbor_book_inner_id]
            neighbor_book_id = product_serie['id']
            relevant_transactions = self.all_transactions_df[self.all_transactions_df['product_id'] == neighbor_book_id]
            relevant_transactions = relevant_transactions.sort_values(by='rate', ascending=False)
            # remove where  product_id product_id
            if len(relevant_transactions) == 0:
                continue
            
            user_id = relevant_transactions.iloc[0]['user_id']
            
            pred = self.model.predict(user_id, neighbor_book_id)
            recommendation_list.append((self.id_to_products[neighbor_book_id], pred.est))
        
        # sort recommendations
        recommendation_list.sort(key=lambda x: x[1], reverse=True)
        return recommendation_list[:n]

    def collaborativestore_predict_population(self, transactions: List[str], n=5):
        """
        Adds the transactions to the use history to be considered when training the model. Doesnt not save the model with this transactions,
        proceeds to use the models to create recommendations. This is pattern was added for KNN and Matrix Factorizations
        """
        # Add transactions to the self.transactions_df as a new user
        transaction_rows = []
        random_user_id = "user" + str(random.randint(0, 1000000))
        for transaction in transactions:
            transaction_rows.append({'user_id': 'user_id', 'product_id': transaction, 'rate': 5})
        
        # Convert to a DataFrame
        new_transactions_df = pd.DataFrame(transaction_rows)

        # Append using concat
        all_transactions_df: pd.Dataframe = pd.concat([self.all_transactions_df, new_transactions_df], ignore_index=True)
        
        model = self.train(all_transactions_df, dont_save_self_state=True)
        
        return self.predict_recommendations(random_user_id, transactions, model, n)
    
    def predict_recommendations(self, user_id: str, transactions: List[str], model, n=5):
        books_to_predict = [book_id for book_id in self.product_ids if book_id not in transactions]
        predictions = []
        
        for book_id in books_to_predict:
            pred = model.predict(user_id, book_id)
            predictions.append((book_id, pred.est))
        
        pred_products = []
        # sort predictions
        predictions.sort(key=lambda x: x[1], reverse=True)
        for book_id, confidence in predictions[:n]:
            product = self.id_to_products[book_id]
            pred_products.append(product)
            
        return pred_products
        

    def recommend_from_past(self, transactions: List[str], n=10):
        """
        Calls for each transaction the recommend_from_single method.
        Gives Priority if seen multiple recommendations.
        Shuffle and returns :n
        """
        recs = set()
        recs_seen_times = {}
        products_dictionary = {}
        
        # Deprecated.
        # if(len(transactions) > 2):
        #     return self.collaborativestore_predict_population(
        #         transactions, n=n
        #     )
        
        for transaction in transactions:
            recs = self.recommend_from_single(transaction)
            for rec_id, confidence in recs:
                
                if rec_id in recs:
                    recs_seen_times[rec_id['id']] += 1
                else:
                    products_dictionary[rec_id['id']] = rec_id
                    recs_seen_times[rec_id['id']] = 1
        
        for rec_id in recs_seen_times:
            recs.append((products_dictionary[rec_id], recs_seen_times[rec_id]))
            
        recs = list(recs)
        
        recs.sort(key=lambda x: x[1], reverse=True)
        return recs
    
class KNNWithZScoreRecommender(RecommendationAbstract):
    strategy_name: str = "KNN With ZScore"
    slug_name: str = "knn_with_zscore"
    version: str = "v1"
    details: str = "REQUIRES IMPLEMENTATION"
    link: str = "REQUIRES IMPLEMENTATION"
    supports_single_recommendation: bool = True
    supports_past_recommendation: bool = True
    
    def __init__(self, products: pd.DataFrame, product_data: dict, transactions = None):
        super().__init__(products, product_data)
        self.products = products
        self.model = None
        
        # Get the product ids and store them.
        self.product_ids = self.products['id'].unique()
        self.all_transactions_df = transactions
        
    def train(self, transactions, auto_save=True, dont_save_self_state=False) :
        
        sim_options = {"name": "pearson_baseline", "user_based": False}
        model = KNNWithZScore(sim_options=sim_options)
        
        reader = Reader(rating_scale=(1, 5))
        
        data = Dataset.load_from_df(transactions[['user_id', 'product_id', 'rate']], reader)
        
        model.fit(data.build_full_trainset())
        
        if dont_save_self_state:
            return model
        
        self.model = model
        self.all_transactions_df = transactions
        # self.accuracy = accuracy.rmse(model.test(data.build_full_trainset().build_testset()), verbose=True)
        
        if auto_save:
            self.save()
            
        return model
        
        
    def get_filename(self):
        return "models/" + self.slug_name + self.product_data["unique_name"] + ".pik"
    
    def save(self):
        # Store self.pt
        filename = self.get_filename()
        model_file = open(filename, 'wb')
        pickle.dump(self.model, model_file)
        model_file.close()
        
    def load(self, auto_create=True):
        
        filename = self.get_filename()
        try:
            model_file = open(filename, 'rb')
            self.model = pickle.load(model_file)
            model_file.close()
        except:
            self.save()
        

    def recommend_from_single(self, product_id: str, n=5) -> List[Tuple[dict, float]]:
        """
        
        # Retrieve inner ids of the nearest neighbors of Toy Story.
        toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)
        """
        recommendation_list: List[tuple[dict, float]] = []
        product_inner_id = self.model.trainset.to_inner_iid(product_id)
        neighbors = self.model.get_neighbors(product_inner_id, k=n*2)
        
        # for each neighbor, try to predict and prioritize given a user in all_transactions_that shared that book as well.
        for neighbor_book_inner_id in neighbors:
            product_serie = self.products.iloc[neighbor_book_inner_id]
            neighbor_book_id = product_serie['id']
            relevant_transactions = self.all_transactions_df[self.all_transactions_df['product_id'] == neighbor_book_id]
            # get user_id that top rated the product sort the relevant_transactions
            relevant_transactions = relevant_transactions.sort_values(by='rate', ascending=False)
            
            user_id = relevant_transactions.iloc[0]['user_id']
            
            pred = self.model.predict(user_id, neighbor_book_id)
            recommendation_list.append((self.id_to_products[neighbor_book_id], pred.est))
        
        # sort recommendations
        recommendation_list.sort(key=lambda x: x[1], reverse=True)
        return recommendation_list[:n]

    def collaborativestore_predict_population(self, transactions: List[str], n=5):
        """
        Adds the transactions to the use history to be considered when training the model. Doesnt not save the model with this transactions,
        proceeds to use the models to create recommendations. This is pattern was added for KNN and Matrix Factorizations
        """
        # Add transactions to the self.transactions_df as a new user
        transaction_rows = []
        random_user_id = "user" + str(random.randint(0, 1000000))
        for transaction in transactions:
            transaction_rows.append({'user_id': 'user_id', 'product_id': transaction, 'rate': 5})
        
        # Convert to a DataFrame
        new_transactions_df = pd.DataFrame(transaction_rows)

        # Append using concat
        all_transactions_df: pd.Dataframe = pd.concat([self.all_transactions_df, new_transactions_df], ignore_index=True)
        
        model = self.train(all_transactions_df, dont_save_self_state=True)
        
        return self.predict_recommendations(random_user_id, transactions, model, n)
    
    def predict_recommendations(self, user_id: str, transactions: List[str], model, n=5):
        books_to_predict = [book_id for book_id in self.product_ids if book_id not in transactions]
        predictions = []
        
        for book_id in books_to_predict:
            pred = model.predict(user_id, book_id)
            predictions.append((book_id, pred.est))
        
        pred_products = []
        # sort predictions
        predictions.sort(key=lambda x: x[1], reverse=True)
        for book_id, confidence in predictions[:n]:
            product = self.id_to_products[book_id]
            pred_products.append(product)
            
        return pred_products
        

    def recommend_from_past(self, transactions: List[str], n=10):
        """
        Calls for each transaction the recommend_from_single method.
        Gives Priority if seen multiple recommendations.
        Shuffle and returns :n
        """
        recs = set()
        recs_seen_times = {}
        products_dictionary = {}
        
        # Deprecated.
        # if(len(transactions) > 2):
        #     return self.collaborativestore_predict_population(
        #         transactions, n=n
        #     )
        
        for transaction in transactions:
            recs = self.recommend_from_single(transaction)
            for rec_id, confidence in recs:
                
                if rec_id in recs:
                    recs_seen_times[rec_id['id']] += 1
                else:
                    products_dictionary[rec_id['id']] = rec_id
                    recs_seen_times[rec_id['id']] = 1
        
        for rec_id in recs_seen_times:
            recs.append((products_dictionary[rec_id], recs_seen_times[rec_id]))
            
        recs = list(recs)
        
        recs.sort(key=lambda x: x[1], reverse=True)
        return recs
    
class KNNWithBaselineRecommender(RecommendationAbstract):
    strategy_name: str = "KNN With Means"
    slug_name: str = "knn_with_baseline"
    version: str = "v1"
    details: str = "REQUIRES IMPLEMENTATION"
    link: str = "REQUIRES IMPLEMENTATION"
    supports_single_recommendation: bool = True
    supports_past_recommendation: bool = True
    
    def __init__(self, products: pd.DataFrame, product_data: dict, transactions = None):
        super().__init__(products, product_data)
        self.products = products
        self.model = None
        
        # Get the product ids and store them.
        self.product_ids = self.products['id'].unique()
        self.all_transactions_df = transactions
        
    def train(self, transactions, auto_save=True, dont_save_self_state=False) :
        
        sim_options = {"name": "pearson_baseline", "user_based": False}
        model = KNNBaseline(sim_options=sim_options)
        
        reader = Reader(rating_scale=(1, 5))
        
        data = Dataset.load_from_df(transactions[['user_id', 'product_id', 'rate']], reader)
        
        model.fit(data.build_full_trainset())
        
        if dont_save_self_state:
            return model
        
        self.model = model
        self.all_transactions_df = transactions
        # self.accuracy = accuracy.rmse(model.test(data.build_full_trainset().build_testset()), verbose=True)
        
        if auto_save:
            self.save()
            
        return model
        
        
    def get_filename(self):
        return "models/" + self.slug_name + self.product_data["unique_name"] + ".pik"
    
    def save(self):
        # Store self.pt
        filename = self.get_filename()
        model_file = open(filename, 'wb')
        pickle.dump(self.model, model_file)
        model_file.close()
        
    def load(self):
        filename = self.get_filename()
        model_file = open(filename, 'rb')
        self.model = pickle.load(model_file)
        model_file.close()
        

    def recommend_from_single(self, product_id: str, n=5) -> List[Tuple[dict, float]]:
        """
        
        # Retrieve inner ids of the nearest neighbors of Toy Story.
        toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)
        """
        recommendation_list: List[tuple[dict, float]] = []
        product_inner_id = self.model.trainset.to_inner_iid(product_id)
        neighbors = self.model.get_neighbors(product_inner_id, k=n*2)
        
        # for each neighbor, try to predict and prioritize given a user in all_transactions_that shared that book as well.
        for neighbor_book_inner_id in neighbors:
            product_serie = self.products.iloc[neighbor_book_inner_id]
            neighbor_book_id = product_serie['id']
            relevant_transactions = self.all_transactions_df[self.all_transactions_df['product_id'] == neighbor_book_id]
            # get user_id that top rated the product sort the relevant_transactions
            relevant_transactions = relevant_transactions.sort_values(by='rate', ascending=False)
            
            user_id = relevant_transactions.iloc[0]['user_id']
            
            pred = self.model.predict(user_id, neighbor_book_id)
            recommendation_list.append((self.id_to_products[neighbor_book_id], pred.est))
        
        # sort recommendations
        recommendation_list.sort(key=lambda x: x[1], reverse=True)
        return recommendation_list[:n]

    def collaborativestore_predict_population(self, transactions: List[str], n=5):
        """
        Adds the transactions to the use history to be considered when training the model. Doesnt not save the model with this transactions,
        proceeds to use the models to create recommendations. This is pattern was added for KNN and Matrix Factorizations
        """
        # Add transactions to the self.transactions_df as a new user
        transaction_rows = []
        random_user_id = "user" + str(random.randint(0, 1000000))
        for transaction in transactions:
            transaction_rows.append({'user_id': 'user_id', 'product_id': transaction, 'rate': 5})
        
        # Convert to a DataFrame
        new_transactions_df = pd.DataFrame(transaction_rows)

        # Append using concat
        all_transactions_df: pd.Dataframe = pd.concat([self.all_transactions_df, new_transactions_df], ignore_index=True)
        
        model = self.train(all_transactions_df, dont_save_self_state=True)
        
        return self.predict_recommendations(random_user_id, transactions, model, n)
    
    def predict_recommendations(self, user_id: str, transactions: List[str], model, n=5):
        books_to_predict = [book_id for book_id in self.product_ids if book_id not in transactions]
        predictions = []
        
        for book_id in books_to_predict:
            pred = model.predict(user_id, book_id)
            predictions.append((book_id, pred.est))
        
        pred_products = []
        # sort predictions
        predictions.sort(key=lambda x: x[1], reverse=True)
        for book_id, confidence in predictions[:n]:
            product = self.id_to_products[book_id]
            pred_products.append(product)
            
        return pred_products
        

    def recommend_from_past(self, transactions: List[str], n=10):
        """
        Calls for each transaction the recommend_from_single method.
        Gives Priority if seen multiple recommendations.
        Shuffle and returns :n
        """
        recs = set()
        recs_seen_times = {}
        products_dictionary = {}
        
        # Deprecated.
        # if(len(transactions) > 2):
        #     return self.collaborativestore_predict_population(
        #         transactions, n=n
        #     )
        
        for transaction in transactions:
            recs = self.recommend_from_single(transaction)
            for rec_id, confidence in recs:
                
                if rec_id in recs:
                    recs_seen_times[rec_id['id']] += 1
                else:
                    products_dictionary[rec_id['id']] = rec_id
                    recs_seen_times[rec_id['id']] = 1
        
        for rec_id in recs_seen_times:
            recs.append((products_dictionary[rec_id], recs_seen_times[rec_id]))
            
        recs = list(recs)
        
        recs.sort(key=lambda x: x[1], reverse=True)
        return recs
    
class KNNWithMeansRecommender(RecommendationAbstract):
    strategy_name: str = "KNN With Means"
    slug_name: str = "knn_with_means"
    version: str = "v1"
    details: str = "REQUIRES IMPLEMENTATION"
    link: str = "REQUIRES IMPLEMENTATION"
    supports_single_recommendation: bool = True
    supports_past_recommendation: bool = True
    
    def __init__(self, products: pd.DataFrame, product_data: dict, transactions = None):
        super().__init__(products, product_data)
        self.products = products
        self.model = None
        
        # Get the product ids and store them.
        self.product_ids = self.products['id'].unique()
        self.all_transactions_df = transactions
        
    def train(self, transactions, auto_save=True, dont_save_self_state=False) :
        
        sim_options = {"name": "pearson_baseline", "user_based": False}
        model = KNNWithMeans(sim_options=sim_options)
        
        reader = Reader(rating_scale=(1, 5))
        
        data = Dataset.load_from_df(transactions[['user_id', 'product_id', 'rate']], reader)
        
        model.fit(data.build_full_trainset())
        
        if dont_save_self_state:
            return model
        
        self.model = model
        self.all_transactions_df = transactions
        # self.accuracy = accuracy.rmse(model.test(data.build_full_trainset().build_testset()), verbose=True)
        
        if auto_save:
            self.save()
            
        return model
        
        
    def get_filename(self):
        return "models/" + self.slug_name + self.product_data["unique_name"] + ".pik"
    
    def save(self):
        # Store self.pt
        filename = self.get_filename()
        model_file = open(filename, 'wb')
        pickle.dump(self.model, model_file)
        model_file.close()
        
    def load(self):
        filename = self.get_filename()
        model_file = open(filename, 'rb')
        self.model = pickle.load(model_file)
        model_file.close()
        

    def recommend_from_single(self, product_id: str, n=5) -> List[Tuple[dict, float]]:
        """
        
        # Retrieve inner ids of the nearest neighbors of Toy Story.
        toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)
        """
        recommendation_list: List[tuple[dict, float]] = []
        product_inner_id = self.model.trainset.to_inner_iid(product_id)
        neighbors = self.model.get_neighbors(product_inner_id, k=n*2)
        
        # for each neighbor, try to predict and prioritize given a user in all_transactions_that shared that book as well.
        for neighbor_book_inner_id in neighbors:
            product_serie = self.products.iloc[neighbor_book_inner_id]
            neighbor_book_id = product_serie['id']
            relevant_transactions = self.all_transactions_df[self.all_transactions_df['product_id'] == neighbor_book_id]
            # get user_id that top rated the product sort the relevant_transactions
            relevant_transactions = relevant_transactions.sort_values(by='rate', ascending=False)
            
            user_id = relevant_transactions.iloc[0]['user_id']
            
            pred = self.model.predict(user_id, neighbor_book_id)
            recommendation_list.append((self.id_to_products[neighbor_book_id], pred.est))
        
        # sort recommendations
        recommendation_list.sort(key=lambda x: x[1], reverse=True)
        return recommendation_list[:n]

    def collaborativestore_predict_population(self, transactions: List[str], n=5):
        """
        Adds the transactions to the use history to be considered when training the model. Doesnt not save the model with this transactions,
        proceeds to use the models to create recommendations. This is pattern was added for KNN and Matrix Factorizations
        """
        # Add transactions to the self.transactions_df as a new user
        transaction_rows = []
        random_user_id = "user" + str(random.randint(0, 1000000))
        for transaction in transactions:
            transaction_rows.append({'user_id': 'user_id', 'product_id': transaction, 'rate': 5})
        
        # Convert to a DataFrame
        new_transactions_df = pd.DataFrame(transaction_rows)

        # Append using concat
        all_transactions_df: pd.Dataframe = pd.concat([self.all_transactions_df, new_transactions_df], ignore_index=True)
        
        model = self.train(all_transactions_df, dont_save_self_state=True)
        
        return self.predict_recommendations(random_user_id, transactions, model, n)
    
    def predict_recommendations(self, user_id: str, transactions: List[str], model, n=5):
        books_to_predict = [book_id for book_id in self.product_ids if book_id not in transactions]
        predictions = []
        
        for book_id in books_to_predict:
            pred = model.predict(user_id, book_id)
            predictions.append((book_id, pred.est))
        
        pred_products = []
        # sort predictions
        predictions.sort(key=lambda x: x[1], reverse=True)
        for book_id, confidence in predictions[:n]:
            product = self.id_to_products[book_id]
            pred_products.append(product)
            
        return pred_products
        

    def recommend_from_past(self, transactions: List[str], n=10):
        """
        Calls for each transaction the recommend_from_single method.
        Gives Priority if seen multiple recommendations.
        Shuffle and returns :n
        """
        recs = set()
        recs_seen_times = {}
        products_dictionary = {}
        
        # Deprecated.
        # if(len(transactions) > 2):
        #     return self.collaborativestore_predict_population(
        #         transactions, n=n
        #     )
        
        for transaction in transactions:
            recs = self.recommend_from_single(transaction)
            for rec_id, confidence in recs:
                
                if rec_id in recs:
                    recs_seen_times[rec_id['id']] += 1
                else:
                    products_dictionary[rec_id['id']] = rec_id
                    recs_seen_times[rec_id['id']] = 1
        
        for rec_id in recs_seen_times:
            recs.append((products_dictionary[rec_id], recs_seen_times[rec_id]))
            
        recs = list(recs)
        
        recs.sort(key=lambda x: x[1], reverse=True)
        return recs
    



cosineSimilarRecommender = CosineSimilarityRecommender
# cosineSimilarRecommender.load()

wordVecBodyRecommender = WordVecBodyRecommender
# wordVecBodyRecommender.load()

titleWordVecTitleRecommender = TitleWordVecTitleyRecommender
# titleWordVecTitleRecommender.load()

titleWordVectRecommender2 = TitleWordVecTitleyRecommenderV2


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
    "KNN Basic": {
        "title": "KNN Basic",
        "engine": KNNBasicRecommender
    },
    "matrix_factorization KNNWithMeans": {
        "title": "Base KNNWithMeans",
        "engine": KNNWithMeansRecommender
    },
    
    "matrix_factorization KNNWithZscore": {
        "title": "Base KNNWithMeans",
        "engine": KNNWithZScoreRecommender
    },
    "matrix_factorization KNNWithBaseline": {
        "title": "Base KNNWithMeans",
        "engine": KNNWithBaselineRecommender
    },
    "matrix_factorization Means Recommender": {
        "title": "KNN Means reocmmender",
        "engine": KNNWithMeansRecommender
    },   
    
    "matrix_factorization SVD": {
        "title": "Base Algorithm",
        "engine": cosineSimilarRecommender
    },
    "matrix_factorization SVD++": {
        "title": "Base Algorithm",
        "engine": cosineSimilarRecommender
    },
}

