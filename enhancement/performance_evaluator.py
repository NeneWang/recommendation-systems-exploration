import pandas as pd
import time
from customrec_engine import engines_list, RecommendationAbstract, PRODUCT_DATAS
import pprint
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

product_datas = PRODUCT_DATAS
CUSTOM_SEED = 42
REPORT_NAME = "performance_evaluator_v3"

def custom_filename(report_name, customseed):
    filename = f"{report_name}_{customseed}"
    return filename

def create_recommender_report(product_datas, filename=REPORT_NAME, seed=CUSTOM_SEED):
    results = [] 
    for product_data in product_datas:
        pprint.pprint(product_data)

        productdf =  pd.read_csv("../" + product_data["product_filepath"])
        transactiondf = pd.read_csv("../" + product_data["transactions_filepath"])
    
        training_df_arr = []
    
    
    # join transactions by same user_id. into a dict of user_id: [transactions]
        user_transactions = {}
        for row in transactiondf.iterrows():
    # for row in transactiondf.iterrows():
            training_df_arr.append(row[1])
            user_id = row[1]["user_id"]
            if user_id not in user_transactions:
                user_transactions[user_id] = []
            user_transactions[user_id].append(row[1]['product_id'])
    
        training_transactions_users_ids, test_user_ids = train_test_split(list(user_transactions.keys()), test_size=.2, random_state=seed)
    # trainning_usertransactions, test_usertransactions = train_test_split(list(user_transactions.values()), test_size=.2, random_state=42)
        trainning_usertransactions = [user_transactions[user_id] for user_id in training_transactions_users_ids]
        test_usertransactions = [user_transactions[user_id] for user_id in test_user_ids] 
    
    # Create transactiondf from trainning_usertransactions by filtering transactions df where user_id is in trainning_usertransactions
        user_ids = [transaction for transaction in training_transactions_users_ids]
    
        training_transactiondf = pd.DataFrame(training_df_arr)
        training_transactiondf = training_transactiondf[training_transactiondf['user_id'].isin(user_ids)]
    
    # for each engine rec. Train, test:
        start_time = time.time()
        for rec_engine_class in engines_list:
            start_time = time.time()
            print("=========", rec_engine_class.strategy_name, start_time, "=========")
            rec_engine: RecommendationAbstract  = rec_engine_class(products=productdf, product_data=product_data, transactions = training_transactiondf)
            rec_engine.train(auto_save=False)
            hits = []
            true_values = []  # Actual values
            predicted_values = []  # Predicted values
            failures = 0
        
            for user_transactions in test_usertransactions:
                try:
                    if len(user_transactions) < 2:
                        failures += 1
                    # print("skipping user with less than 2 transactions")
                        continue
                
                    past_transactions, pred_transactions = train_test_split(user_transactions, test_size=.25, random_state=42)
                    recs: List[Tuple[dict, float]] = rec_engine.recommend_from_past(past_transactions)
                    if len(recs) == 0:
                        failures += 1
                        print("skipping user with no recommendations")
                        continue
                
                    recommendation_ids = [rec[0]['product_id'] for rec in recs]
                    true_values.append(1)  # Assuming 1 represents a hit
                    hit = 0
                # print('eval', len(recommendation_ids), len(pred_transactions), len(past_transactions))
                    for rec in recommendation_ids:
                        if rec in pred_transactions:
                            hit = 1
                            break
                    predicted_values.append(hit)
                except Exception as e:
                    failures += 1
                    print(e)
        
            accuracy = accuracy_score(true_values, predicted_values)
            precision = precision_score(true_values, predicted_values)
            recall = recall_score(true_values, predicted_values)
        # assert(len(true_values) == len(predicted_values))
        
            end_time = time.time()  # End time for this recommender model
            duration = end_time - start_time  # Duration for this recommender model
            rows_data = {
            "recommender_model": rec_engine_class.strategy_name,
            "unique_name": product_data['unique_name'],
            "hits": sum(predicted_values),
            "out of": sum(true_values),
            "data_context": product_data["data_context"],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "failures": failures,
            "users on test": len(test_usertransactions),
            "users on train": len(trainning_usertransactions),
            "unique_product_count": len(productdf["product_id"].unique()),
            "duration": duration,
        }
        
            try:
                f1 = f1_score(true_values, predicted_values)
                mae = mean_absolute_error(true_values, predicted_values)
                mse = mean_squared_error(true_values, predicted_values)
                r2 = r2_score(true_values, predicted_values)
                rows_data["f1"] = f1
                rows_data["mae"] = mae
                rows_data["mse"] = mse
                rows_data["r2"] = r2
            except Exception as e:
                print('Error calculating f1, mae, mse, r2')
                print(e)
        
            pprint.pprint(rows_data)
            results.append(rows_data)
    
    df_results = pd.DataFrame(results)
    # store results
    df_results.to_csv( filename + "results.csv")
    print("Results stored in results.csv")
    print(df_results.head(40))
    print('=========== FINISHED ===========')


for seed in [42, 43, 44, 45, 46]:
    create_recommender_report(product_datas, custom_filename(REPORT_NAME, seed), seed)

