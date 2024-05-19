import pandas as pd
from customrec_engine import engines_list, RecommendationAbstract, PRODUCT_DATAS
import pprint
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time


product_datas = PRODUCT_DATAS
REPORT_NAME = "performance_evaluator_timed_prio_products_500_20"
total_start_time = time.time()  # Start time for total computation


results = [] 
for product_data in product_datas[2:]:
    pprint.pprint(product_data)

    productdf =  pd.read_csv("../" + product_data["product_filepath"])
    transactions_df = pd.read_csv("../" + product_data["transactions_filepath"])
    
    training_df_arr = []
    
    
    # join transactions by same user_id. into a dict of user_id: [transactions]
    user_transactions = {}
    
    # This solution to limit users seems to be better, as it takes advatage of the speedy sort from the dataframe.
    count_users_to_limit = 100
    count_valid_users = 0
    
    
    training_transactiondf = transactions_df.sort_values(by=['product_id'])
    for row in transactions_df.iterrows():
    # for row in transactiondf.iterrows():
        training_df_arr.append(row[1])
        user_id = row[1]["user_id"]
        if user_id not in user_transactions:
            user_transactions[user_id] = []
        else:
            if len(user_transactions[user_id]) >= 10:
                count_valid_users += 1
        user_transactions[user_id].append(row[1]['product_id'])
        if count_valid_users >= count_users_to_limit:
            break
    
    trainning_usertransactions, test_usertransactions = train_test_split(list(user_transactions.values()), test_size=.2, random_state=42)
    training_transactions_users_ids, _ = train_test_split(list(user_transactions.keys()), test_size=.2, random_state=42)
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
                
                train_transactions, pred_transactions = train_test_split(user_transactions, test_size=.25, random_state=42)
                recs: List[Tuple[dict, float]] = rec_engine.recommend_from_past(train_transactions)
                
                if len(recs) == 0:
                    failures += 1
                    print("skipping user with no recommendations")
                    continue
                
                recommendation_ids = [rec[0]['product_id'] for rec in recs]
                hit = 0
                for rec in recommendation_ids:
                    if rec in pred_transactions:
                        hit = 1
                        break
                hits.append(hit)
                true_values.append(1)  # Assuming 1 represents a hit
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
            "hits": sum(hits),
            "out of": len(hits),
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
        pprint.pprint(rows_data)
        
        results.append(rows_data)
        
        try:
            total_end_time = time.time() 
            f1 = f1_score(true_values, predicted_values)
            
            mae = mean_absolute_error(true_values, predicted_values)
            mse = mean_squared_error(true_values, predicted_values)
            r2 = r2_score(true_values, predicted_values)
            results["f1"] = f1
            results["mae"] = mae
            results["mse"] = mse
            results["r2"] = r2
        except Exception as e:
            print(e)
            pass
        
        
        
df_results = pd.DataFrame(results)
# store results
df_results.to_csv( REPORT_NAME+ "results.csv")
print("Results stored in results.csv")
print(df_results.head(40))
print('=========== FINISHED ===========')


