import pandas as pd
from customrec_engine import engines_list, RecommendationAbstract, PRODUCT_DATAS
import pprint
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

product_datas = PRODUCT_DATAS


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
    
    # create df from transactionsdf
    transactiondf = pd.DataFrame(training_df_arr)
    
    past_transactions, test_transactions = train_test_split(list(user_transactions.values()), test_size=.2, random_state=42)
    
    # for each engine rec. Train, test:
    for rec_engine_class in engines_list:
        print("=========", rec_engine_class.strategy_name, "=========")
        rec_engine: RecommendationAbstract  = rec_engine_class(products=productdf, product_data=product_data, transactions = transactiondf)
        rec_engine.train(auto_save=False)
        hits = []
        true_values = []  # Actual values
        predicted_values = []  # Predicted values
        failures = 0
        
        for user_transactions in test_transactions:
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
        
        results.append({
            "recommender_model": rec_engine_class.strategy_name,
            "unique_name": product_data['unique_name'],
            "hits": sum(hits),
            "out of": len(hits),
            "data_context": product_data["data_context"],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "failures": failures,
            "count_unique_users_tested": len(test_transactions),
            "count_unique_users_train": len(training_df_arr),
        })
        
        try:
            
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
df_results.to_csv("results.csv")
print("Results stored in results.csv")
print(df_results.head(40))
print('=========== FINISHED ===========')


