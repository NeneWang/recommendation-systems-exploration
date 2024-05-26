from customrec_engine import RecommendationAbstract, PRODUCT_DATAS, engines_list
import pandas as pd
import pprint

engine_trainning_report = []

for product_data in PRODUCT_DATAS:
    products_filepath = product_data["product_filepath"]
    transactions_filepath = product_data["transactions_filepath"]
    # create for each engine.
    
    productdf = pd.read_csv(products_filepath)
    transactiondf = pd.read_csv(transactions_filepath)
    
    for rec_engine_class in engines_list:
        print("Engine class", rec_engine_class.strategy_name)
        rec_engine: RecommendationAbstract = rec_engine_class(products=productdf, product_data=product_data, transactions=transactiondf)
        rec_engine.train(auto_save=True)
        
        # get some random ids from proct dataframe
        recommendations_arr = []
        report_arr = []
        single_report = 0
        for product_id in productdf[:10]["id"]:
            recommendations_arr.append(product_id)
            try:
                recs = rec_engine.recommend_from_single(product_id)
                product_detail = rec_engine.id_to_productDetail(product_id)
                # Just show the first recommendation product_title
                # report_arr.append(f" {product_detail['product_title']} => {recs[0]['product_title']} ")
                single_report += 1
            except Exception as e:
                print(e)
            
        
        print(recommendations_arr)
        past_recommend_report = 0
        reports_past_report = []
        try:
            past_recommend_report = rec_engine.recommend_from_past(recommendations_arr)
            reports_past_report.append(f" => {past_recommend_report} ")
            past_recommend_report += 1
        except Exception as e:
            print(e)
        
        results = {
            "engine": rec_engine.strategy_name,
            "product_data": product_data["data_context"],
            "single_report_detailed": len(report_arr),
            "single_report": single_report,
            "single_report_total": len(recommendations_arr),
            "past_recommend_report": len(reports_past_report),
            "past_recommend_report_total": 1 
        }
        engine_trainning_report.append(results)
        pprint.pprint(results)

        
    
df_engine_report = pd.DataFrame(engine_trainning_report)
pprint.pprint(df_engine_report)
# store the report in a file
df_engine_report.to_csv("reports/engine_report.csv", index=False)