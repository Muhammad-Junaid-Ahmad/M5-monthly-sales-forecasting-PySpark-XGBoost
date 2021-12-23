# %%
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from transformers import *
from estimators import *
from evaluators import *
import json
import utils
import sys

# %%
# Create a SparkSession
spark = SparkSession.builder.master("local[*]")\
                            .appName("M5-forecasting")\
                            .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
                            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")\
                            .getOrCreate()

spark

# %%
# Reading Files

sales = spark.read.option("header", "true").option("inferSchema", "true")\
    .csv("./M5-forecasting/sales_train_evaluation.csv")

# sales.printSchema()

calendar = spark.read.option("header", "true").option("inferSchema", "true")\
    .csv("./M5-forecasting/calendar.csv")

# calendar.printSchema()

config = json.load(open("config.json"))
# config


# %%
# Estimator definition and decalrations
space_pyspark = utils.json_to_space(config["hp_space_pyspark_RF"])
rf_est = RandomForestEstimator(featuresCol="features", labelCol="sales", 
                                               hyperParamsSpace=space_pyspark, maxIter=5, 
                                               train_validation_split=config["train_validation_split"])

space_xgboost = utils.json_to_space(config["hp_space_xgboost_RF"])
xgb_est = XGBoostEstimator(featuresCol="features", labelCol="sales", 
                                           hyperParamsSpace=space_xgboost, maxIter=5,
                                           train_validation_split=config["train_validation_split"])

space_fbprophet = utils.json_to_space(config["hp_space_fbprophet"])
fbp_est  = FBProphetEstimator(featuresCol="ds", labelCol="sales", dataGroupCols=["store_id"],
                                                    hyperParamsSpace=space_fbprophet, maxIter=5,
                                                    train_validation_split=config["train_validation_split"])

estimators_and_models_info = [{"model_name": "RF", "estimator": rf_est}, 
                              {"model_name": "XGB", "estimator": xgb_est},
                              {"model_name": "FBP", "estimator": fbp_est}]
# estimators_and_models_info = [{"model_name": "RF", "estimator": rf_est}, {"model_name": "FBP", "estimator": fbp_est}]

# %%
# Feature Engineering
fltr = FilterDF(filterCond="dept_id == '{}'".format(config["train_on_dept_id"]))
explode_days = ExplodeDays(inputCols=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
join_dfs = DFsJoiner(joinOn={"df1_col": "day", "df2_col": "d"}, df2=calendar)
groupBy = GroupByTransformer(groupByCols=["store_id", "year", "month"], aggExprs={"sales": "sum"})
log_sales = LogTransformer(inputCols=["sales"])
lag_sales = LagFeatures(inputCol="sales", lagVals=config["lag_values_to_create"], partCols=["store_id"], orderCols=["year", "month"])
fltr_null_lags = FilterDF(filterCond="lag_sales_{} is not null".format(config["lag_values_to_create"][-1]))
store_indxr = StringIndexer(inputCol="store_id", outputCol="store_id_index")
vectorize = VectorAssembler(inputCols=["store_id_index", "month", "year"] + ["lag_sales_" + str(i) for i in range(1, 13)], outputCol="features")
date_stamp = GenerateDSTransformer(inputCols=["year", "month"], outputCol="ds")

train_test_val_ind = TrainTestValIndicator(outputCol="train_test_val_indicator", 
                                           indicatorCond = {"train_before": config["train_validation_split"], 
                                                            "test_after": config["train_test_split"] })

#This will fit all the given estimators and return a global model containing list of all trained models
#its fit method will also update every dict of the "estimators_and_models_info" list with key "model" containing the fitted model
run_estimators = MasterEstimator(estimatorsToFit = estimators_and_models_info) 
best_model_selection = BestModelSelection(evaluator=MAPE, models=estimators_and_models_info, validationIndicatorCol="train_test_val_indicator", labelCol="sales")


df, models, best_model_ind = Pipeline(stages=[fltr, 
                                            explode_days, 
                                            join_dfs, 
                                            groupBy, 
                                            log_sales, 
                                            lag_sales, 
                                            fltr_null_lags, 
                                            store_indxr, 
                                            vectorize, 
                                            date_stamp,
                                            train_test_val_ind,
                                            run_estimators,
                                            best_model_selection])\
                                        .fit(sales)\
                                        .transform(sales)

df.cache()

# %%
# Saving predictions of best model into CSV
predCol = models[best_model_ind]["model_name"] + "_prediction"
best_pred_df = df.filter("train_test_val_indicator == 2")\
                 .select(["store_id", "year", "month", "sales", predCol])\
                 .orderBy("store_id", "year", "month")
best_pred_df = AntiLogTransformer(inputCols=["sales", predCol]).transform(best_pred_df)
best_pred_df.toPandas()\
            .to_csv("{}_{}_forecasts.csv".format(models[best_model_ind]["model_name"], 
                                                 config["train_on_dept_id"]), 
                    index=False, header=True)


