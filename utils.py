from hyperopt import hp
from hyperopt.pyll.base import scope
import sys
from pyspark.sql import DataFrame
import pyspark

# function for converting a dictionary read from a json file into a hyperopt space
def json_to_space(j):
    space = {}

    for key, value in j.items():
        if value[0] == 'quniform':
            space[key] = hp.quniform(key, *value[2:])
        elif value[0] == 'choice':
            space[key] = hp.choice(key, value[1:])

        if value[1] == 'int':
            space[key] = scope.int(space[key])
        # etc ...

    return space

def splitOnYM(df: DataFrame, ym):
    year = ym["year"]
    month = ym["month"]

    # if type of the DataFrame is not pyspark then it must be pandas. So, pandas uses a function query to filter out rows
    if(type(df) == pyspark.sql.dataframe.DataFrame):
        df1 = df.filter("year<{} or year={} and month<={}".format(year, year, month))
        df2 = df.filter("year>{} or year={} and month>{}".format(year, year, month))
    else:
        df1 = df.query("year<{} or year=={} and month<={}".format(year, year, month))
        df2 = df.query("year>{} or year=={} and month>{}".format(year, year, month))
        
    return df1, df2
    
def selectBestOfBest(bestModels: dict, evaluator, testDF: DataFrame):
    # Evaluating the best models
    """
    bestModels = [{"model_name": "PySpark_Random_Forest", "model": bestRFModel, "loss": 0}, 
                {"model_name": "XGBoost_Random_Forest", "model": bestXGBModel, "loss": 0},
                {"model_name": "FB_Prophet", "model": bestFBProphetModel, "loss": 0}]
    
    evaluator = Evaluators.MAPE(labelCol="sales", predictionCol="prediction")

    """
    
    bestModel = {"model_name": "", "model": None, "loss": sys.float_info.max, "predictions": None}

    for model_info in bestModels:
        print("Evaluating: {}".format(model_info["model_name"]), end='\t')
        preds = model_info["model"].transform(testDF)
        model_info["loss"] = evaluator.evaluate(preds)
        print("Loss: {}".format(model_info["loss"]))

        #Saving the Best Model
        if(bestModel["loss"] > model_info["loss"]):
            bestModel = model_info.copy()
            bestModel["predictions"] = preds
    return bestModel