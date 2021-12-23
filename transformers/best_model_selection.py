from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasLabelCol, HasValidationIndicatorCol, Param, Params, TypeConverters
from pyspark.sql import DataFrame
import pyspark
import sys
import concurrent.futures
from evaluators import MAPE



class BestModelSelection(Transformer, HasLabelCol, HasValidationIndicatorCol):
    """
    Tranformer for Selecting the best model out of list of different tunned models
    """

    evaluator = Param(Params._dummy(), "evaluator",
                             "evaluator object for measuring accuracy", None)
    """
    Example hyperParamsSpace:
    MAPE
    """

    models = Param(Params._dummy(), "models",
                  "variable for storing tunned collection of various estimator models", None)
    """
    Example models:
    models = [{"model_name": "PySpark_Random_Forest", "model": bestRFModel}, 
                  {"model_name": "XGBoost_Random_Forest", "model": bestXGBModel},
                  {"model_name": "FB_Prophet", "model": bestFBProphetModel}]
    """
    
    @keyword_only
    def __init__(self, evaluator, models, labelCol, validationIndicatorCol):
        """
        models => List(Dict). every dict must be {"model_name": String, "model": instance_of_any_Model} 
        """
        super(BestModelSelection, self).__init__()

        self._setDefault(evaluator=None, labelCol=None, validationIndicatorCol=None, models=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, evaluator=None, labelCol=None, validationIndicatorCol=None, models=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getEvaluator(self):
        return self.getOrDefault(self.evaluator)
    
    def getModels(self):
        return self.getOrDefault(self.models)

    def _transform(self, df: DataFrame):
        evaluator = self.getEvaluator()
        validationIndicatorCol = self.getValidationIndicatorCol()
        labelCol = self.getLabelCol()
        models = self.getModels()

        trainDF = df.filter("{} < 2".format(validationIndicatorCol)).cache()
        testDF = df.filter("{} = 2".format(validationIndicatorCol)).cache()

        
        futuresList = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for model in models:
                predCol = model["model_name"] + "_prediction"
                futuresList.append({"train_loss" : executor.submit(evaluator(labelCol=labelCol, predictionCol=predCol).evaluate, trainDF),
                                    "test_loss" :  executor.submit(evaluator(labelCol=labelCol, predictionCol=predCol).evaluate, testDF)})
        bestModelInd = 0
        for i in range(len(futuresList)):
            models[i]["train_loss"] = futuresList[i]["train_loss"].result()
            models[i]["test_loss"] = futuresList[i]["test_loss"].result()

            if(models[i]["test_loss"] < models[bestModelInd]["test_loss"]):
                bestModelInd = i
            
        
        trainDF.unpersist()
        testDF.unpersist()

        return df, models, bestModelInd
