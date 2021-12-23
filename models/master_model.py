from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.param.shared import HasFeaturesCol, HasPredictionCol, Param, Params
from pyspark.sql import DataFrame
import concurrent.futures
import pyspark.sql.functions as F


class MasterModel(Model, HasFeaturesCol, HasPredictionCol):
    """
    It is a collection of best tunned models for individual estimator types returned by the MasterEstimator's fit method. 
    """

    bestModels = Param(Params._dummy(), "bestModels",
                  "variable for storing tunned collection of various estimator models", None)
    """
    Example bestModels:
    bestModels = [{"model_name": "PySpark_Random_Forest", "model": bestRFModel}, 
                  {"model_name": "XGBoost_Random_Forest", "model": bestXGBModel},
                  {"model_name": "FB_Prophet", "model": bestFBProphetModel}]
    """

    @keyword_only
    def __init__(self, bestModels=None):
        super(MasterModel, self).__init__()
        self._setDefault(bestModels=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, bestModels=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getBestModels(self):
        return self.getOrDefault(self.bestModels)

    def _transform(self, df: DataFrame):

        bestModels = self.getBestModels()
        df.cache()

        futuresList = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for model in bestModels:
                futuresList.append(executor.submit(model["model"].transform, df))

        for i in range(len(futuresList)):
            predCol = bestModels[i]["model"].getPredictionCol()
            df2 = futuresList[i].result().select(["store_id", "year", "month", F.col(predCol).alias(bestModels[i]["model_name"] + "_prediction") ])
            df = df.join(df2, ["store_id", "year", "month"])

        return df
