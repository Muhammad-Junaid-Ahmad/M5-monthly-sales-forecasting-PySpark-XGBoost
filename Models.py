from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.param.shared import HasFeaturesCol, HasPredictionCol, Param, Params
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.sql.functions import pandas_udf, udf
import pandas as pd

"""
Model For XGBoost 
"""

class XGBoostModel(Model, HasFeaturesCol, HasPredictionCol):

    model = Param(Params._dummy(), "model", "parameter for storing saved xgboost booster model", None)

    @keyword_only
    def __init__(self, featuresCol='features', predictionCol='prediction', model=None):
        super(XGBoostModel, self).__init__()
        self._setDefault(featuresCol='features', predictionCol='prediction', model=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, featuresCol=None, predictionCol=None, model=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getModel(self):
        return self.getOrDefault(self.model)

    

    def _transform(self, df: DataFrame):
        featuresCol = self.getFeaturesCol()
        predictionCol = self.getPredictionCol()
        model = self.getModel()

        @udf(returnType=ArrayType(DoubleType()))  
        def getDeVectorizedColumn(denseVector):
            return denseVector.values.tolist()  

        @pandas_udf(DoubleType())
        def getPredictions(features):
            preds = model.predict(features.tolist())
            return pd.Series(preds)
        
        if(df.select(featuresCol).dtypes[0][1] == "vector"):
            df = df.withColumn("devectorized", getDeVectorizedColumn(featuresCol))
            df = df.withColumn(predictionCol, getPredictions("devectorized")).drop("devectorized")
        else:
            df = df.withColumn(predictionCol, getPredictions(featuresCol))
        
        return df
