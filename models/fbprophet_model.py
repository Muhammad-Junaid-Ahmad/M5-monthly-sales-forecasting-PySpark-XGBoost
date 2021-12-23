from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.param.shared import HasFeaturesCol, HasPredictionCol, Param, Params
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType
import pandas as pd
from copy import deepcopy
from pyspark.ml.param import TypeConverters
import pickle as pkl


class FBProphetModel(Model, HasFeaturesCol, HasPredictionCol):
    """
    Model For FB Prophet 
    """

    models = Param(Params._dummy(), "models",
                   "A pandas dataframe for storing trained fbprophet models for each individual group of data specified by dataGroupCols", None)
    """
    The keys for the dictionary will be the strings joined by "_". 
    For Example dataGroupCols = ["store_id", "dept_id"], the keys should be: "STORE_1_DEPT_1", "STORE_1_DEPT_2", ...., "STORE_N_DEPT_N" 

    Example model:
    model = {"STORE_1_DEPT_1": fbprophet trained model on data filtered on STORE_1 and DEPT_1, 
             "STORE_1_DEPT_2": fbprophet trained model on data filtered on STORE_1 and DEPT_2,
             ...,
             ...,
             "STORE_N_DEPT_N": fbprophet trained model on data filtered on STORE_N and DEPT_N}
    """

    dataGroupCols = Param(Params._dummy(), "dataGroupCols",
                              "A list of columns on which data is grouped and individual models are trained on", TypeConverters.toListString)

    @keyword_only
    def __init__(self, featuresCol='ds', predictionCol='fbp_prediction', models=None, dataGroupCols=None):
        super(FBProphetModel, self).__init__()
        self._setDefault(featuresCol='ds', predictionCol='fbp_prediction',
                         models=None, dataGroupCols=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, featuresCol=None, predictionCol=None, models=None, dataGroupCols=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getModels(self):
        return self.getOrDefault(self.models)

    def getDataGroupCols(self):
        return self.getOrDefault(self.dataGroupCols)

    def _transform(self, df: DataFrame):
        featuresCol = self.getFeaturesCol()
        predictionCol = self.getPredictionCol()
        dataGroupCols = self.getDataGroupCols()
        models = self.getModels().to_dict()
        # The models["key"] dictionary is in the form {index: key}. Changing it to {key: index}. for ease of use.
        models["key"] = dict( [ (v,k) for k,v in models["key"].items() ] )

        for c in df.dtypes:
            if(c[1] == "vector"):
                df = df.drop(c[0])

        schema = deepcopy(df.schema).add(predictionCol, DoubleType())
        
        def getPredictions(keys, pdf: DataFrame):
            key = "_".join([str(k) for k in keys])

            model = pkl.loads( models["model"][models["key"][key]] )
            
            pdf["ds"] = pd.to_datetime(pdf[featuresCol])

            predsDF = (model.predict(pdf)[["ds", "yhat"]])\
                            .merge(pdf, on=["ds"], how="left")\
                            .rename(columns={"yhat": predictionCol})

            return pd.DataFrame(predsDF[schema.fieldNames()], columns=schema.fieldNames())
        
        df = df.groupBy(dataGroupCols).applyInPandas(getPredictions, schema=schema)
        
        return df