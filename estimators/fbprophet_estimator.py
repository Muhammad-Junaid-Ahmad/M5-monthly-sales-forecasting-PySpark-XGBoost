from pyspark import keyword_only
from pyspark.ml import Estimator
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasMaxIter, HasPredictionCol, Param, Params, TypeConverters
from pyspark.sql import DataFrame
from hyperopt import fmin, Trials, tpe, STATUS_OK
from functools import partial
import fbprophet
import pandas as pd
import pickle as pkl
from transformers import DFSplitter
from models import FBProphetModel
import utils

class FBProphetEstimator(Estimator, HasLabelCol, HasFeaturesCol, HasPredictionCol, HasMaxIter, DFSplitter):
    """
    FB Prophet Estimator
    """

    hyperParamsSpace = Param(Params._dummy(), "hyperParamsSpace",
                             "HyperOpt object for tunning hyperparameters for the Random Forest Regressor", None)
    """
    Example hyperParamsSpace:
    hyperParamsSpace = {
        "changepoint_prior_scale": hp.quniform("changepoint_prior_scale", 0.001, 0.5, 0.001),
        "seasonality_prior_scale": hp.quniform("seasonality_prior_scale", 0.01, 10, 0.01),
        "seasonality_mode": hp.choice("seasonality_mode", ["additive", "multiplicative"])
    }
    
    All the labels must be the same as names of positional arguments to the fbprophet's Prophet class.
    """
    train_validation_split = Param(Params._dummy(), "train_validation_split",
                                   "A dictionary containing year and month on which to split the dataframe into train and validate set", None)
    """
    Example train_validation_split:
    train_validation_split = {"year": 2015, "month": 0}
    """

    dataGroupCols = Param(Params._dummy(), "dataGroupCols",
                          "A list of columns on which data is grouped and individual models are trained on", TypeConverters.toListString)
    """
    Example dataGroupCols = ["store_id", "dept_id"]
    The key to store each group's trained model should be: "STORE_1_DEPT_1", "STORE_1_DEPT_2", ...., "STORE_N_DEPT_N"
    """

    @keyword_only
    def __init__(
            self,
            featuresCol=None,
            labelCol=None,
            hyperParamsSpace=None,
            maxIter=20,
            predictionCol="fbp_prediction",
            train_validation_split=None,
            dataGroupCols=None
    ):
        kwargs = self._input_kwargs
        
        super(FBProphetEstimator, self).__init__()
        self._setDefault(featuresCol=None, labelCol=None, hyperParamsSpace=None, maxIter=20,
                         predictionCol="fbp_prediction", train_validation_split=None, dataGroupCols=None)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
            self,
            featuresCol=None,
            labelCol=None,
            hyperParamsSpace=None,
            maxIter=None,
            predictionCol=None,
            train_validation_split=None,
            dataGroupCols=None
    ):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getHyperParams(self):
        return self.getOrDefault(self.hyperParamsSpace)

    def getTrainValidateSplit(self):
        return self.getOrDefault(self.train_validation_split)

    def getDataGroupCols(self):
        return self.getOrDefault(self.dataGroupCols)

    def tuneParams(self, df, hyperParam):
        labelCol = self.getLabelCol()
        featuresCol = self.getFeaturesCol()
        dataGroupCols = self.getDataGroupCols()
        trainValSplit = self.getTrainValidateSplit()
        #splitFunc = DFSplitter.splitOnYM

        def train_evaluate(keys, pdf: DataFrame):

            pdf["y"] = pdf[labelCol]
            pdf["ds"] = pd.to_datetime(pdf[featuresCol])
            trainDF, valDF = utils.splitOnYM(pdf, trainValSplit)

            model = fbprophet.Prophet(weekly_seasonality=False, daily_seasonality=False, **hyperParam)\
                             .fit(trainDF)

            preds = model.predict(valDF)[["ds", "yhat"]]\
                         .merge(valDF, on=["ds"], how="left")

            mape = (abs((preds["y"] - preds["yhat"]) / preds["y"])).mean() * 100 #MAPE.evaluate(preds) #0#Evaluators.MAPE(labelCol="y", predictionCol="yhat").evaluate(preds)
            
            key = "_".join([str(k) for k in keys])

            return pd.DataFrame(data=[[key, pkl.dumps(hyperParam), mape]],
                                columns=["key", "hyper_params", "loss"])

        modelsParams = df.groupBy(dataGroupCols)\
            .applyInPandas(train_evaluate, schema="key string, hyper_params binary, loss double").toPandas()

        if(self.bestModelsParams.empty):
            self.bestModelsParams = modelsParams
        else:
            # Choosing the rows from bestModelsParams and modelsParams which have the lowest loss
            # for the same id. Means for that particular data group we found parameters that best fits it.
            self.bestModelsParams = pd.merge_ordered(self.bestModelsParams, modelsParams)\
                                      .drop_duplicates("key", keep="first", ignore_index=True)
            
        avg_loss = self.bestModelsParams["loss"].mean() #self.bestModelsParams.select("loss").cache().groupBy().avg().first()[0] 

        return {'loss': avg_loss, 'status': STATUS_OK, 'models_params': modelsParams}

    def _fit(self, df: DataFrame):
        labelCol = self.getLabelCol()
        featuresCol = self.getFeaturesCol()
        hyperParamsSpace = self.getHyperParams()
        maxIter = self.getMaxIter()
        dataGroupCols = self.getDataGroupCols()

        # This will be used to save the hyper params of the best model for every group seen so far.
        self.bestModelsParams = pd.DataFrame()

        df = df.select(dataGroupCols + [featuresCol, labelCol, "year", "month"]).cache()

        print("Tunning FBProphet hyper-parameters")
        trials = Trials()
        bestParams = fmin(partial(self.tuneParams, df),
                          space=hyperParamsSpace, algo=tpe.suggest, max_evals=maxIter, trials=trials)

        bestParams = self.bestModelsParams.to_dict()
        
        # The bestParams["key"] dictionary is in the form {index: key}. Changing it to {key: index}. for ease of use.
        bestParams["key"] = dict( [ (v,k) for k,v in bestParams["key"].items() ] )
        
        def reTrainBest(keys, pdf: DataFrame):    
            pdf["y"] = pdf[labelCol]
            pdf["ds"] = pd.to_datetime(pdf[featuresCol])
            key = "_".join([str(k) for k in keys])

            hyperParam = pkl.loads( bestParams["hyper_params"][bestParams["key"][key]] )

            model = fbprophet.Prophet(weekly_seasonality=False, daily_seasonality=False, **hyperParam)\
                             .fit(pdf)

            return pd.DataFrame(data=[[key, pkl.dumps(model), pkl.dumps(hyperParam)]],
                                columns=["key", "model", "hyper_params"])

        bestModels = df.groupBy(dataGroupCols)\
            .applyInPandas(reTrainBest, schema="key string, model binary, hyper_params binary").toPandas()
        
        return FBProphetModel(featuresCol=featuresCol, models=bestModels, dataGroupCols=dataGroupCols)
