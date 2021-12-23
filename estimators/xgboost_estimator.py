from pyspark import keyword_only
from pyspark.ml import Estimator
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasMaxIter, HasPredictionCol, Param, Params
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.sql.functions import avg, udf, pandas_udf
import evaluators
from hyperopt import fmin, Trials, tpe, STATUS_OK
from functools import partial
from models import XGBoostModel
import xgboost
from transformers import DFSplitter


class XGBoostEstimator(Estimator, HasLabelCol, HasFeaturesCol, HasPredictionCol, HasMaxIter, DFSplitter):
    """
    XGBoost Estimator
    """
    
    hyperParamsSpace = Param(Params._dummy(), "hyperParamsSpace",
                             "HyperOpt object for tunning hyperparameters for the Random Forest Regressor", None)
    """
    Example hyperParamsSpace:
    hyperParamsSpace = {"n_estimators": scope.int(hp.quniform("n_estimators", 15,25,1)), "max_depth": scope.int(hp.quniform("max_depth", 50,70,1)),
                        "subsample": hp.quniform("subsample", 0.5,0.9,0.1), "colsample_bytree": hp.quniform("colsample_bytree", 0.4,0.9,0.1)}
    
    All the labels must be the same as names of positional arguments to the XGBoost's RandomForestRegressor class.
    """
    train_validation_split = Param(Params._dummy(), "train_validation_split",
                                   "A dictionary containing year and month on which to split the dataframe into train and validate set", None)
    """
    Example train_validation_split:
    train_validation_split = {"year": 2015, "month": 0}
    """

    @keyword_only
    def __init__(
            self,
            featuresCol=None,
            labelCol=None,
            hyperParamsSpace=None,
            maxIter=20,
            predictionCol="xgb_prediction",
            train_validation_split=None
    ):
        kwargs = self._input_kwargs
        
        super(XGBoostEstimator, self).__init__()
        self._setDefault(featuresCol=None, labelCol=None, hyperParamsSpace=None,
                         maxIter=20, predictionCol="prediction", train_validation_split=None)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
            self,
            featuresCol=None,
            labelCol=None,
            hyperParamsSpace=None,
            maxIter=None,
            predictionCol=None,
            train_validation_split=None
    ):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getHyperParams(self):
        return self.getOrDefault(self.hyperParamsSpace)

    def getTrainValidateSplit(self):
        return self.getOrDefault(self.train_validation_split)

    def getTrainedModel(self, params, df: DataFrame):
        featuresCol = self.getFeaturesCol()
        labelCol = self.getLabelCol()
        predictionCol = self.getPredictionCol()

        model = xgboost.XGBRegressor(**params)\
                       .fit(X=list(df[featuresCol]), y=list(df[labelCol]))

        return XGBoostModel(featuresCol=featuresCol, predictionCol=predictionCol, model=model)

    def trainEvaluate(self, trainDF: DataFrame, valDF: DataFrame, hyperParam):
        labelCol = self.getLabelCol()
        predictionCol = self.getPredictionCol()

        trainedModel = self.getTrainedModel(hyperParam, trainDF)
        eval = evaluators.MAPE(labelCol=labelCol, predictionCol=predictionCol)
        
        preds = trainedModel.transform(valDF)
        mape = eval.evaluate(preds)

        return {'loss': mape,
                'status': STATUS_OK,
                'params': hyperParam}

    def _fit(self, df: DataFrame):
        labelCol = self.getLabelCol()
        featuresCol = self.getFeaturesCol()
        hyperParamsSpace = self.getHyperParams()
        maxIter = self.getMaxIter()

        @udf(returnType=ArrayType(DoubleType()))
        def getDeVectorizedColumn(denseVector):
            return denseVector.values.tolist()

        if(df.select(featuresCol).dtypes[0][1] == "vector"):
            df = df.withColumn(featuresCol, getDeVectorizedColumn(featuresCol))

        trainDF, valDF = self.splitOnYM(df, self.getTrainValidateSplit())

        trainDF = trainDF.select([featuresCol, labelCol]).toPandas()
        valDF = valDF.select([featuresCol, labelCol]).cache()

        print("Tunning XGB hyper-parameters")
        trials = Trials()
        bestParams = fmin(partial(self.trainEvaluate, trainDF, valDF),
                          space=hyperParamsSpace, algo=tpe.suggest, max_evals=maxIter, trials=trials)

        print("bestParams: ", bestParams)

        valDF.unpersist()

        return self.getTrainedModel(trials.best_trial["result"]["params"], df.select([featuresCol, labelCol]).toPandas())
