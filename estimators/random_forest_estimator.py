from pyspark import keyword_only
from pyspark.ml import Estimator
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasMaxIter, HasPredictionCol, Param, Params
from pyspark.sql import DataFrame
from pyspark.ml.regression import RandomForestRegressor
import evaluators
from hyperopt import fmin, Trials, tpe, STATUS_OK
from functools import partial
from transformers import DFSplitter


class RandomForestEstimator(Estimator, HasLabelCol, HasFeaturesCol, HasPredictionCol, HasMaxIter, DFSplitter):
    """
    Random Forest Regressor Estimator
    """
    
    hyperParamsSpace = Param(Params._dummy(), "hyperParamsSpace",
                             "HyperOpt object for tunning hyperparameters for the Random Forest Regressor", None)
    """
    Example hyperParamsSpace:
    hyperParamsSpace = {"maxDepth": hp.quniform("maxDepth", 10,20,1), "maxBins": hp.quniform("maxBins", 40,60,1),
                        "numTrees": hp.quniform("numTrees", 18,25,1), "minInfoGain": hp.quniform("minInfoGain", 0,0.5,0.1)}
    
    All the labels must be the same as names of positional arguments to the PySpark's RandomForestRegressor class.
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
            predictionCol="rf_prediction",
            train_validation_split=None
    ):
        kwargs = self._input_kwargs
        
        super(RandomForestEstimator, self).__init__()
        self._setDefault(featuresCol=None, labelCol=None, hyperParamsSpace=None,
                         maxIter=20, predictionCol="rf_prediction", train_validation_split=None)
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

    def train_evaluate(self, trainDF: DataFrame, valDF: DataFrame, hyperParam):
        featuresCol = self.getFeaturesCol()
        labelCol = self.getLabelCol()
        predictionCol = self.getPredictionCol()

        est = RandomForestRegressor(
            featuresCol=featuresCol, labelCol=labelCol, predictionCol=predictionCol, **hyperParam)
        eval = evaluators.MAPE(labelCol=labelCol, predictionCol=predictionCol)

        trainedModel = est.fit(trainDF)

        mape = eval.evaluate(trainedModel.transform(valDF))

        return {'loss': mape,
                'status': STATUS_OK,
                'params': hyperParam}

    def _fit(self, trainDF: DataFrame, valDF: DataFrame):
        hyperParamsSpace = self.getHyperParams()
        featuresCol = self.getFeaturesCol()
        labelCol = self.getLabelCol()
        maxIter = self.getMaxIter()
        predictionCol = self.getPredictionCol()

        trainDF = trainDF.select([featuresCol, labelCol]).cache()
        valDF = valDF.select([featuresCol, labelCol]).cache()

        print("Tunning RF hyper-parameters")
        trials = Trials()
        bestParams = fmin(partial(self.train_evaluate, trainDF, valDF),
                          space=hyperParamsSpace, algo=tpe.suggest, max_evals=maxIter, trials=trials)

        print("bestParams: ", bestParams)

        M = RandomForestRegressor(
            featuresCol=featuresCol, labelCol=labelCol, predictionCol=predictionCol, **bestParams).fit(trainDF)
        trainDF.unpersist()
        valDF.unpersist()

        return M

    def fit(self, df: DataFrame):
        trainDF, valDF = self.splitOnYM(df, self.getTrainValidateSplit())
        return self._fit(trainDF=trainDF, valDF=valDF)