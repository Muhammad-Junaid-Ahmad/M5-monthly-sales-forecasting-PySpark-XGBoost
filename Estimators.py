from pyspark import keyword_only
from pyspark.ml import Estimator
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasMaxIter, HasPredictionCol, Param, Params
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.sql.functions import udf, pandas_udf
from pyspark.ml.regression import RandomForestRegressor
import Evaluators
from hyperopt import fmin, Trials, tpe, STATUS_OK
from functools import partial
import Models
import xgboost
import utils



"""
Random Forest Regressor Estimator
"""
class RandomForestEstimator(Estimator, HasLabelCol, HasFeaturesCol, HasPredictionCol, HasMaxIter):
    hyperParamsSpace = Param(Params._dummy(), "hyperParamsSpace", "HyperOpt object for tunning hyperparameters for the Random Forest Regressor", None)
    """
    Example hyperParamsSpace:
    hyperParamsSpace = {"maxDepth": hp.quniform("maxDepth", 10,20,1), "maxBins": hp.quniform("maxBins", 40,60,1),
                        "numTrees": hp.quniform("numTrees", 18,25,1), "minInfoGain": hp.quniform("minInfoGain", 0,0.5,0.1)}
    
    All the labels must be the same as names of positional arguments to the PySpark's RandomForestRegressor class.
    """
    train_validation_split = Param(Params._dummy(), "train_validation_split", "A dictionary containing year and month on which to split the dataframe into train and validate set", None)
    """
    Example train_validation_split:
    train_validation_split = {"year": 2015, "month": 0}
    """

    @keyword_only
    def __init__(self, featuresCol=None, labelCol=None, hyperParamsSpace=None, maxIter = 20, predictionCol = "prediction", train_validation_split=None):
        super(RandomForestEstimator, self).__init__()
        self._setDefault(featuresCol=None, labelCol=None, hyperParamsSpace=None, maxIter = 20, predictionCol = "prediction", train_validation_split=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, featuresCol=None, labelCol=None, hyperParamsSpace=None, maxIter = None, predictionCol = None, train_validation_split=None):
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

        est = RandomForestRegressor(featuresCol=featuresCol, labelCol=labelCol, **hyperParam)
        eval = Evaluators.MAPE(labelCol=labelCol, predictionCol=predictionCol)

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

        print("Tunning hyper-parameters")
        trials = Trials()
        bestParams = fmin(partial(self.train_evaluate, trainDF, valDF), 
                    space=hyperParamsSpace, algo=tpe.suggest, max_evals=maxIter, trials=trials)

        print("bestParams: ", bestParams)

        return RandomForestRegressor(featuresCol=featuresCol, labelCol=labelCol, **bestParams).fit(trainDF)

    def fit(self, df: DataFrame):
        trainDF, valDF = utils.df_split(df, **self.getTrainValidateSplit())
        return self._fit(trainDF=trainDF, valDF=valDF)



"""
XGBoost Estimator
"""
class XGBoostEstimator(Estimator, HasLabelCol, HasFeaturesCol, HasPredictionCol, HasMaxIter):
    hyperParamsSpace = Param(Params._dummy(), "hyperParamsSpace", "HyperOpt object for tunning hyperparameters for the Random Forest Regressor", None)
    """
    Example hyperParamsSpace:
    hyperParamsSpace = {"n_estimators": scope.int(hp.quniform("n_estimators", 15,25,1)), "max_depth": scope.int(hp.quniform("max_depth", 50,70,1)),
                        "subsample": hp.quniform("subsample", 0.5,0.9,0.1), "colsample_bytree": hp.quniform("colsample_bytree", 0.4,0.9,0.1)}
    
    All the labels must be the same as names of positional arguments to the PySpark's RandomForestRegressor class.
    """
    train_validation_split = Param(Params._dummy(), "train_validation_split", "A dictionary containing year and month on which to split the dataframe into train and validate set", None)
    """
    Example train_validation_split:
    train_validation_split = {"year": 2015, "month": 0}
    """

    @keyword_only
    def __init__(self, featuresCol=None, labelCol=None, hyperParamsSpace=None,maxIter = 20, predictionCol = "prediction", train_validation_split=None):
        super(XGBoostEstimator, self).__init__()
        self._setDefault(featuresCol=None, labelCol=None, hyperParamsSpace=None, maxIter = 20, predictionCol = "prediction", train_validation_split=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, featuresCol=None, labelCol=None, hyperParamsSpace=None, maxIter = None, predictionCol = None, train_validation_split=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getHyperParams(self):
        return self.getOrDefault(self.hyperParamsSpace)
    
    def getTrainValidateSplit(self):
        return self.getOrDefault(self.train_validation_split)
    
    def getTrainedModel(self, params, df:DataFrame):
        featuresCol = self.getFeaturesCol()
        labelCol = self.getLabelCol()
        predictionCol = self.getPredictionCol()

        model = xgboost.XGBRegressor(**params).fit(X=list(df[featuresCol]), y=list(df[labelCol]))

        return Models.XGBoostModel(featuresCol=featuresCol, predictionCol=predictionCol, model=model)

    def train_evaluate(self, trainDF: DataFrame, valDF: DataFrame, hyperParam):
        labelCol = self.getLabelCol()
        predictionCol = self.getPredictionCol()

        trainedModel = self.getTrainedModel(hyperParam, trainDF)
        eval = Evaluators.MAPE(labelCol=labelCol, predictionCol=predictionCol)

        mape = eval.evaluate(trainedModel.transform(valDF))

        return {'loss': mape, 
                'status': STATUS_OK, 
                'params': hyperParam}

    def _fit(self, trainDF: DataFrame, valDF: DataFrame):
        labelCol = self.getLabelCol()
        featuresCol = self.getFeaturesCol()
        hyperParamsSpace = self.getHyperParams()
        maxIter = self.getMaxIter()

        trainDF = trainDF.select([featuresCol, labelCol])
        valDF = valDF.select([featuresCol, labelCol])
        
        @udf(returnType=ArrayType(DoubleType()))  
        def getDeVectorizedColumn(denseVector):
            return denseVector.values.tolist()  
        
        if(valDF.select(featuresCol).dtypes[0][1] == "vector"):
            valDF = valDF.withColumn(featuresCol, getDeVectorizedColumn(featuresCol))
        if(trainDF.select(featuresCol).dtypes[0][1] == "vector"):
            trainDF = trainDF.withColumn(featuresCol, getDeVectorizedColumn(featuresCol))

        trainDF = trainDF.select([featuresCol, labelCol]).toPandas()
        valDF = valDF.select([featuresCol, labelCol]).cache()
            
        print("Tunning hyper-parameters")
        trials = Trials()
        bestParams = fmin(partial(self.train_evaluate, trainDF, valDF), 
                    space=hyperParamsSpace, algo=tpe.suggest, max_evals=maxIter, trials=trials)
        
        print("bestParams: ", bestParams)
        valDF.unpersist()

        return self.getTrainedModel(trials.best_trial["result"]["params"], trainDF)
        

    def fit(self, df: DataFrame):
        trainDF, valDF = utils.df_split(df, **self.getTrainValidateSplit())
        return self._fit(trainDF=trainDF, valDF=valDF)
