from pyspark import keyword_only
from pyspark.ml.base import Estimator
from pyspark.sql import DataFrame
from pyspark.ml.param.shared import HasValidationIndicatorCol
from pyspark.ml.param import Params, Param, TypeConverters
import utils
import concurrent.futures
from models import MasterModel




class MasterEstimator(Estimator, HasValidationIndicatorCol):
    """
    A class that is used to fit/train all the required estimators and returns a collection of best models for each individual estimator type. \n
    This will fit all the given estimators and return a global model (instance of MasterModel) containing list of all the trained models.
    """

    estimatorsToFit = Param(Params._dummy(), "estimatorsToFit", "It contains a list of dictionary objects for the estimators to be trained", TypeConverters.toList)
    """
    This list must contain dictionary of estimators along with some information.
    e.g. 
    estimatorsToFit= [{"model_name": RandomForest, "estimator": ObjectOfEstimator}]
    The originally passed list will be updated with the best tunned model for individual estimator types.\n
    The fit method will remove the key "estimator" from all dictionaries in the list and add a new key "model" which contains the fitted data.
    """

    @keyword_only
    def __init__(self, validationIndicatorCol=None, estimatorsToFit=None):
        
        kwargs = self._input_kwargs

        super(MasterEstimator, self).__init__()

        self._setDefault(validationIndicatorCol=None, estimatorsToFit=None)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, validationIndicatorCol=None, estimatorsToFit=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def trainAndReturn(self, data: DataFrame, estimatorInfo: dict):
        estimatorInfo["model"] = estimatorInfo["estimator"].fit(data)
        return 

    def getEstimatorsToFit(self):
        return self.getOrDefault(self.estimatorsToFit)

    def _fit(self, trainDF:DataFrame):

        estimatorsToFit = self.getEstimatorsToFit()
        trainDF.cache()

        futuresList = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for est in estimatorsToFit:
                futuresList.append(executor.submit(est["estimator"].fit, trainDF))

        for i in range(len(futuresList)):
            estimatorsToFit[i]["model"] = futuresList[i].result()
            # To remove the estimator key data to free up some space, as it is not needed now
            del(estimatorsToFit[i]["estimator"])

        trainDF.unpersist()

        return MasterModel(bestModels=estimatorsToFit)

    
    def fit(self, df:DataFrame):
        df.cache()
        trainDF, testDF = utils.splitOnYM(df, {"year": 2015, "month": 5})
        return self._fit(trainDF)