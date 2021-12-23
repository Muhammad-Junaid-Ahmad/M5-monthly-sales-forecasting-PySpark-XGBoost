from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from pyspark.ml.param import Param, Params

class TrainTestValIndicator(Transformer, HasInputCols, HasOutputCol):

    indicatorCond = Param(Params._dummy(), "indicatorCond", "A dictionary conatining condtion values for train/test/val splitting", None)
    """
    Example indicatorCond:
    indicatorCond = {"train_before": {"year": 2015, "month": 0}, 
                     "test_after": {"year": 2015, "month": 5} }
    """

    @keyword_only
    def __init__(self, inputCols=None, outputCol=None, indicatorCond=None):
        super(TrainTestValIndicator, self).__init__()
        self._setDefault(inputCols=None, outputCol=None, indicatorCond=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None, indicatorCond=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getIndicatorCond(self):
        return self.getOrDefault(self.indicatorCond)

    def _transform(self, df: DataFrame):
        inputCols = self.getInputCols()
        outputCol = self.getOutputCol()
        indicatorCond = self.getIndicatorCond()
        
        trainInd = (df["year"] < indicatorCond["train_before"]["year"])\
                   | ( (df["year"] == indicatorCond["train_before"]["year"]) & 
                       (df["month"] < indicatorCond["train_before"]["month"]) )

        testInd = (df["year"] > indicatorCond["test_after"]["year"])\
                   | ( (df["year"] == indicatorCond["test_after"]["year"]) & 
                       (df["month"] > indicatorCond["test_after"]["month"]) )

        df = df.withColumn(outputCol,  F.when(trainInd, 0)\
                                        .when(testInd, 2)\
                                        .otherwise(1))

        return df