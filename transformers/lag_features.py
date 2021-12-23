from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCols, Param, Params, TypeConverters
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window


class LagFeatures(Transformer, HasInputCol, HasOutputCols):
    """
    Transofrmer for computing Lag features for inputCol.
    Number of lag features to compute is given in lagVals. 
    """

    partCols = Param(Params._dummy(), "partCols",
                     "Partitioning column name for Windowing", TypeConverters.toListString)
    orderCols = Param(Params._dummy(), "orderCols",
                      "Ordering column name for Windowing", TypeConverters.toListString)
    lagVals = Param(Params._dummy(), "lagVals",
                    "Value for the lag window", TypeConverters.toListInt)

    @keyword_only
    def __init__(self, inputCol=None, partCols=None, orderCols=None, lagVals=None):
        super(LagFeatures, self).__init__()

        self._setDefault(inputCol=None,  partCols=None,
                         orderCols=None, lagVals=None, outputCols=None)
        kwargs = self._input_kwargs
        kwargs["outputCols"] = ["lag_" + inputCol +
                                "_" + str(lagVal) for lagVal in lagVals]
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None,  partCols=None, orderCols=None, lagVals=None, outputCols=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    # Getters for Input Params

    def getPartCols(self):
        return self.getOrDefault(self.partCols)

    def getOrderCols(self):
        return self.getOrDefault(self.orderCols)

    def getLagVals(self):
        return self.getOrDefault(self.lagVals)

    def _transform(self, df: DataFrame):
        inputCol = self.getInputCol()
        outputCols = self.getOutputCols()
        partCols = self.getPartCols()
        orderCols = self.getOrderCols()
        lagVals = self.getLagVals()

        windowSpec = Window.partitionBy(partCols).orderBy(orderCols)
        for lagVal, outputCol in zip(lagVals, outputCols):
            df = df.withColumn(outputCol, F.lag(
                inputCol, lagVal).over(windowSpec))
        return df
