from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

class LogTransformer(Transformer, HasInputCols, HasOutputCols):
    @keyword_only
    def __init__(self, inputCols=None, outputCols=None):
        super(LogTransformer, self).__init__()
        self._setDefault(inputCols=None, outputCols=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, df: DataFrame):
        inputCols = self.getInputCols()
        outputCols = self.getOutputCols()

        for c in inputCols:
          df = df.withColumn(c, F.log(F.col(c)))

        return df