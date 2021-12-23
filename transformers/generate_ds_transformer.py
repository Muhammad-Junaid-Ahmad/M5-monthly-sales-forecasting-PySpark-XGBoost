from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


class GenerateDSTransformer(Transformer, HasInputCols, HasOutputCol):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(GenerateDSTransformer, self).__init__()
        self._setDefault(inputCols=None, outputCol=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)
    
    def _transform(self, df: DataFrame):
        inputCols = self.getInputCols()
        outputCol = self.getOutputCol()

        df = df.withColumn(outputCol, F.date_format(F.concat_ws("-", *inputCols),"yyyy-MM-01").cast("date") )

        return df
