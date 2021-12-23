from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


class ExplodeDays(Transformer, HasInputCols):
    """
    Tranformer for Exploding the DataFrame for 'd_'.
    All 'd_xx' columns will be stacked up under one 'day' column. 
    """

    @keyword_only
    def __init__(self, inputCols=None):
        super(ExplodeDays, self).__init__()

        self._setDefault(inputCols=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, df: DataFrame):
        inputCols = self.getInputCols()

        # Filter dtypes and split into column names and type description
        cols, dtypes = zip(*((c, t)
                           for (c, t) in df.dtypes if c not in inputCols))
        # Spark SQL supports only homogeneous columns
        assert len(set(dtypes)) == 1, "All columns have to be of the same type"

        # Create and explode an array of (column_name, column_value) structs
        melt = F.explode(F.array([
            F.struct(F.lit(c).alias("day"), F.col(c).alias("sales")) for c in cols
        ])).alias("melt")

        return df.select(inputCols + [melt]).select(inputCols + ["melt.day", "melt.sales"])