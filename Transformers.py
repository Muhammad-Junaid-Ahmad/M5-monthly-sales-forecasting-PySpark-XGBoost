"""
Custom Transformers
"""

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols, Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
from pyspark.sql.window import Window


"""
Transformer to Filter out a Given Department from the DataFrame
"""


class FilterDF(Transformer):
    filterCond = Param(Params._dummy(), "filterCond",
                       "Condition for filtering the dataframe", TypeConverters.toString)

    @keyword_only
    def __init__(self, filterCond=None):
        super(FilterDF, self).__init__()
        self._setDefault(filterCond=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, filterCond=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getFilterCond(self):
        return self.getOrDefault(self.filterCond)

    def _transform(self, df: DataFrame):
        filterCond = self.getFilterCond()
        return df.filter(filterCond)


"""
Transformer to GroupBy a DataFrame on given inputCols.
"""


class GroupByTransformer(Transformer):

    groupByCols = Param(Params._dummy(), "groupByCols",
                        "list of Column names used for Grouping", TypeConverters.toListString)
    aggExprs = Param(Params._dummy(), "aggExprs",
                     "Aggregation expressions for the columns", None)

    @keyword_only
    def __init__(self, groupByCols=None, aggExprs=None):
        super(GroupByTransformer, self).__init__()
        self._setDefault(groupByCols=None, aggExprs=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, groupByCols=None, aggExprs=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getGroupByCols(self):
        return self.getOrDefault(self.groupByCols)

    def getAggExprs(self):
        return self.getOrDefault(self.aggExprs)

    def _transform(self, df: DataFrame):
        groupByCols = self.getGroupByCols()
        aggExprs = dict(self.getAggExprs())

        df = df.groupBy(groupByCols).agg(aggExprs)

        for aggC, aggF in aggExprs.items():
            df = df.withColumnRenamed("{}({})".format(aggF, aggC), aggC)

        return df

"""
Transofrmer for computing Lag features based on the inputCol and lag value.
"""


class LagFeatures(Transformer, HasInputCol, HasOutputCols):

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
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, inputCol=None,  partCols=None, orderCols=None, lagVals=None, outputCols=None):
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


"""
Tranformer for Exploding the DataFrame for 'd_'
"""


class ExplodingDays(Transformer, HasInputCols):

    @keyword_only
    def __init__(self, inputCols=None):
        super(ExplodingDays, self).__init__()

        self._setDefault(inputCols=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, inputCols=None):
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


"""
Tranformer for Filling NaN/null values in the DataFrame
"""


class FillNaN(Transformer):

    @keyword_only
    def __init__(self, inputCols=None):
        super(FillNaN, self).__init__()

    def _transform(self, df: DataFrame):
        return df.fillna(0)


class LogTransformer(Transformer, HasInputCols, HasOutputCols):
    @keyword_only
    def __init__(self, inputCols=None, outputCols=None):
        super(LogTransformer, self).__init__()
        self._setDefault(inputCols=None, outputCols=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, inputCols=None, outputCols=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, df: DataFrame):
        inputCols = self.getInputCols()
        outputCols = self.getOutputCols()

        for c in inputCols:
          df = df.withColumn(c, F.log(F.col(c)))

        return df


class AntiLogTransformer(Transformer, HasInputCols, HasOutputCols):
    @keyword_only
    def __init__(self, inputCols=None, outputCols=None):
        super(AntiLogTransformer, self).__init__()
        self._setDefault(inputCols=None, outputCols=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, inputCols=None, outputCols=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, df: DataFrame):
        inputCols = self.getInputCols()
        outputCols = self.getOutputCols()

        for c in inputCols:
          df = df.withColumn(c, F.exp(F.col(c)))

        return df

