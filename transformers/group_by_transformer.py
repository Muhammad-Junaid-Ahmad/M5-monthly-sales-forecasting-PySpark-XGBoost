from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.sql import DataFrame

class GroupByTransformer(Transformer):
    """
    Transformer to GroupBy a DataFrame on given groupByCols and aggregate 
    the values of remaining columns using expression given in aggExprs.
    """

    groupByCols = Param(Params._dummy(), "groupByCols",
                        "list of Column names used for Grouping", TypeConverters.toListString)
    aggExprs = Param(Params._dummy(), "aggExprs",
                     "Aggregation expressions for the columns", None)

    @keyword_only
    def __init__(self, groupByCols=None, aggExprs=None):
        super(GroupByTransformer, self).__init__()
        self._setDefault(groupByCols=None, aggExprs=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, groupByCols=None, aggExprs=None):
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
