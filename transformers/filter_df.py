from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.sql import DataFrame



class FilterDF(Transformer):
    """
    Transformer to Filter out row of the DataFrame based on the filterCond
    """

    filterCond = Param(Params._dummy(), "filterCond",
                       "Condition for filtering the dataframe", TypeConverters.toString)

    @keyword_only
    def __init__(self, filterCond=None):
        super(FilterDF, self).__init__()
        self._setDefault(filterCond=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, filterCond=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getFilterCond(self):
        return self.getOrDefault(self.filterCond)

    def _transform(self, df: DataFrame):
        filterCond = self.getFilterCond()
        return df.filter(filterCond)
