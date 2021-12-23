from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params
from pyspark.sql import DataFrame
import pyspark


class DFSplitter(Transformer):
    """
    Tranformer for Splitting the DataFrame based on year and month
    """

    splittingYM = Param(Params._dummy(), "splittingYM",
                       "A dictionary containing year and month on which Dataframe needs to be splitted", None)

    @keyword_only
    def __init__(self, splittingYM=None):
        super(DFSplitter, self).__init__()

        self._setDefault(splittingYM=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, splittingYM=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getSplittingYM(self):
        return self.getOrDefault(self.splittingYM)

    @staticmethod
    def splitOnYM(df: DataFrame, ym):
        year = ym["year"]
        month = ym["month"]

        # if type of the DataFrame is not pyspark then it must be pandas. So, pandas uses a function query to filter out rows
        if(type(df) == pyspark.sql.dataframe.DataFrame):
            df1 = df.filter("year<{} or year={} and month<={}".format(year, year, month))
            df2 = df.filter("year>{} or year={} and month>{}".format(year, year, month))
        else:
            df1 = df.query("year<{} or year=={} and month<={}".format(year, year, month))
            df2 = df.query("year>{} or year=={} and month>{}".format(year, year, month))
            
        return df1, df2

    def _transform(self, df: DataFrame):
        splittingYM = self.getSplittingYM()
        return self.splitOnYM(df, splittingYM)
