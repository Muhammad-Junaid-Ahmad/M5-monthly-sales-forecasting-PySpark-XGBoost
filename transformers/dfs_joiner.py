from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.sql import DataFrame
import pyspark


class DFsJoiner(Transformer):
    """
    Tranformer for Joining two DataFrames based on given column and condition
    """

    joinOn = Param(Params._dummy(), "joinOn",
                       "joining column information", None)
    """
    Instance of column that gives boolean results. e.g. (df1.id == df2.id)\n
    OR 
    column name in string. e.g 'id'\n
    OR
    dictionary of the form {"df1_col": "id", "df1_col": "id"}.\n
    For more info on the correct 'joinOn' variable type, see the doc of pyspark.DataFrame.join().
    """

    df2 = Param(Params._dummy(), "df2", "Second DataFrame used in the join", None)

    @keyword_only
    def __init__(self, joinOn=None, df2=None):
        super(DFsJoiner, self).__init__()

        self._setDefault(joinOn=None, df2=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, joinOn=None, df2=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getJoinOn(self):
        return self.getOrDefault(self.joinOn)
    
    def getDF2(self):
        return self.getOrDefault(self.df2)

    def _transform(self, df: DataFrame):
        joinOn = self.getJoinOn()
        df2 = self.getDF2()

        if type(joinOn) == dict:
            joinOn = df[joinOn["df1_col"]] == df2[joinOn["df2_col"]]
        

        return df.join(df2, joinOn)
