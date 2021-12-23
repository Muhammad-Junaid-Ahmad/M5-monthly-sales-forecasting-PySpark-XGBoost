from pyspark import keyword_only
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark



class MAPE(Evaluator, HasLabelCol, HasPredictionCol):
    @keyword_only
    def __init__(self, labelCol=None, predictionCol=None):
        super(MAPE, self).__init__()
        self._setDefault(labelCol=None, predictionCol=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, labelCol=None, predictionCol=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _evaluate(self, df: DataFrame):
        labelCol = self.getLabelCol()
        predictionCol = self.getPredictionCol()

        # For Pysaprk DataFrame
        if(type(df) == pyspark.sql.dataframe.DataFrame):
            mape = df.select(F.abs( (F.col(labelCol) - F.col(predictionCol)) / F.col(labelCol) ) ).groupBy().avg().collect()[0][0] * 100
        else:
            mape = (abs((df[labelCol] - df[predictionCol]) / df[labelCol])).mean() * 100

        return mape