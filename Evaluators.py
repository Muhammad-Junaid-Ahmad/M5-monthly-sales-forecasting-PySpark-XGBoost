from pyspark import keyword_only
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol
from pyspark.sql import DataFrame
import pyspark.sql.functions as F



class MAPE(Evaluator, HasLabelCol, HasPredictionCol):
    @keyword_only
    def __init__(self, labelCol=None, predictionCol=None):
        super(MAPE, self).__init__()
        self._setDefault(labelCol=None, predictionCol=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, labelCol=None, predictionCol=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def get_label_col(self):
        return self.getOrDefault(self.labelCol)
    
    def get_prediction_col(self):
        return self.getOrDefault(self.predictionCol)

    def _evaluate(self, df: DataFrame):
        labelCol = self.get_label_col()
        predictionCol = self.get_prediction_col()

        mape = df.select(F.abs( (F.col(labelCol) - F.col(predictionCol)) / F.col(labelCol) ) ).groupBy().avg().collect()[0][0] * 100
        return mape