from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.sql import DataFrame


class FillNaN(Transformer):
    """
    Tranformer for Filling NaN/null values in the DataFrame
    for colums given in inputCols
    """
    
    @keyword_only
    def __init__(self, inputCols=None):
        super(FillNaN, self).__init__()

    def _transform(self, df: DataFrame):
        return df.fillna(0)