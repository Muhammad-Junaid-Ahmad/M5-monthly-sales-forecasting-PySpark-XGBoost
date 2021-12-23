import pytest
from evaluators.mape import MAPE

class TestMAPETransformer():

    def getExpected(self, data):
        return 100 * sum( [  abs( (d[0] - d[1])/d[0] ) for d in data] ) / len(data)

    def prettyClose(self, actual, expected, epsilon = 1e-5):
        return abs(actual - expected) < epsilon

    @pytest.mark.usefixtures("spark")
    def testMAPECloseness(self, spark):
        data = [(1.0, 1.0), 
                (1.0, 1.5), 
                (1.5, 1.0),
                (1.0, 2.0),
                (-1.0, 1.0), 
                (1.0, -1.5), 
                (-1.5, 1.0),
                (1.0, -2.0),
                (100.0, 100.0239),
                (2000.0, 30000.0) ]

        in_df = spark.createDataFrame( data, ["labels", "predictions"])


        expected = self.getExpected(data)

        actual = MAPE(labelCol="labels", predictionCol="predictions").evaluate(in_df)

        assert self.prettyClose(actual, expected)

    @pytest.mark.usefixtures("spark")
    def testInvalidColumns(self, spark):
        data = [(-1.0, 1.0), 
                (1.0, -1.5), 
                (-1.5, 1.0),
                (1.0, -2.0),
                (100.0, 100.0239),
                (2000.0, 30000.0) ]
        in_df = spark.createDataFrame( data, ["label", "predictions"])
    
        with pytest.raises(Exception):
            actual = MAPE(labelCol="labels", predictionCol="predictions").evaluate(in_df)

    