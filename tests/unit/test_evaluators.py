import pytest
import Evaluators

class TestMAPETransformer():

    def get_expected(self, data):
        return 100 * sum( [  abs( (d[0] - d[1])/d[0] ) for d in data] ) / len(data)

    def pretty_close(self, actual, expected, epsilon = 1e-5):
        return abs(actual - expected) < epsilon

    @pytest.mark.usefixtures("spark")
    def test_1(self, spark):
        data = [(1.0, 1.0), 
                (1.0, 1.5), 
                (1.5, 1.0),
                (1.0, 2.0) ]
        in_df = spark.createDataFrame( data, ["labels", "predictions"])


        expected = self.get_expected(data)

        actual = Evaluators.MAPE(labelCol="labels", predictionCol="predictions").evaluate(in_df)

        assert self.pretty_close(actual, expected)

    @pytest.mark.usefixtures("spark")
    def test_2(self, spark):
        data = [(-1.0, 1.0), 
                (1.0, -1.5), 
                (-1.5, 1.0),
                (1.0, -2.0),
                (100.0, 100.0239),
                (2000.0, 30000.0) ]
        in_df = spark.createDataFrame( data, ["labels", "predictions"])


        expected = self.get_expected(data)

        actual = Evaluators.MAPE(labelCol="labels", predictionCol="predictions").evaluate(in_df)

        assert self.pretty_close(actual, expected)