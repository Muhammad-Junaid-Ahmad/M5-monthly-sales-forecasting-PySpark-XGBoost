import pytest
from pyspark_test import assert_pyspark_df_equal
from transformers import GroupByTransformer

class TestGroupByTransformer():

    @pytest.mark.usefixtures("spark")
    def testGroupBySum(self, spark):
        in_df = spark.createDataFrame(
            [("a", 1), ("a", 2), ("c", 4), ("c", 5)], ["alpha", "num"])
        expected_df = spark.createDataFrame(
            [("c", 9), ("a", 3)], ["alpha", "num"])

        actual_df = GroupByTransformer(groupByCols=["alpha"], aggExprs={
                                                    "num": "sum"}).transform(in_df)

        assert_pyspark_df_equal(actual_df, expected_df, order_by=["alpha"])

    @pytest.mark.usefixtures("spark")
    def testGroupbyAvg(self, spark):
        in_df = spark.createDataFrame(
            [("a", 1), ("a", 2), ("c", 4), ("c", 5)], ["alpha", "num"])
        expected_df = spark.createDataFrame(
            [("c", 4.5), ("a", 1.5)], ["alpha", "num"])

        actual_df = GroupByTransformer(groupByCols=["alpha"], aggExprs={
                                                    "num": "avg"}).transform(in_df)

        assert_pyspark_df_equal(actual_df, expected_df, order_by=["alpha"])

    @pytest.mark.usefixtures("spark")
    def testGroupByMax(self, spark):
        in_df = spark.createDataFrame(
            [("a", 1), ("a", 2), ("c", 4), ("c", 5)], ["alpha", "num"])
        expected_df = spark.createDataFrame(
            [("c", 5), ("a", 2)], ["alpha", "num"])

        actual_df = GroupByTransformer(groupByCols=["alpha"], aggExprs={
                                                    "num": "max"}).transform(in_df)

        assert_pyspark_df_equal(actual_df, expected_df, order_by=["alpha"])
