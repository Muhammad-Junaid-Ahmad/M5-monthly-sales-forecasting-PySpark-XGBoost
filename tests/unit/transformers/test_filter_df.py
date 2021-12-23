import pytest
from pyspark_test import assert_pyspark_df_equal
from transformers import FilterDF

class TestFilterDFTransformer():

    @pytest.mark.usefixtures("spark")
    def testFiltering(self, spark):
        in_df = spark.createDataFrame(
            [("Alice", 1), ("Bob", 2)], ["name", "num"])
        expected_df = spark.createDataFrame([("Alice", 1)], ["name", "num"])

        actual_df = FilterDF(
            filterCond="num < 2").transform(in_df)

        assert_pyspark_df_equal(actual_df, expected_df)

    @pytest.mark.usefixtures("spark")
    def testInvalidFilteringCond(self, spark):
        in_df = spark.createDataFrame(
            [("Alice", 1), ("Bob", 2)], ["name", "num"])
        
        with pytest.raises(Exception):
            actual_df = FilterDF(
                filterCond="nums >= 2").transform(in_df)