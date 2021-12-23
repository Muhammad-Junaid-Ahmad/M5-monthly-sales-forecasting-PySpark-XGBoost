import pyspark
import pytest
from pyspark_test import assert_pyspark_df_equal
from transformers import AntiLogTransformer
import math
import pyspark.sql.functions as F

class TestAntiLogTransformer():

    def pretty_close(self, actual, expected, epsilon=1e-5):
        return abs(actual - expected) < epsilon

    @pytest.mark.usefixtures("spark")
    def testAntiLogOfColumn(self, spark):
        in_df = spark.createDataFrame([("id_1", 0.0),
                                       ("id_2", 1.0),
                                       ("id_3", math.log(1)),
                                       ("id_4", math.log(10)),
                                       ("id_5", -10.0)],
                                      ["id", "value"])

        expected_df = spark.createDataFrame([("id_1", 1.0),
                                            ("id_2", round(math.exp(1), 4)),
                                            ("id_3", 1.0),
                                            ("id_4", 10.0),
                                            ("id_5", round(math.exp(-10), 4))],
                                            ["id", "value"])

        actual_df = AntiLogTransformer(inputCols=["value"]).transform(
            in_df).withColumn("value", F.round("value", 4))

        # print(expected_df.show())
        # print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df)

    @pytest.mark.usefixtures("spark")
    def testEmptyColumnName(self, spark):
        in_df = spark.createDataFrame([("id_1", 0.0),
                                       ("id_2", 1.0),
                                       ("id_3", math.log(1)),
                                       ("id_4", math.log(10)),
                                       ("id_5", -10.0)],
                                      ["id", "value"])

        with pytest.raises(Exception):
            actual_df = AntiLogTransformer(inputCols=[""]).transform(
                in_df).withColumn("value", F.round("value", 4))

        # print(expected_df.show())
        # print(actual_df.show())