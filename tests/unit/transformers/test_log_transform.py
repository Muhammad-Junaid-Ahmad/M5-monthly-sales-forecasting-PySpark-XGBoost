import pytest
from pyspark_test import assert_pyspark_df_equal
from transformers import LogTransformer
import math

class TestLogTransformer():
    @pytest.mark.usefixtures("spark")
    def testLogOfColumns(self, spark):
        in_df = spark.createDataFrame([("id_1", 0.0),
                                       ("id_2", 1.0),
                                       ("id_3", math.exp(1)),
                                       ("id_4", math.exp(10)),
                                       ("id_5", math.exp(-10))],
                                      ["id", "value"])

        expected_df = spark.createDataFrame([("id_1", None),
                                            ("id_2", 0.0),
                                            ("id_3", 1.0),
                                            ("id_4", 10.0),
                                            ("id_5", -10.0)],
                                            ["id", "value"])

        actual_df = LogTransformer(
            inputCols=["value"]).transform(in_df)

        # print(expected_df.show())
        # print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df)
