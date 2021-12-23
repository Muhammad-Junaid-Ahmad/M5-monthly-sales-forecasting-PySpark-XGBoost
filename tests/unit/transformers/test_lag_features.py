import pytest
from pyspark_test import assert_pyspark_df_equal
from transformers import LagFeatures

class TestLagFeaturesTransformer():
    @pytest.mark.usefixtures("spark")
    def testGenerationOfLagFeatures(self, spark):
        in_df = spark.createDataFrame([("id_1", "d_1", 1),
                                       ("id_1", "d_2", 2),
                                       ("id_1", "d_3", 3),
                                       ("id_1", "d_4", 4),
                                       ("id_1", "d_5", 5),
                                       ("id_2", "d_1", 1),
                                       ("id_2", "d_2", 2),
                                       ("id_2", "d_3", 3),
                                       ("id_2", "d_4", 4),
                                       ("id_2", "d_5", 5)],
                                      ["id", "day", "sales"])

        expected_df = spark.createDataFrame([("id_1", "d_1", 1, None, None),
                                             ("id_1", "d_2", 2, 1, None),
                                             ("id_1", "d_3", 3, 2, 1),
                                             ("id_1", "d_4", 4, 3, 2),
                                             ("id_1", "d_5", 5, 4, 3),
                                             ("id_2", "d_1", 1, None, None),
                                             ("id_2", "d_2", 2, 1, None),
                                             ("id_2", "d_3", 3, 2, 1),
                                             ("id_2", "d_4", 4, 3, 2),
                                             ("id_2", "d_5", 5, 4, 3)],
                                            ["id", "day", "sales", "lag_sales_1", "lag_sales_2"])

        actual_df = LagFeatures(inputCol="sales", partCols=["id"], orderCols=[
                                             "day"], lagVals=[1, 2]).transform(in_df)

        # print(expected_df.show())
        # print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df, order_by=["id", "day"])
