import pytest
from pyspark_test import assert_pyspark_df_equal
from transformers import ExplodeDays

class TestExplodeDaysTransformer():

    @pytest.mark.usefixtures("spark")
    def testMeltingOfCols(self, spark):
        in_df = spark.createDataFrame([("id_1", 1, 2, 3, 4, 5),
                                       ("id_2", 1, 2, 3, 4, 5)],
                                      ["id", "d_1", "d_2", "d_3", "d_4", "d_5"])

        expected_df = spark.createDataFrame([("id_1", "d_1", 1),
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

        actual_df = ExplodeDays(inputCols=["id"]).transform(in_df)

        # print(expected_df.show())
        # print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df)

    @pytest.mark.usefixtures("spark")
    def testMeltingWithEmptyColName(self, spark):
        in_df = spark.createDataFrame([("id_1", 1, 2, 3, 4, 5),
                                       ("id_2", 1, 2, 3, 4, 5)],
                                      ["id", "d_1", "d_2", "d_3", "d_4", "d_5"])

        with pytest.raises(Exception):
            actual_df = ExplodeDays(inputCols=[""]).transform(in_df)
