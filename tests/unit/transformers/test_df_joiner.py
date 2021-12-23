import pytest
from pyspark_test import assert_pyspark_df_equal
from transformers import DFsJoiner

class TestDFsJoinerTransformer():

    @pytest.mark.usefixtures("spark")
    def testDFsJoinOnDiffColNames(self, spark):
        df1 = spark.createDataFrame([("id_1", "d_1"),
                                    ("id_2", "d_5")],
                                ["id", "day"])

        df2 = spark.createDataFrame([("id_1", 1),
                                    ("id_2", 5)],
                                ["id1", "sales"])

        expected_df = spark.createDataFrame([("id_1", "d_1", "id_1",1),
                                              ("id_2", "d_5", "id_2",5)],
                                             ["id", "day", "id1" ,"sales"])

        actual_df = DFsJoiner(joinOn= (df1.id == df2.id1) , df2=df2).transform(df1)

        # print(expected_df.show())
        # print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df)

    @pytest.mark.usefixtures("spark")
    def testDFsJoinOnSameColNames(self, spark):
        df1 = spark.createDataFrame([("id_1", "d_1"),
                                    ("id_2", "d_5")],
                                ["id", "day"])

        df2 = spark.createDataFrame([("id_1", 1),
                                    ("id_2", 5)],
                                ["id", "sales"])

        expected_df = spark.createDataFrame([("id_1", "d_1", "id_1",1),
                                              ("id_2", "d_5", "id_2",5)],
                                             ["id", "day", "id" ,"sales"])

        actual_df = DFsJoiner(joinOn = (df1.id == df2.id) , df2=df2).transform(df1)

        # print(expected_df.show())
        # print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df)

    @pytest.mark.usefixtures("spark")
    def testDFsJoinUsingColNamesInString(self, spark):
        df1 = spark.createDataFrame([("id_1", "d_1"),
                                    ("id_2", "d_5")],
                                ["id", "day"])

        df2 = spark.createDataFrame([("id_1", 1),
                                    ("id_2", 5)],
                                ["id", "sales"])

        expected_df = spark.createDataFrame([("id_1", "d_1", 1),
                                              ("id_2", "d_5",5)],
                                             ["id", "day", "sales"])

        actual_df = DFsJoiner(joinOn = "id" , df2=df2).transform(df1)

        # print(expected_df.show())
        # print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df)

    @pytest.mark.usefixtures("spark")
    def testDFsJoinUsingColNamesInDict(self, spark):
        df1 = spark.createDataFrame([("id_1", "d_1"),
                                    ("id_2", "d_5")],
                                ["id", "day"])

        df2 = spark.createDataFrame([("id_1", 1),
                                    ("id_2", 5)],
                                ["id1", "sales"])

        expected_df = spark.createDataFrame([("id_1", "d_1", "id_1",1),
                                              ("id_2", "d_5", "id_2",5)],
                                             ["id", "day", "id1" ,"sales"])

        actual_df = DFsJoiner(joinOn = {"df1_col": "id", "df2_col": "id1"} , df2=df2).transform(df1)

        # print(expected_df.show())
        # print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df)

    @pytest.mark.usefixtures("spark")
    def testDFsJoinOnInvalidCols(self, spark):
        df1 = spark.createDataFrame([("id_1", "d_1"),
                                    ("id_2", "d_5")],
                                ["id", "day"])

        df2 = spark.createDataFrame([("id_1", 1),
                                    ("id_2", 5)],
                                ["id", "sales"])
        
        with pytest.raises(Exception):
            actual_df = DFsJoiner(joinOn = "id1" , df2=df2).transform(df1)

       