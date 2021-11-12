import pytest
from pyspark_test import assert_pyspark_df_equal
import Transformers
import math
import pyspark.sql.functions as F


class TestFilterDFTransformer():

    @pytest.mark.usefixtures("spark")
    def test_1(self, spark):
        in_df = spark.createDataFrame([("Alice", 1), ("Bob", 2)], ["name", "num"])
        expected_df = spark.createDataFrame([("Alice",1)], ["name", "num"])

        actual_df = Transformers.FilterDF(filterCond="num < 2").transform(in_df)
        
        assert_pyspark_df_equal(actual_df, expected_df)

    @pytest.mark.usefixtures("spark")
    def test_2(self, spark):
        in_df = spark.createDataFrame([("Alice", 1), ("Bob", 2)], ["name", "num"])
        expected_df = spark.createDataFrame([("Bob",2)], ["name", "num"])

        actual_df = Transformers.FilterDF(filterCond="num >= 2").transform(in_df)
        
        assert_pyspark_df_equal(actual_df, expected_df)
        

class TestGroupByTransformer():

    @pytest.mark.usefixtures("spark")
    def test_1(self, spark):
        in_df = spark.createDataFrame([("a", 1), ("a", 2), ("c", 4), ("c", 5)], ["alpha", "num"])
        expected_df = spark.createDataFrame([("c",9), ("a", 3)], ["alpha", "num"])

        actual_df = Transformers.GroupByTransformer(groupByCols=["alpha"], aggExprs={"num": "sum"}).transform(in_df)
        
        assert_pyspark_df_equal(actual_df, expected_df)

    @pytest.mark.usefixtures("spark")
    def test_2(self, spark):
        in_df = spark.createDataFrame([("a", 1), ("a", 2), ("c", 4), ("c", 5)], ["alpha", "num"])
        expected_df = spark.createDataFrame([("c",4.5), ("a", 1.5)], ["alpha", "num"])

        actual_df = Transformers.GroupByTransformer(groupByCols=["alpha"], aggExprs={"num": "avg"}).transform(in_df)
        
        assert_pyspark_df_equal(actual_df, expected_df)

    @pytest.mark.usefixtures("spark")
    def test_3(self, spark):
        in_df = spark.createDataFrame([("a", 1), ("a", 2), ("c", 4), ("c", 5)], ["alpha", "num"])
        expected_df = spark.createDataFrame([("c",5), ("a", 2)], ["alpha", "num"])

        actual_df = Transformers.GroupByTransformer(groupByCols=["alpha"], aggExprs={"num": "max"}).transform(in_df)
        
        assert_pyspark_df_equal(actual_df, expected_df)


class TestExplodingDaysTransformer():

    @pytest.mark.usefixtures("spark")
    def test_1(self, spark):
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

        actual_df = Transformers.ExplodingDays(inputCols=["id"]).transform(in_df)
        
        #print(expected_df.show())
        #print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df)


class TestLagFeaturesTransformer():
    @pytest.mark.usefixtures("spark")
    def test_1(self, spark):
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

        actual_df = Transformers.LagFeatures(inputCol="sales", partCols=["id"], orderCols=["day"], lagVals=[1,2]).transform(in_df)
        
        #print(expected_df.show())
        #print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df, order_by=["id", "day"])

class TestLogTransformer():
    @pytest.mark.usefixtures("spark")
    def test_1(self, spark):
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

        actual_df = Transformers.LogTransformer(inputCols=["value"]).transform(in_df)
        
        #print(expected_df.show())
        #print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df)


class TestAntiLogTransformer():
    
    def pretty_close(self, actual, expected, epsilon = 1e-5):
        return abs(actual - expected) < epsilon

    @pytest.mark.usefixtures("spark")
    def test_1(self, spark):
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

        actual_df = Transformers.AntiLogTransformer(inputCols=["value"]).transform(in_df).withColumn("value", F.round("value", 4))

        
        #print(expected_df.show())
        #print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df)
