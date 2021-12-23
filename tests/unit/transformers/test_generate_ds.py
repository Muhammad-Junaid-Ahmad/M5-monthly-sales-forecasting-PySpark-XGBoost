from pyspark.sql.types import StructField, StructType, DateType, LongType
import pytest
from pyspark_test import assert_pyspark_df_equal
from transformers import *
import datetime

class TestGenerateDSTransformer():

    @pytest.mark.usefixtures("spark")
    def testValidDateGeneration(self, spark):
        in_df = spark.createDataFrame([(2011, 12),
                                       (2010, 1),
                                       (2015, 5),
                                       (2012, 0)],
                                      ["year", "month"])

        expected_df = spark.createDataFrame([(2011, 12, datetime.date(2011, 12, 1)),
                                             (2010, 1, datetime.date(2010, 1, 1)),
                                             (2015, 5, datetime.date(2015, 5, 1)),
                                             (2012, 0, None)],
                                            schema=StructType([StructField("year", LongType()),
                                                               StructField("month", LongType()),
                                                               StructField("ds", DateType())]))

        actual_df = GenerateDSTransformer(
            inputCols=["year", "month"], outputCol="ds").transform(in_df)

        # print(expected_df.show())
        # print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df)