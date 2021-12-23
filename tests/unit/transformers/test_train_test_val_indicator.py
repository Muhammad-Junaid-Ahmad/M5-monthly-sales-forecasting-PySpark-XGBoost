import pytest
from pyspark_test import assert_pyspark_df_equal
from transformers import TrainTestValIndicator
import math
import datetime
from pyspark.sql.types import IntegerType, StructField, StructType, DateType, LongType


class TestTrainTestValIndicator():
    @pytest.mark.usefixtures("spark")
    def testIndicatorGeneration(self, spark):
        in_df = spark.createDataFrame([(2011, 12),
                                       (2010, 1),
                                       (2015, 5),
                                       (2015, 2),
                                       (2015, 6),
                                       (2016, 2),
                                       (2012, 0)],
                                      ["year", "month"])

        expected_df = spark.createDataFrame([(2011, 12, 0),
                                            (2010, 1, 0),
                                            (2015, 5, 1),
                                            (2015, 2, 1),
                                            (2015, 6, 2),
                                            (2016, 2, 2),
                                            (2012, 0, 0)],
                                            schema=StructType([StructField("year", LongType()),
                                                               StructField("month", LongType()),
                                                               StructField("train_test_val_indicator", IntegerType())]))

        actual_df = TrainTestValIndicator( indicatorCond={"train_before": {"year":2015, "month": 0},  "test_after": {"year":2015, "month": 5}} )\
                                .transform(in_df)

        #print(expected_df.show())
        #print(actual_df.show())

        assert_pyspark_df_equal(actual_df, expected_df)
