import pytest
from pyspark_test import assert_pyspark_df_equal
from transformers import DFSplitter

class TestDFSplitterTransformer():

    @pytest.mark.usefixtures("spark")
    def testDFSplitOnYear(self, spark):
        in_df = spark.createDataFrame([(2011, 12),
                                       (2010, 1),
                                       (2015, 5),
                                       (2012, 0)],
                                      ["year", "month"])

        expected_df1 = spark.createDataFrame([(2011, 12),
                                             (2010, 1),
                                             (2012, 0)],
                                            ["year", "month"])

        actual_df1, _ = DFSplitter(splittingYM={"year":2012, "month":0}).transform(in_df)

        # print(expected_df.show())
        # print(actual_df.show())

        assert_pyspark_df_equal(actual_df1, expected_df1)

    @pytest.mark.usefixtures("spark")
    def testDFSplitOnYearAndMonth(self, spark):
        in_df = spark.createDataFrame([(2011, 12),
                                       (2010, 1),
                                       (2015, 5),
                                       (2012, 0)],
                                      ["year", "month"])

        expected_df2 = spark.createDataFrame([(2015, 5)], ["year", "month"])
                                             

        _, actual_df2 = DFSplitter(splittingYM={"year":2012, "month":0}).transform(in_df)

        # print(expected_df.show())
        # print(actual_df.show())

        assert_pyspark_df_equal(actual_df2, expected_df2)

    @pytest.mark.usefixtures("spark")
    def testDFSplitForWrongColNames(self, spark):
        in_df = spark.createDataFrame([(2011, 12),
                                       (2010, 1),
                                       (2015, 5),
                                       (2012, 0)],
                                      ["years", "months"])
                                     
        with pytest.raises(Exception):
            _, actual_df2 = DFSplitter(splittingYM={"year":2012, "month":0}).transform(in_df)
