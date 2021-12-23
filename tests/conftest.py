import pytest
from pyspark.sql import SparkSession


# Create a SparkSession (Note, the config section is only for Windows!)
@pytest.fixture(scope="session")
def spark(request):
    print("Setting up Spark Session")
    spark = SparkSession.builder.master("local[*]").appName("unit_tests").getOrCreate() 

    def teardown():
        print("Tearing Down Spark Session")
        spark.stop()
    
    request.addfinalizer(teardown)
    return spark
