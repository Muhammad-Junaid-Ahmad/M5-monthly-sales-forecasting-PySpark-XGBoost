from hyperopt import hp
from hyperopt.pyll.base import scope
from pyspark.sql import DataFrame
# function for converting a dictionary read from a json file into a hyperopt space
def json_to_space(j):
    space = {}

    for key, value in j.items():
        if value[0] == 'quniform':
            space[key] = hp.quniform(key, *value[2:])
        elif value[0] == 'choice':
            space[key] = hp.choice(key, *value[2:])

        if value[1] == 'int':
            space[key] = scope.int(space[key])
        # etc ...

    return space


def df_split(df: DataFrame, year, month=0):
    df1 = df.filter("year<{} or year={} and month<={}".format(year, year, month))
    df2 = df.filter("year>{} or year={} and month>{}".format(year, year, month))
    
    return df1, df2


def visualize_predictions():
    pass