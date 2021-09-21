import pandas as pd
import numpy as np
import pickle


def process_strings(data):
    """
    prepare the string by cleaning and normalizing it
    :param data: data to be pre processed
    :return: processed data
    """
    series = pd.Series(data["text"], dtype="string")

    # remove html tags
    series = series.str.replace("[<][a-zA-Z]+ [/][>]+", "", case=False, regex=True)

    # normalize string
    series = series.str.lower()
    series = series.str.findall("[a-zA-Z]+")
    # TODO: add emoticons

    return series.str.join(" ")
