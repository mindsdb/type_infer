""" base.py

    Base implementation of engines to infer data type and other
    useful information from tables.
"""
import pandas


class ColumnInfo:
    """ Simple container for important column information
        like data type, modifiers, length, number of
        invalid entries, etc.
    """
    def __init__(self, column_name: str):
        """ Initializer

            :param column_name (str)
                name of column
        """
        # base attributes
        self.name_ = column_name
        self.length_ = -1
        self.data_type_ = 'unknown'
        self.data_type_info_ = {}
        self.modifier_ = 'unkonwn'
        self.modifier_info_ = {}

    def set_column_length(self, length: int):
        """ Set length of column.
        """
        self.length_ = length

    def get_column_length(self):
        """ Returns column length.

            :note
            if `set_column_length()` hasn't been called, then
            `get_column_length()` will return -1.
        """
        return self.length_

    def get_name(self):
        """ Return column name.
        """
        return self.name_

    def get_data_type(self):
        """ Returns column data type.
        """
        return self.data_type_

    def get_data_type_info(self):
        """ Returns copy of data type information.
        """
        d = {}
        d.update(self.data_type_info_)
        return d

    def get_modifier(self):
        """ Returns column modifier.
        """
        return self.modifier_

    def get_modifier_info(self):
        """ Returns copy of modifier info.
        """
        d = {}
        d.update(self.modifier_info_)
        return d


class TypeInferenceEngine:
    """ Base implementation for column type inference.
    """
    def __init__(self, name: str):
        """ Initializer

            :param name (str)
                name of the engine.
        """
        self.name_ = name
        self.dfs_ = []

    def attach_dataframe(self, df: pandas.DataFrame):
        """ Adds dataframe for analysis.

            :param df (pandas.Dataframe)
                dataframe to be analyzed.

            :note
                to avoid side effects, a copy of the original
                dataframe is made.
        """
        self.dfs_.append(df.copy())

