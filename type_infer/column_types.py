"""
column_types.py

This modules implements a class called `ColumnType` and derived ones which
are used to keep track of the data type for each column in a tabular dataset.

Currently supported data types are

    - **Categorical**:
        Data that represents a class or label and is discrete.
        Currently ``binary``, ``multi-class`` are supported.

    - **Numerical**:
        Data that should be represented in the form of a number.
        Currently ``integer``, ``float``, and ``quantity`` are supported.

    - **Date/DateTime**:
        Time-aware data that is temporal/sequential.
        Currently ``date`` (no time information), and ``datetime`` are supported.

    - **Text**:
        Data that can be considered as language information.
        Currently ``short_text``, and ``rich_text`` are supported. Short text has a
        small vocabulary (~ 100 words) and is generally a limited number of characters.
        Rich text is anything with greater complexity.

    `ColumnType` differs from `dtype` in that `ColumnType` has information about the hierarchy
    of data types. For example, `text` is a more general data type than `numerical`, and
    in turn `numerical` is more general than `float`. This is useful when performing type inference
    in distributed environments. Other types are derived from DataType using this.
"""
from typing import Tuple
from typing import Any


class ColumnType(object):
    """
    Implementatios `ColumnType`.
    """
    def __init__(self, type_name: str):
        """ Initializer

            All column types have a name and are derived from another
            data type except for `text`, which is the more general one.

            @param type_name: str
                name of the data type.
        """
        self.type_ = type_name
        self.origin_ = None

    def get_type(self) -> str:
        """ Returns name of type.
        """
        return self.type_

    def get_parent_types(self) -> Tuple[Any]:
        """ Returns parent data types.
        """
        return self.__class__.__bases__

    def has_parent_types(self) -> bool:
        """ Returns true type is a sub-type.

            Checks for the number of base classes. By construction, all
            ColumnType objects derive from the 'object' class, and the
            ColumnType object itself. Hence, 2 must be subtracted from
            the length of `__bases__`.
        """
        n_c = len(self.get_parent_types)
        return n_c > 2


class Invalid(ColumnType):
    """ Implements invalid column type.
    """
    def __init__(self):
        """ Initializer
        """
        super(Invalid, self).__init__('invalid')


class Text(ColumnType):
    """ Implements text column type.
    """
    def __init__(self):
        """ Initializer
        """
        super(Text, self).__init__('text')


class ShortText(Text):
    """ Implements short-text column type.
    """
    def __init__(self):
        """ Initializer

            ShortText derives from Text.
        """
        self.type_ = 'short-text'
        super(ShortText, self).__init__()


class RichText(Text):
    """ Implements short-text column type.
    """
    def __init__(self):
        """ Initializer

            ShortText derives from Text.
        """
        self.type_ = 'rich-text'
        super(RichText, self).__init__()


class Categorical(Text):
    """ Implements categorical column type.
    """
    def __init__(self):
        """ Initializer.

            Categorical column type derives from text.
        """
        self.type_ = 'categorical'
        super(Categorical, self).__init__()


class MultiClass(Categorical):
    """ Implements multi-class categorical column type.
    """
    def __init__(self):
        """ Initializer.

            Categorical column type derives from categorical.
        """
        self.type_ = 'multi-class'
        super(MultiClass, self).__init__()


class Binary(MultiClass):
    """ Implements binary column type.
    """
    def __init__(self):
        """ Initializer.

            Categorical column type derives from multi-class.
        """
        self.type_ = 'binary'
        super(Binary, self).__init__()


class NonCategorical(Text):
    """ Implements non-categorical column type.
    """
    def __init__(self):
        """ Initializer.

            Non-categorical column type derives from text.
        """
        self.type_ = 'non-categorical'
        super(NonCategorical, self).__init__()


class Numerical(NonCategorical):
    """ Implements numerical column type.
    """
    def __init__(self):
        """ Initializer.

            Numerical column type derives from non-categorical.
        """
        self.type_ = 'numerical'
        super(Numerical, self).__init__()


class Complex(Numerical):
    """ Implements complex-valued numerical column type.
    """
    def __init__(self):
        """ Initializer.

            Real column type derives from numerical.
        """
        self.type_ = 'complex'
        super(Complex, self).__init__()


class Real(Complex):
    """ Implements real-valued numerical column type.
    """
    def __init__(self):
        """ Initializer.

            Real column type derives from complex.
        """
        self.type_ = 'real'
        super(Real, self).__init__()


class Ordinal(Real):
    """ Implements integer-valued numerical column type.
    """
    def __init__(self):
        """ Initializer.

            Ordinal column type derives from real.
        """
        self.type_ = 'real'
        super(Ordinal, self).__init__()


class Date(NonCategorical):
    """ Implements date-time column type.
    """
    def __init__(self):
        """ Initializer.

            DateTime column type derives from non-categorical.
        """
        self.type_ = 'date'
        super(Date, self).__init__()


class DateTime(Date):
    """ Implements date-time column type.
    """
    def __init__(self):
        """ Initializer.

            DateTime column type derives from date.
        """
        self.type_ = 'datetime'
        super(DateTime, self).__init__()
