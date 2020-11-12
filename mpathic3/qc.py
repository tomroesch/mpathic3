import re

def get_cols_from_df(df, col_types):
    """
    Returns all colnames of a given type from a df, sorted alphabetically
    """
    return sorted([c for c in df.columns if is_col_type(c, col_types)])


def is_col_type(col_name, col_types='all'):
    """ 
    Checks whether col_name is a valid column name, as specified by col_types. col_types can be either a string (for a single column type) or a list of strings (for multimple column types). Default col_types='all' causes function to check all available column types
    """
    col_match = False

    # Make col_types_list
    if type(col_types) == list:
        col_types_list = col_types
    elif type(col_types) == str:
        if col_types=='all':
            col_types_list = col_patterns.values()
        else:
            col_types_list = [col_types]
    else:
        raise SortSeqError('col_types is not a string or a list.')

    # Check for matches wihtin col_type list
    for col_type in col_types_list:
        pattern = col_patterns[col_type]
        if re.search(pattern, col_name):
            col_match = True

    # Return true if any match found
    return col_match