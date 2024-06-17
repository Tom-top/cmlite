def as_type(value, types=[int, float]):
    """Tries to convert value to given data types.

    Arguments
    ---------
    value : object
      The value to be converted.
    types : list of types
      The list of types to try to convert the value to.

    Returns
    -------
    value : object
      The value converted to the types if possible.
    """
    for t in types:
        try:
            return t(value)
        except Exception:
            pass
    return value


def ensure(value, dtype):
    """Ensure values have a specified type but allowing for None values.

    Arguments
    ---------
    value : object
      The value to copy
    dtype : class
      The class type of the value to be copied.

    Returns
    -------
    value : object
      The value with the requested type.
    """
    if value is None:
        return None
    else:
        if not isinstance(value, dtype):
            value = dtype(value)
        return dtype(value)
