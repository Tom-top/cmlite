def parallel_traceback(func):
    """Wrapper to obtain a full traceback when exceuting a function in parallel.

    Arguments
    ---------
    func : function
     The function to call.

    Returns
    -------
    wrapper : function
      The function wrapped with an appropiate traceback functionality.
    """
    import traceback, functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = "{}\n\nOriginal {}".format(e, traceback.format_exc())
            raise type(e)(msg)

    return wrapper