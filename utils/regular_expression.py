import re
import sre_parse as sre


def is_expression(expression, groups=None, nPatterns=None, exclude=None, verbose=False):
    """Checks if the regular expression fullfill certain criteria

    Arguments:
      expression (str): regular expression to check
      groups (list or None): list of group names that should be present
      nPatterns (int or None): number of patterns to expect
      exclude (list or None): exculde these tokens
      verbose (bool): if True, print reason for expression to not fullfil desired criteria

    Returns
      bool: True if the expression fullfills the desired criteria
    """

    # parse regular expression
    # p = sre.parse(expression);

    # group patterns
    # gd = p.pattern.groupdict

    p = re.compile(expression)

    # Access the group dictionary directly from the compiled regex
    gd = p.groupindex

    if groups is None:
        groups = []
    elif groups is all:
        groups = gd.keys()

    for gn in gd.keys():
        if gn not in groups:
            if verbose:
                print('Expression does contain a non required group %s!' % gn)
            return False

    for gn in groups:
        if gn not in gd.keys():
            if verbose:
                print('Expression does not contain required group %s!' % gn)
            return False

    if exclude is None:
        exclude = []

    n = 0
    p = sre.parse(expression)
    for i, l in enumerate(p):
        lit = l[0]
        if lit != sre.LITERAL and lit not in exclude:
            n = n + 1

    if nPatterns is not None:
        if nPatterns > 0:
            if n != nPatterns:
                if verbose:
                    print('Expression has more than %d regular expression patterns!' % nPatterns)
                return False

        elif nPatterns < 0:
            if n <= -nPatterns:
                if verbose:
                    print('Expression has less than %d regular expression patterns!' % -nPatterns)
                return False

        else:
            if n == 0:
                if verbose:
                    print('Expression has no regular expression patterns!')
                return False

    return True