import re
import sre_parse as sre


def is_expression(expression, groups=None, nPatterns=None, exclude=None, verbose=False):
    """Checks if the regular expression fulfills certain criteria

    Arguments:
      expression (str): regular expression to check
      groups (list or None): list of group names that should be present
      nPatterns (int or None): number of patterns to expect
      exclude (list or None): exclude these tokens
      verbose (bool): if True, print reason for expression to not fulfill desired criteria

    Returns:
      bool: True if the expression fulfills the desired criteria
    """

    try:
        p = re.compile(expression)
    except re.error as e:
        if verbose:
            print(f"Invalid regular expression: {e}")
        return False

    gd = p.groupindex

    if groups is None:
        groups = []
    elif groups is all:
        groups = gd.keys()

    for gn in gd.keys():
        if gn not in groups:
            if verbose:
                print(f'Expression contains a non-required group {gn}!')
            return False

    for gn in groups:
        if gn not in gd.keys():
            if verbose:
                print(f'Expression does not contain required group {gn}!')
            return False

    if exclude is None:
        exclude = []

    n = 0
    parsed = sre.parse(expression)
    for token in parsed:
        lit = token[0]
        if lit != sre.LITERAL and lit not in exclude:
            n += 1

    if nPatterns is not None:
        if nPatterns > 0:
            if n != nPatterns:
                if verbose:
                    print(f'Expression has more than {nPatterns} regular expression patterns!')
                return False
        elif nPatterns < 0:
            if n <= -nPatterns:
                if verbose:
                    print(f'Expression has less than {-nPatterns} regular expression patterns!')
                return False
        else:
            if n == 0:
                if verbose:
                    print('Expression has no regular expression patterns!')
                return False

    return True
