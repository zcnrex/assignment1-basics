DEBUG = 2


def print_(
    values: object, level = 0
):
    if DEBUG == level:
        print(values)
