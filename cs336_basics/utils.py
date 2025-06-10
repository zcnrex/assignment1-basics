DEBUG = 6


def print_(msg: str, values: object, level=0):
    if DEBUG == level:
        print(f"{msg} {values}")
