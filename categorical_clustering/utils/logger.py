
class LEVEL:
    ERROR = 0,
    INFO = 1,
    VERBOSE = 2,


verbosity = LEVEL.INFO


def log(message, tabs=0, level=LEVEL.INFO):
    if level > verbosity:
        return
    print(get_tabs(tabs) + message)


def get_tabs(count):
    return '  ' * count
