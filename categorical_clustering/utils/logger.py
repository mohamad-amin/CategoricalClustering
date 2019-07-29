
class VERBOSITY:
    ERROR = 0,
    INFO = 1,
    VERBOSE = 2,


verbosity = VERBOSITY.INFO


def log(message, level=VERBOSITY.INFO):
    if level > verbosity:
        return
    print(message)
