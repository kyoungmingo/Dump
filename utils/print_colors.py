class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def _print_color_base(text, color_type, *args, **kwargs):
    print(color_type + text + BColors.ENDC, *args, **kwargs)


def print_warning(text, *args, **kwargs):
    _print_color_base(text, BColors.WARNING, *args, **kwargs)


def print_green(text, *args, **kwargs):
    _print_color_base(text, BColors.OKGREEN, *args, **kwargs)


def print_blue(text, *args, **kwargs):
    _print_color_base(text, BColors.OKBLUE, *args, **kwargs)
