COLOR_LIST = ['#ef27a6', '#4ea5d9', '#7d53de', '#44cfcb']
BASE_COLOR = '#a6a7a9'
BACKGROUND_COLOR = '#f8f8f8'
STD_ALPHA = 0.2
SAVE_DPI = 300


def get_color_with_generator(color_generator, item=None, item_selection=None, base_color=None):
    if base_color is None:
        base_color = BASE_COLOR
    if item_selection is not None and item not in item_selection:
        return base_color
    else:
        return next(color_generator)


def color_list_generator(color_list=None):
    if color_list is None:
        color_list = COLOR_LIST
    color_list = color_list * 100  # upper bound
    for color in color_list:
        yield color
