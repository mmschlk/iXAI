from typing import Optional, Generator, Any, Iterable, List

BASE_COLOR = '#a6a7a9'  # base color for elements in a plot that should not be highlighted / colored
BACKGROUND_COLOR = '#f8f8f8'  # background color of plot objects (canvas)

DEFAULT_COLOR_LIST = ['#ef27a6', '#4ea5d9', '#7d53de', '#44cfcb', '#44001A']  # default list of colors for highlighting elements
# in a plot

STD_ALPHA = 0.25  # alpha-channel value for std tunnel around an element (line) for a plot


def get_color_with_generator(
        color_generator: Generator[str, None, None],
        item_id: Optional[Any] = None,
        color_item_ids: Optional[List[Any]] = None,
        base_color: Optional[str] = None
) -> str:
    if base_color is None:
        base_color = BASE_COLOR
    if color_item_ids is not None and item_id not in color_item_ids:
        return base_color
    else:
        return next(color_generator)


def color_list_generator(
        color_list: Optional[List[str]] = None
) -> Iterable[str]:
    if color_list is None:
        color_list = DEFAULT_COLOR_LIST
    color_list = color_list * 100  # upper bound
    for color in color_list:
        yield color
