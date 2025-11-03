import pandas as pd
from IPython.display import HTML, display

def show_scrollable(df, max_height=400, max_width=1000):
    """
    Display a DataFrame inside a scrollable <div>.
    max_height / max_width are in px; tweak as you like.
    """
    html = df.to_html(max_rows=None, max_cols=None)
    div  = (f'<div style="max-height:{max_height}px; max-width:{max_width}px; '
            f'overflow:auto; border:1px solid lightgrey; padding:4px">'
            f'{html}</div>')
    display(HTML(div))