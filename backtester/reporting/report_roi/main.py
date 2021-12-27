"""
See 
https://github.com/bokeh/bokeh/tree/branch-3.0/examples/app/movies

To Run
------
bokeh serve --show report_roi


"""
import numpy as np
import pandas as pd
import datetime
import time 
import pdb 
from os.path import dirname, join

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, Div, Select, Slider, TextInput,
                          DateRangeSlider, RangeSlider)

from bokeh.plotting import figure, show, output_file


fname = __file__
basedir = dirname(dirname(dirname(dirname(fname))))
print('directory.......................')
print(basedir)

import sys
sys.path.append(basedir)


from bokeh.plotting import figure
from calculate import PostData


_filepath = __file__
_dirpath = dirname(_filepath)
_picklepath = join(_dirpath, 'roi_output.pkl')

# df_stats = pd.read_pickle(_picklepath)

postdata = PostData(_picklepath)
# df_stats = data.roi

# df_stats['len_yrs'] = df_stats['len'] / 253
# df_stats['alpha'] = .5
# df_stats['color'] = 'grey'

# roi_p01 = np.percentile(df_stats['avg 91'], 5)
# roi_mean = df_stats['avg 91'].mean()
# roi_p99 = np.percentile(df_stats['avg 91'], 95)

# std_p01 = np.percentile(df_stats['std 91'], 5)
# std_mean = df_stats['std 91'].mean()
# std_p99 = np.percentile(df_stats['std 91'], 95)

axis_map = {
    'ROI 7 day (%)'  : 'avg 7',
    'ROI 30 day (%)' : 'avg 30',
    'ROI 91 day (%)' : 'avg 91',
    'STD 7 day (%)'  : 'std 7',
    'STD 30 day (%)' : 'std 30',
    'STD 91 day (%)' : 'std 91',
    
    }

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), sizing_mode="stretch_width")


# Create Input controls
# ---------------------
# yrs_data_slider = Slider(title='Mininum # of data years',
#                    value=5, start=1, end=20, step=1)

today = datetime.date.today()
year_slider = DateRangeSlider(title='Years to process', 
                          value=(datetime.datetime(2017,1,1), today),
                          start=datetime.datetime(2001,1,1),
                          end=today)

x_axis_selecter = Select(title='X Axis', 
                         options=sorted(axis_map.keys()),
                         value='STD 30 day (%)'
                         )
highlighter_input = TextInput(title='Symbols to highlight')

y_axis_selecter = Select(title='Y Axis', 
                         options=sorted(axis_map.keys()),
                         value='ROI 30 day (%)'
                         )
# x_lim_slider = RangeSlider(title='X Axis limits',
#                            value = std_mean,
#                            start = std_p01,
#                            end = std_p99,)



source = ColumnDataSource(
    data=dict(x=[], y=[], color=[], symbol=[], alpha=[], numpts=[]))
TOOLTIPS = [
    ('Symbol', '@symbol'),
    ('# yrs', '@num_yrs')
    ]

p = figure(height = 300,
           width = 450, 
           title = "", 
           # toolbar_location = None, 
           tooltips = TOOLTIPS,
           # tools
            sizing_mode = "scale_both"
           )

p.circle(
    x="x",
    y="y",
    source=source, 
    size=4, 
    color="color", 
    line_color=None, 
    fill_alpha="alpha",
    )


def select_symbols():
    time1, time2 = year_slider.value_as_datetime
    time1, time2 = np.datetime64(time1), np.datetime64(time2)
    
    df_stats = postdata.roi_between(time1, time2)
    maxlen = df_stats['len 7'].max()
    selected = df_stats.loc[df_stats['len 7'] == maxlen]
    # selected = df_stats
    selected = selected.copy()
    
    selected['color'] = 'gray'
    selected['alpha'] = 0.5
    
    
    # selected = df_stats[df_stats['len_yrs'] > yrs_data_slider.value]
    symbols_highlighted = highlighter_input.value.split()
    print(symbols_highlighted)
    for symbol in symbols_highlighted:
        selected.loc[symbol, 'color'] = 'orange'
        selected.loc[symbol, 'alpha'] = 1.0
    return selected


def update():
    df = select_symbols()
    print('len...', len(df))
    x_name = axis_map[x_axis_selecter.value]
    y_name = axis_map[y_axis_selecter.value]
    p.xaxis.axis_label = x_axis_selecter.value
    p.yaxis.axis_label = y_axis_selecter.value
    p.title.text = '%d symbols seleted' % len(df)
    
    source.data = dict(
        x = df[x_name]*100,
        y = df[y_name]*100,
        alpha = df['alpha'],
        color=df['color'],
        symbol=df.index,
        num_yrs=df['len 7']/52.1429
        )


controls = [year_slider, x_axis_selecter, y_axis_selecter, highlighter_input]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())
    
inputs = column(*controls, width=250)
l = column(desc, row(inputs, p), sizing_mode='scale_both')
update()
curdoc().add_root(l)
curdoc().title = 'Movies'



if __name__ == '__main__':
    df = select_symbols()
    update()
    show(l)