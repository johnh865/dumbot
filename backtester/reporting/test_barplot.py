# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from backtester.reporting.plots import barplot
from bokeh.plotting import figure, show, output_file
from bokeh.io import reset_output
from bokeh.embed import file_html
from bokeh.resources import CDN

x = np.linspace(0, 1, 15)
labels = ['mark-' + str(np.round(xi*100)) for xi in x]
# labels = x.astype(str)
y = np.sin(x)
z = np.cos(x)
d = {'labels' : labels, 'y' : y, 'z' : z}
df = pd.DataFrame(d)


def write_html(fname, plot):
    html = file_html(plot, CDN, 'my plot')
    with open(fname, 'w') as f:
        f.write(html)
    


def test():
    p1 = barplot(df, 'y', 'labels', 'z', tooltipcols=['z'])
    write_html('test1.html', p1)
    p2 = barplot(df, 'y', 'labels', None, tooltipcols=['z'])
    write_html('test2.html', p2)
    p3 = barplot(df, 'y', 'labels', None, tooltipcols=['z'], title='test')
    write_html('test3.html', p3)

# show(p1)
# show(p2)
# show(p3)

if __name__ == '__main__':
    test()