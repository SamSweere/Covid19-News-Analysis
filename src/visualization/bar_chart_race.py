import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
from datetime import datetime, timedelta


# taken from: https://towardsdatascience.com/bar-chart-race-in-python-with-matplotlib-8e687a5c8a41

def get_sent_color(x):
    if x > 0:
        return (1-x, 1, 1-x)
    else:
        return (1, 1+x, 1+x)


def draw_barchart(ax, df, date, date_column:str, name_column:str, group_column:str, value_column:str, color_column:str):
    """ 
    @param df: dataframe to be used
    @param date: current date for which to display values
    @param name_column: name of the entity
    @param group_column: -
    @param value_column: number of counts
    """
    # sort by values of current year
    dff = df[df[date_column].eq(date)].sort_values(by=value_column, ascending=True).tail(10)
    # re-define axis object
    ax.clear()
    # TODO mix colors in RGB!
    dff["color"] = dff[color_column].apply(get_sent_color)
    ax.barh(dff[name_column], dff[value_column], color=dff["color"]) # for groups only: , color=[colors[group_lk[x]] for x in dff[name_column]])
    dx = dff[value_column].max() / 200
    # bar names
    for i, (value, name) in enumerate(zip(dff[value_column], dff[name_column])):
        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')
        # # for group only
        # ax.text(value-dx, i-.25, group_lk[name], size=10, color='#444444', ha='right', va='baseline')
        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')
    # axis labels
    ax.text(1, 0.4, str(date)[0:10], transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    ax.text(0, 1.06, 'Primary entity of n articles (coreference resolved)', transform=ax.transAxes, size=12, color='#777777')
    # axis ticks
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.15, 'Most mentioned entities',
            transform=ax.transAxes, size=24, weight=600, ha='left', va='top')
    ax.text(1, 0, 'by @AlexanderReisach, @SamSweere', transform=ax.transAxes, color='#777777', ha='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)

# def curry_barchart_standard(days):
#     """ keep tutorial version around for comparison only """
#     draw_barchart(df, days, "year", "name", "group", "value")


def create_barchart_race(df, start_date, end_date, name_col, color_col):

    # TODO use start_date, end_date
    period_length = (end_date - start_date).days

    def curry_barchart(days):
        nonlocal df
        nonlocal ax
        date = start_date + timedelta(days)

        # CUSTOM
        draw_barchart(
            ax=ax,
            df=df,
            date=date,
            date_column="publication_date",
            name_column=name_col,
            group_column=None,
            value_column="cum_sum",
            color_column=color_col
        )
    fig, ax = plt.subplots(figsize=(15, 8))

    animator = animation.FuncAnimation(fig, curry_barchart, frames=range(0, period_length))
    animator.save("src/figures/animation_attempt.mp4", fps=1)
    plt.close("all")
