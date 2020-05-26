import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
import datetime

import seaborn as sns

def visualize_sent(df, name):
    start_date = datetime.datetime.strptime(df.publication_date.min(), '%Y-%m-%d')
    end_date = datetime.datetime.strptime(df.publication_date.max(), '%Y-%m-%d')

  
    formatter = DateFormatter('%d-%m-%y')
    delta = datetime.timedelta(days=1)
    
    dates = drange(start_date, end_date + datetime.timedelta(days=1), delta)
    
    df['dates'] = dates

    p = sns.lmplot(x = 'dates', y = 'avg_sent', data=df, aspect=11.7/8.27)
    p.ax.xaxis.set_major_formatter(formatter)
    p.ax.xaxis.set_tick_params(rotation=30, labelsize=10)

    p.set_xlabels("Date")
    p.set_ylabels("Average Sentiment")

    plt.savefig("figures/avg_sent_" + name + ".png", dpi=200)
    
    plt.show()
