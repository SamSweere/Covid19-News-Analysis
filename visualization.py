""" Visualization Module """

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import FuncAnimation
from matplotlib.animation import writers
from datetime import datetime, timedelta


# # show in browser using displacy
# test_id = 5
# displacy.serve(nlp(str(articles[test_id])), style='ent')
# displacy.serve(nlp(str(articles[test_id])), style='dep', options = {'distance': 120})
import os
if not os.path.exists("figures"):
    os.makedirs("figures")

def animate_NER(df_most_common):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax = sns.barplot(data=df_most_common, x="most_common_1_num", y="most_common_1")

    def animate(i):
        animation_start_date = min(df_most_common["publication_date"])
        new_date = animation_start_date + timedelta(days=i)
        df_filtered = df_most_common.loc[df_most_common["publication_date"]==new_date,:]
        plt.cla()  # clear current axes
        ax = sns.barplot(data=df_filtered, x="most_common_1_num", y="most_common_1")
        ax.set(xlabel=str(new_date).split(" ")[0])

    n_frames = len(df_most_common["publication_date"].unique()) #Number of frames
    anim = FuncAnimation(fig, animate, repeat=True, blit=False, frames=n_frames-1, interval=800)
    
    # save as .gif
    anim.save("figures/topic_trends.gif", writer='imagemagick', fps=1)

    # save as .mp4 for yt upload
    FFwriter = FFMpegWriter(fps=1, codec="libx265")     
    anim.save('figures/topic_trends.mp4', writer=FFwriter)

