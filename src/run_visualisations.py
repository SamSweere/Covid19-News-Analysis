import numpy as np
import pandas as pd

import sys

from datetime import datetime, timedelta

sys.path.append("src/")
sys.path.append("src/visualization/")
import visualization.get_viz_data as get_viz_data
import visualization.matplotlib_viz as viz
import visualization.bar_chart_race as bar_chart_race
import visualization.world_map as world_map
from visualization.visualize_sentiment_over_time import visualize_sent


def cum_sum_df(df, mc_column, mc_num_column):
        """ return df with monotonically increasing entity mentions """
        # TODO this does not ensure continuity in case some entity is missing entirely on some days
        # should be kinda rare but still possible...
        df = df.sort_values(by=["publication_date", mc_column])
        df_gb = df.groupby(by=[mc_column])
        # TODO cumsum stuff doesn't seem to be working yet
        df["cum_sum"] = df_gb[mc_num_column].apply(pd.Series.cumsum)
        df.sort_values(by=["publication_date", "cum_sum"], ascending=False, inplace=True)
        return df

def average_sent_columns(df, mc_num_column):
        sent_columns = [i for i in df.columns if i.endswith("sent")]
        for i in sent_columns:
            df[i] = df[i] / df[mc_num_column]
        return df

def fill_entity_gaps(df_most_common, mc_column, mc_num_column):
        """
        make sure we keep an absent but known entity around
        to avoid the bar from disappearing in bar chart race 
        """
        # identify missing values
        all_entities = set(df_most_common[mc_column].unique())
        df_gb = df_most_common.groupby(by=["publication_date"])
        date_entities = df_gb[mc_column].aggregate(lambda x: set(x))
        
        # TODO add the full rows!
        # collect rows to add
        publication_idx = list(df_most_common.columns).index("publication_date")
        mc_column_idx = list(df_most_common.columns).index(mc_column)
        mc_column_num_idx = list(df_most_common.columns).index(mc_num_column)
        rows_to_add = []
        for i, d_e in enumerate(date_entities):
            missing_ents = all_entities.difference(d_e)
            date = date_entities.index[i]
            # i: column we want to more or less replicate
            for e in missing_ents:
                row = [np.NaN] * len(df_most_common.columns)
                row[publication_idx] = date  # publication_date
                row[mc_column_idx] = e  # most_common_1
                row[mc_column_num_idx] = 0.0   # most_common_1_num
                rows_to_add.append(tuple(row))

        # add missing rows and sort again
        df_missing = pd.DataFrame(rows_to_add, columns=df_most_common.columns)
        df_most_common = pd.concat((df_most_common, df_missing), axis=0)
        df_most_common.sort_values(by=[mc_column, "publication_date"], inplace=True)
        df_most_common.fillna(method="ffill", inplace=True)
        df_most_common.dropna(subset=["g_sent"], inplace=True)
        df_most_common.sort_values(by=["publication_date", mc_column], inplace=True)
        return df_most_common.reset_index(drop=True)

def select_most_common_per_period(df_most_common):
        # select top 10 for each publication date by cum_sum
        df_most_common = df_most_common.sort_values(by=["publication_date", "cum_sum"], ascending=False).reset_index()
        df_most_common = df_most_common.groupby(by=["publication_date"])
        df_most_common = df_most_common.head(10)
        return df_most_common

def sum_period_most_common_entities(df, mc_column):
        """ 
        @param df: DataFrame with "nlp" column containing spacy preprocessing
        @param visualization: create bar chart race
        """
        # sum up entities for each publication date
        df['counts'] = 1
        df_gb = df.groupby(by=["publication_date", mc_column])
        

        df_most_common = df_gb.agg(sum).reset_index()
        return df_most_common

def visualize(df_most_common, start_date, end_date, name_col, color_col):
        print("Starting Visualization...\t", str(datetime.now()))
        # viz.animate_NER(df_most_common)
        bar_chart_race.create_barchart_race(df_most_common, start_date, end_date, name_col, color_col)


def prepare_viz(df_most_common, mc_column="mc_p", mc_num_column="mc_p_num", sent_col="g_sent", with_sentiment=True):
        df_most_common = sum_period_most_common_entities(df_most_common, mc_column)
        df_most_common[mc_column].replace("None", np.NaN, inplace=True)
        df_most_common.dropna(subset=[mc_column], inplace=True)
        if "Unnamed: 0" in df_most_common.columns:
                df_most_common.drop(columns=["Unnamed: 0"], inplace=True)
        if with_sentiment:
                df_most_common = average_sent_columns(df_most_common, mc_num_column)
                df_most_common.replace(np.inf, np.NaN, inplace=True)
                df_most_common.replace(-np.inf, np.NaN, inplace=True)
        df_most_common["publication_date"] = df_most_common["publication_date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        df_most_common.sort_values(by=["publication_date", mc_num_column], ascending=False, inplace=True)
        df_most_common.reset_index(drop=True, inplace=True)
        df_most_common = fill_entity_gaps(df_most_common, mc_column, mc_num_column)
        # now they should all have the same entities
        # test_date = datetime.strptime("2020-03-08", "%Y-%m-%d")
        # df_most_common.loc[df_most_common.publication_date=="2020-03-15", :].shape
        df_most_common = cum_sum_df(df_most_common, mc_column, mc_num_column)
        df_most_common = select_most_common_per_period(df_most_common)
        
        #### HOT FIX: remove Nicola Tesla, Cain and Abel
        if sent_col == "mc_p_sent":
                mask = df_most_common.mc_p.apply(lambda x: x not in ["Nikola Tesla", "Cain and Abel"])
                df_most_common = df_most_common[mask]
        # df_most_common[sent_col] = (df_most_common[sent_col] + 1)**2
        temp = df_most_common[sent_col] - df_most_common[sent_col].mean()
        sent_range = df_most_common[sent_col].max() - df_most_common[sent_col].min()
        df_most_common["sent_norm"] = temp/sent_range
        df_most_common["rolling_sent_norm"] = df_most_common["sent_norm"].rolling(3).mean()
        ###

        df_most_common["rolling_sent"] = df_most_common[sent_col].rolling(5).mean()
        df_most_common.dropna(subset=["rolling_sent"], inplace=True)
        return df_most_common

def prepare_countries(df, mc_column="mc_c"):
        df = sum_period_most_common_entities(df, mc_column)
        df[mc_column].replace("None", np.NaN, inplace=True)

        # Fixes:
        df[mc_column].replace("Kingdom of Italy", "Italy", inplace=True)
        df[mc_column].replace("Republic of Ireland", "Ireland", inplace=True)
        # df[mc_column].replace("Singapore", "Malaysia", inplace=True)
        

        df.dropna(subset=[mc_column], inplace=True)
        if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)

        
        df = df.groupby([mc_column]).sum().reset_index()

        df = df[[mc_column, mc_column+"_sent", 'counts']]
        # counts_df = df[mc_column].value_counts().rename_axis('country').reset_index(name='counts')
        # df[mc_columnt].groupby(mc_column)[mc_column].count()
        # print(counts_df) 
        
        # df["publication_date"] = df["publication_date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        # df.sort_values(by=["publication_date"], ascending=False, inplace=True)
        # df.reset_index(drop=True, inplace=True)
        return df

def prepare_sentiment(df, sent_column="c_sent"):
        df['counts'] = 1
        df = df.groupby(by=["publication_date"])
        

        df = df.agg(sum).reset_index()

        df["avg_sent"] = df.apply(lambda x: pd.Series((x[sent_column]/x["counts"])), axis=1)

        df = df[["publication_date", 'avg_sent']]

        return df

if __name__ == "__main__":
        df = get_viz_data.load_data("final_run")
        
        df = df.drop_duplicates()
        # -------  Country visualizer -------         
        # df_country = prepare_viz(df, mc_column="mc_p", mc_num_column="mc_p_num",
        #         sent_col="mc_p_sent", with_sentiment=True)
        # print(df_country.head())
        # start_date = df_country.publication_date.min()
        # end_date = df_country.publication_date.max()
        # visualize(df_country, start_date, end_date, "mc_p", "rolling_sent")
        
        # -------  Entity visualizer ------- 
        
        df_most_common = prepare_viz(df, mc_column="mc_p", mc_num_column="mc_p_num",
                sent_col="mc_p_sent", with_sentiment=True)
        print(df_most_common.head())

        df_most_common["sent_bin"] = df_most_common["sent_norm"].apply(lambda x: 1 if x>0 else -1)

        start_date = df_most_common.publication_date.min()
        end_date = df_most_common.publication_date.max()
        visualize(df_most_common, start_date, end_date, "mc_p", "mc_p_sent")
        # visualize(df_most_common, start_date, end_date, "mc_p", None)

        # -------  Country visualizer ------- 
        
        # c_list = prepare_countries(df, mc_column="mc_c")

        # world_map.show_world_map(c_list)

        # --------- Sentiment visualizer ---------

        # --- corona sent
        # df_sent = prepare_sentiment(df, sent_column="c_sent")
        # visualize_sent(df_sent, name="Corona")


        # --- general sent
        # df_sent = prepare_sentiment(df, sent_column="g_sent")
        # visualize_sent(df_sent, name="general")

        pass
