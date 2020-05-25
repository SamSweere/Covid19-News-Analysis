import numpy as np
import pandas as pd

import sys

from datetime import datetime, timedelta

sys.path.append("src/")
sys.path.append("src/visualization/")
import visualization.get_viz_data as get_viz_data
import visualization.matplotlib_viz as viz
import visualization.bar_chart_race as bar_chart_race


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
        df_gb = df.groupby(by=["publication_date", mc_column])
        df_most_common = df_gb.agg(sum).reset_index()
        return df_most_common

def visualize(df_most_common, start_date, end_date, name_col, color_col):
        print("Starting Visualization...\t", str(datetime.now()))
        # viz.animate_NER(df_most_common)
        bar_chart_race.create_barchart_race(df_most_common, start_date, end_date, name_col, color_col)


def prepare_viz(df_most_common, mc_column="mc_p", mc_num_column="mc_p_num", with_sentiment=True):
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
        return df_most_common



if __name__ == "__main__":

    # TODO: dates 2019-11-06 and 2020-01-01 throw errors
    start_date=datetime.strptime("2020-03-01", "%Y-%m-%d")
    end_date=datetime.strptime("2020-04-05", "%Y-%m-%d")
    
    # run_and_save(start_date, end_date, articles_per_period = 1000, max_length = 500, debug=True)

    # TODO load all dataframes for one experiment and concat them together
    df_most_common = get_viz_data.load_data("new_run_s_01_03_2020_e_05_04_2020_app_100_ml_300_d_25_05_t_18_51")
    df_most_common = prepare_viz(df_most_common, mc_column="mc_p", mc_num_column="mc_p_num", with_sentiment=True)
    print(df_most_common.head())
    visualize(df_most_common, start_date, end_date, "mc_p", "g_sent")
