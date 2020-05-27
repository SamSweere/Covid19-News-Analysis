import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import country_converter as coco

def show_world_map(df):

    # Convert the country names to ISO3
    df["iso3"] = df['mc_c'].apply(lambda x: coco.convert(names=[x], to="ISO3"))
    df["country_short"] = df['mc_c'].apply(lambda x: coco.convert(names=[x], to='name_short'))
    # standard_names = coco.convert(names=some_names, to='ISO3')
    
    # Drop the long names column
    df.drop(columns=["mc_c"], inplace=True)

    # Remove rows that did not result in anything
    df = df[df.iso3 != "not found"]

    mask = df['iso3'].apply(lambda x: isinstance(x, list))
    df = df[~mask]
    # Re aggregate
    df = df.groupby(by=["iso3", "country_short"]).sum()
    df = df.reset_index()

    # Normalize the sentiment
    df["norm_sent"] = df.apply(lambda x: pd.Series(round((x["mc_c_sent"]/x["counts"]),2)), axis=1)

    # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')


    # -- Referal counts ------
    fig = go.Figure(data=go.Choropleth(
        locations = df['iso3'],
        z = df['counts'],
        text = df['country_short'],
        colorscale = 'Reds',
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title = 'Mentions',
    ))

    fig.update_layout(
        title_text='Mentions in Corona News',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        annotations = [dict(
            x=0.55,
            y=0.1,
            xref='paper',
            yref='paper',
            text='by @SamSweere, @AlexanderReisach',
            showarrow = False
        )]
    )
    fig.write_html("figures/country_mentions.html")

    fig.show()
    


    # -- Sentiment counts ------
    fig = go.Figure(data=go.Choropleth(
        locations = df['iso3'],
        z = df['norm_sent'],
        text = df['country_short'],
        colorscale = 'RdYlGn',
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title = 'Sentiment towards',
    ))

    fig.update_layout(
        title_text='Sentiment towards mentioned countries in Corona News',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        annotations = [dict(
            x=0.55,
            y=0.1,
            xref='paper',
            yref='paper',
            text='by @SamSweere, @AlexanderReisach',
            showarrow = False
        )]
    )
    fig.write_html("figures/country_sent.html")
    fig.show()



# import plotly.express as px

# # TODO figure out what exactly to use this for

# df = px.data.gapminder().query("year==2007")

# fig = px.choropleth(df, locations="iso_alpha",
#                     color="lifeExp", # lifeExp is a column of gapminder
#                     hover_name="country", # column to add to hover information
#                     color_continuous_scale=px.colors.sequential.Plasma)
# fig.show()