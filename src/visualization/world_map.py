import plotly.graph_objects as go
import pandas as pd
import country_converter as coco

def show_world_map(df):

    # Convert the country names to ISO3
    df["iso3"] = df['country'].apply(lambda x: coco.convert(names=[x], to="ISO3"))
    df["country_short"] = df['country'].apply(lambda x: coco.convert(names=[x], to='name_short'))
    # standard_names = coco.convert(names=some_names, to='ISO3')

    # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')

    fig = go.Figure(data=go.Choropleth(
        locations = df['iso3'],
        z = df['counts'],
        text = df['country_short'],
        colorscale = 'Reds',
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title = 'Referals',
    ))

    # fig = go.Figure(data=go.Choropleth(
    #     locations = df['CODE'],
    #     z = df['GDP (BILLIONS)'],
    #     text = df['COUNTRY'],
    #     colorscale = 'Blues',
    #     autocolorscale=False,
    #     reversescale=True,
    #     marker_line_color='darkgray',
    #     marker_line_width=0.5,
    #     colorbar_tickprefix = '$',
    #     colorbar_title = 'GDP<br>Billions US$',
    # ))

    fig.update_layout(
        title_text='Referals in Corona News',
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

    fig.show()




# import plotly.express as px

# # TODO figure out what exactly to use this for

# df = px.data.gapminder().query("year==2007")

# fig = px.choropleth(df, locations="iso_alpha",
#                     color="lifeExp", # lifeExp is a column of gapminder
#                     hover_name="country", # column to add to hover information
#                     color_continuous_scale=px.colors.sequential.Plasma)
# fig.show()