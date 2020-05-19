import plotly.express as px

# TODO figure out what exactly to use this for

df = px.data.gapminder().query("year==2007")

fig = px.choropleth(df, locations="iso_alpha",
                    color="lifeExp", # lifeExp is a column of gapminder
                    hover_name="country", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()