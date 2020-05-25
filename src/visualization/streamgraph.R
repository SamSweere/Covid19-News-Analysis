# TODO create topic streamgraph

# Library
library(streamgraph)
library(tidyverse)
library(htmlwidgets)

df = read_csv("src/topic_frequency.csv")

# Stream graph with a legend
pp <- streamgraph(
  df,
  key="main_topic",
  value="sum",
  date="publication_date",
  height="300px",
  width="1000px"
) %>%
  sg_legend(show=TRUE, label="Topics: ") %>%
  sg_axis_x(1, "day", "%d-%m")
  
pp

# TODO get this to work!
# save the widget
saveWidget(pp, file="TopicAnalysisStreamgraph.html")
