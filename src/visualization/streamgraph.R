# TODO create topic streamgraph

# Library
library(streamgraph)
library(tidyverse)
library(htmlwidgets)

df = read_csv("src/TopicAnalysis/topic_frequency.csv")

# Stream graph with a legend
get_sg = function(agg_type){
  print(typeof(agg_type))
  pp <- streamgraph(
    df,
    key="main_topic",
    value="sum",
    date="publication_date",
    height="300px",
    width="1000px"
  ) %>%
    sg_legend(show=TRUE, label="Topics: ") %>%
    sg_axis_x(1, "week", "%d-%m") %>%
    sg_fill_brewer("Spectral")
    # sg_fill_manual(rev(rainbow(10)))
  
  return(pp)
}

# save the widget
pp = get_sg("sum")
saveWidget(pp, file=paste0("TopicAnalysisStreamgraph", "_sum", ".html"))
pp


get_sg = function(agg_type){
  print(typeof(agg_type))
  pp <- streamgraph(
    df,
    key="main_topic",
    value="mean",
    date="publication_date",
    height="300px",
    width="1000px"
  ) %>%
    sg_legend(show=TRUE, label="Topics: ") %>%
    sg_axis_x(1, "week", "%d-%m") %>%
    sg_fill_brewer("Spectral")
    # sg_fill_manual(rev(rainbow(10)))
  
  return(pp)
}

pp = get_sg("mean")
saveWidget(pp, file=paste0("TopicAnalysisStreamgraph", "_mean", ".html"))
pp

