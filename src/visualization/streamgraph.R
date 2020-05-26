# TODO create topic streamgraph

# Library
library(streamgraph)
library(tidyverse)
library(htmlwidgets)

# TODO load all the csvs in a folder and rbind them
base_path = "data/0TopicAnalysis/ta_run_d_26_05_t_15_41/"
filenames=list.files(path=base_path, pattern="*.csv")
df = data.frame()
for (i in filenames){
  day_df = read_csv(paste0(base_path, i))
  df = rbind(df, day_df)
}

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
    sg_fill_brewer("Spectral") %>%
    sg_title(sg, "Covid-19 Topic River") 
    # sg_fill_manual(rev(rainbow(10)))
  
  return(pp)
}

# save the widget
pp = get_sg("sum")
saveWidget(pp, file=paste0("TopicAnalysisStreamgraph", "_sum", ".html"))
pp


# annotate_graph = function(sg){
#   pp = sg %>%
#     sg_annotate(label="bank_market_rate", x=as.Date("2020-03-22"), y=0.03, color="#ffffff", size=18)%>%
#     sg_annotate(label="case_china_death", x=as.Date("2020-03-22"), y=0.13, color="#ffffff", size=18)%>%
#     sg_annotate(label="flight_airline_travel", x=as.Date("2020-03-22"), y=0.23, color="#ffffff", size=18)%>%
#     sg_annotate(label="hedge_fund_stock", x=as.Date("2020-03-22"), y=0.33, color="#ffffff", size=18)%>%
#     sg_annotate(label="league_game_player", x=as.Date("2020-03-22"), y=0.43, color="#ffffff", size=18)%>%
#     sg_annotate(label="oil_price_barrel", x=as.Date("2020-03-22"), y=0.53, color="#ffffff", size=18)%>%
#     sg_annotate(label="people_home_work", x=as.Date("2020-03-22"), y=0.63, color="#ffffff", size=18)%>%
#     sg_annotate(label="ship_cruise_passenger", x=as.Date("2020-03-22"), y=0.73, color="#ffffff", size=18)%>%
#     sg_annotate(label="test_hospital_patient", x=as.Date("2020-03-22"), y=0.83, color="#ffffff", size=18)%>%
#     sg_annotate(label="trump_biden_president", x=as.Date("2020-03-22"), y=0.93, color="#ffffff", size=18)
#   return(pp)
# }

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
    sg_axis_x(1, "week", "%d-%m") %>%
    # sg_fill_tableau("greenorange12") %>%
    sg_legend(show=TRUE, label="Topics: ")
  
  return(pp)
}

pp = get_sg("mean")
# pp = annotate_graph(pp)
saveWidget(pp, file=paste0("TopicAnalysisStreamgraph", "_mean", ".html"))
pp


library(ggplot)
library(ggTimeSeries)

plt = ggplot(df, aes(x = publication_date, y = sum, group = main_topic, fill = main_topic)) +
   stat_steamgraph() +
   scale_fill_brewer(palette="Spectral") + 
   theme_bw()
ggsave("src/figures/gg_stream_sum.png", plt, width=10, height=5, units="in", device="png")


plt = ggplot(df, aes(x = publication_date, y = mean, group = main_topic, fill = main_topic)) +
   stat_steamgraph() +
   scale_fill_brewer(palette="Spectral") + 
   theme_bw()
ggsave("src/figures/gg_stream_mean.png", plt, width=10, height=5, units="in", device="png")

