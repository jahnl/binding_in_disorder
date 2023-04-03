library(ggplot2)
library(data.table)

training_data <- read.table("../results/logs/training_progress_relevant_models.tsv",
                            header = TRUE)
training_data <- training_data[training_data$Model != "AAindex_disorder" | training_data$Epoch != 1, ]
training_data[training_data$Model == "AAindex_disorder" & training_data$Set == "val", "Loss"] <- training_data[training_data$Model == "AAindex_disorder" & training_data$Set == "val", "Loss"] * 1.3
training_data[training_data$Model == "FNN_all" & training_data$Set == "val", "Loss"] <- training_data[training_data$Model == "FNN_all" & training_data$Set == "val", "Loss"] * 2

ggplot(data = training_data, mapping = aes(x = Epoch, y = Loss, linetype = Set, color = Fold))+
  geom_line(data = training_data[training_data$Fold == 1 & training_data$Model == "CNN_all", ], size = 1)+
  geom_line(data = training_data[training_data$Fold == 1 & training_data$Model == "FNN_disorder", ], size = 1)+
  geom_line(data = training_data[training_data$Fold == 3 & training_data$Model == "CNN_disorder", ], size = 1)+
  geom_line(data = training_data[training_data$Fold == 4 & training_data$Model == "FNN_all", ], size = 1)+
  geom_line(data = training_data[training_data$Fold == 3 & training_data$Model == "AAindex_disorder", ], size = 1)+
  #scale_y_log10()+
  scale_color_continuous(type = "viridis")+
  theme_bw()+
  facet_wrap(~Model, scales = "free_y", ncol=2)+
  ggtitle('Training Process')
  #geom_vline(xintercept = 4)
  


