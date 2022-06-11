library(ggplot2)
library(data.table)

data <- data.table(Epoch=c(seq(1, 8), seq(1,8)),
                   Fold = c(rep('4', 8), rep('4',8)),
                   Set = c(rep('train', 8), rep('val', 8)),
                   
                   Loss = c(0.001301, 0.000669, 0.000519, 0.000420, 0.000335, 0.000263, 0.000202, 0.000157,
                            0.000778, 0.000754, 0.000753, 0.000759, 0.000765, 0.000797, 0.000936, 0.001091))

ggplot(data = data, mapping = aes(x = Epoch, y = Loss, linetype = Set, color = Fold))+
  geom_line(data = data[data$Set == 'train' & data$Fold == '4'], size = 1)+
  geom_line(data = data[data$Set == 'val'& data$Fold == '4'], mapping = aes(x = Epoch, y = Loss), size = 1)+
  ggtitle('Training Process of model 1')+
  geom_vline(xintercept = 3)
  


