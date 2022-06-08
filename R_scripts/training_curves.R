library(ggplot2)
library(data.table)

data <- data.table(Epoch=c(seq(1, 22), seq(1,22),
                           seq(1, 13), seq(1,13)),
                   Fold = c(rep('0', 22), rep('0',22),
                            rep('1', 13), rep('1',13)),
                   Set = c(rep('train', 22), rep('val', 22),
                           rep('train', 13), rep('val', 13)),
                   
                   Loss = c(3.53, 0.39, 0.31, 0.31, 0.29, 0.32, 0.39, 0.28, 0.32, 0.40,
                            0.44, 0.27, 0.26, 0.27, 0.24, 0.46, 0.29, 0.27, 0.30, 0.27,
                            0.27, 0.25,
                            0.001049, 0.000950, 0.000902, 0.000868, 0.000944,
                            0.000860, 0.001124, 0.000953, 0.001373, 0.000913,
                            0.001014, 0.000819, 0.000991, 0.001074, 0.000968, 
                            0.001077, 0.001616, 0.001154, 0.001142, 0.001292,
                            0.001322, 0.001016,
                            
                            0.94, 0.56, 0.56, 0.35, 0.38, 0.44, 0.36, 0.37, 0.32, 0.70,
                            0.34, 0.34, 0.32,
                            0.002207, 0.001562, 0.001036, 0.001129, 0.001204, 
                            0.001536, 0.001224, 0.001395, 0.001846, 0.001337,
                            0.002026, 0.001251, 0.001462
                            ))

ggplot(data = data, mapping = aes(x = Epoch, y = Loss, linetype = Set, color = Fold))+
  geom_line(data = data[data$Set == 'train' & data$Fold == '0'], size = 1)+
  geom_line(data = data[data$Set == 'val'& data$Fold == '0'], mapping = aes(x = Epoch, y = 3400*Loss), size = 1)+
  #geom_line(data = data[data$Set == 'train'& data$Fold == '1'], size = 1)+
  #geom_line(data = data[data$Set == 'val'& data$Fold == '1'], mapping = aes(x = Epoch, y = Loss), size = 1)+
  ggtitle('Training Process of model 0')+
  geom_vline(xintercept = 12)
  #scale_color_discrete(type = 'viridis')



