library(ggplot2)
library(data.table)

data <- data.table(Epoch=c(seq(1, 14), seq(1, 14)),
                   Fold = c(rep('0', 14), rep('0',14)),
                   Set = c(rep('train', 14), rep('val', 14)),
                   
                   Loss = c(0.570713, 0.305772, 0.227960, 0.208673, 0.182775, 0.161829, 0.230814, 0.151067, 0.123855, 0.132804,
                            0.101202, 0.082469, 0.097137, 0.054364,  
                            
                            0.410208, 0.334800, 0.355585, 0.274127, 0.407790, 0.677067, 1.380870, 0.837761, 1.172343, 1.722758,
                            1.477623, 2.041316, 1.935029, 2.195568
                            ))

ggplot(data = data, mapping = aes(x = Epoch, y = Loss, linetype = Set))+
  geom_line(data = data[data$Set == 'train' & data$Fold == '0'], size = 1)+
  geom_line(data = data[data$Set == 'val'& data$Fold == '0'], mapping = aes(x = Epoch, y = Loss), size = 1)+
  ggtitle('Training Process of Model 2.2_0.3_lr_0.005 Fold 0')+
  geom_vline(xintercept = 4)
  


