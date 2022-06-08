library(data.table)
library(ggplot2)

data <- read.table('../results/logs/validation_0_simple_without_dropout.txt', sep = '\t', header = TRUE)


ggplot(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = Cutoff))+   # change color to Fold to get match lines with folds
  geom_point(data = data[data$Fold==0,], size = 2)+
  geom_line(data = data[data$Fold==0,], size = 1)+
  geom_point(data = data[data$Fold==1,], size = 2)+
  geom_line(data = data[data$Fold==1,], size = 1)+
  geom_point(data = data[data$Fold==2,], size = 2)+
  geom_line(data = data[data$Fold==2,], size = 1)+
  geom_point(data = data[data$Fold==3,], size = 2)+
  geom_line(data = data[data$Fold==3,], size = 1)+
  geom_point(data = data[data$Fold==4,], size = 2)+
  geom_line(data = data[data$Fold==4,], size = 1)+
  scale_color_continuous(type = 'viridis')+
  geom_abline(slope = 1, intercept = 0)+
  xlim(0,100)+
  ggtitle('ROC Curve on Validation Set')+
  xlab('FPR')+
  ylab('TPR')

print(data[data$Prec == max(data$Prec) ,])
