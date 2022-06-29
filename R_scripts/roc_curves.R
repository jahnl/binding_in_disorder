library(data.table)
library(ggplot2)

# binary classifier ROC curve
data_file <- '../results/logs/validation_2-1_new_oversampling.txt'
data <- read.table(data_file, sep = '\t', header = TRUE)
ggplot(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = Cutoff))+   # change color to Fold to match lines with folds
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
    ylim(0,100)+
    ggtitle('ROC Curve on Validation Set')+
    xlab('FPR')+
    ylab('TPR')
  
print(data[data$Prec == max(data$Prec) ,])



# multilabel classifier ROC curve
data_file <-  '../results/logs/validation_4-1_new_oversampling.txt'
data <- read.table(data_file, sep = "\t", header = TRUE)
#protein-binding
ggplot(mapping = aes(x = P_FP/(P_FP+P_TN)*100, y = P_Rec, color = Cutoff))+   # change color to Fold to match lines with folds
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
  ylim(0,100)+
  ggtitle('ROC Curve on Validation Set, Protein-Binding')+
  xlab('FPR')+
  ylab('TPR')
# Nuc-binding
ggplot(mapping = aes(x = N_FP/(N_FP+N_TN)*100, y = N_Rec, color = Cutoff))+   # change color to Fold to match lines with folds
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
  ylim(0,100)+
  ggtitle('ROC Curve on Validation Set, Nucleic Acid-Binding')+
  xlab('FPR')+
  ylab('TPR')
# Other-binding
ggplot(mapping = aes(x = O_FP/(O_FP+O_TN)*100, y = O_Rec, color = Cutoff))+   # change color to Fold to match lines with folds
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
  ylim(0,100)+
  ggtitle('ROC Curve on Validation Set, "Other"-Binding')+
  xlab('FPR')+
  ylab('TPR')
# comparison of classes
ggplot(data = data[data$Fold==2,])+   # change color to Fold to match lines with folds
  geom_point(mapping = aes(x = P_FP/(P_FP+P_TN)*100, y = P_Rec, color = 'protein-binding'), size = 2)+
  geom_line(mapping = aes(x = P_FP/(P_FP+P_TN)*100, y = P_Rec, color = 'protein-binding'), size = 1)+
  geom_point(mapping = aes(x = N_FP/(N_FP+N_TN)*100, y = N_Rec, color = 'nuc-binding'), size = 2)+
  geom_line(mapping = aes(x = N_FP/(N_FP+N_TN)*100, y = N_Rec, color = 'nuc-binding'), size = 1)+
  geom_point(mapping = aes(x = O_FP/(O_FP+O_TN)*100, y = O_Rec, color = 'other-binding'), size = 2)+
  geom_line(mapping = aes(x = O_FP/(O_FP+O_TN)*100, y = O_Rec, color = 'other-binding'), size = 1)+
  scale_color_discrete(type = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))+
  geom_abline(slope = 1, intercept = 0)+
  xlim(0,100)+
  ylim(0,100)+
  ggtitle('ROC Curve of Different Classes on Validation Set')+
  xlab('FPR')+
  ylab('TPR')


  
  
# compare ROC curves of best folds
data_0 <- read.table('../results/logs/validation_0_simple_without_dropout.txt', sep = '\t', header = TRUE)
data_0 <- data.table(data_0, model = '0: simple CNN')
data_1 <- read.table('../results/logs/validation_1_5_layers.txt', sep = '\t', header = TRUE)
data_1 <- data.table(data_1, model = '1: 5-layer-CNN')
data_2 <- read.table('../results/logs/validation_2_FNN.txt', sep = '\t', header = TRUE)
data_2 <- data.table(data_2, model = '2: FNN')
data_3 <- read.table('../results/logs/validation_3_d_only.txt', sep = '\t', header = TRUE)
data_3 <- data.table(data_3, model = '3: FNN - disorder only')
data_4 <- read.table('../results/logs/validation_2-1_new_oversampling.txt', sep = '\t', header = TRUE)
data_4 <- data.table(data_4, model = '2-1: FNN - residues oversampled')
data_all = rbindlist(list(data_0, data_1, data_2, data_3, data_4), fill = TRUE)

ggplot(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = model))+  
  geom_point(data = data_all[data_all$model == '0: simple CNN' & data_all$Fold==0,], size = 2)+
  geom_line(data = data_all[data_all$model == '0: simple CNN' & data_all$Fold==0,], size = 1)+
  geom_point(data = data_all[data_all$model == '1: 5-layer-CNN' & data_all$Fold==4,], size = 2)+
  geom_line(data = data_all[data_all$model == '1: 5-layer-CNN' & data_all$Fold==4,], size = 1)+
  geom_point(data = data_all[data_all$model == '2: FNN' & data_all$Fold==0,], size = 2)+
  geom_line(data = data_all[data_all$model == '2: FNN' & data_all$Fold==0,], size = 1)+
  geom_point(data = data_all[data_all$model == '3: FNN - disorder only' & data_all$Fold==4,], size = 2)+
  geom_line(data = data_all[data_all$model == '3: FNN - disorder only' & data_all$Fold==4,], size = 1)+
  geom_point(data = data_all[data_all$model == '2-1: FNN - residues oversampled' & data_all$Fold==0,], size = 2)+
  geom_line(data = data_all[data_all$model == '2-1: FNN - residues oversampled' & data_all$Fold==0,], size = 1)+
  
  scale_color_discrete(type = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))+
  geom_abline(slope = 1, intercept = 0)+
  xlim(0,100)+
  ggtitle('ROC Curves of Different Models on Validation Set')+
  xlab('FPR')+
  ylab('TPR')


