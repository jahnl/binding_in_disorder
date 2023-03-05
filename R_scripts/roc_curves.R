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



# ROC curve, binary prediction, including curve within disorder
model_name <- 'mobidb_2_CNN_0'
data_file <- paste(paste('../results/logs/validation_', model_name, sep = ''), '.txt', sep = '')
data <- read.table(data_file, sep = '\t', header = TRUE)
data['D_Rec'] <- data['D_TP']/(data['D_TP']+data['D_FN'])
data['D_Prec'] <- data['D_TP']/(data['D_TP']+data['D_FP'])
ggplot(mapping = aes(x = D_FP/(D_FP+D_TN)*100, y = D_Rec*100, color = Fold, linetype = 'disorder only'))+   # change color to Fold to match lines with folds; Cutoff to have it colorful
  # general P-curve
#  geom_point(data = data[data$Fold==0,], size = 2)+
#  geom_line(data = data[data$Fold==0,], size = 1)+
#  geom_point(data = data[data$Fold==1,], size = 2)+
#  geom_line(data = data[data$Fold==1,], size = 1)+
#  geom_point(data = data[data$Fold==2,], size = 2)+
#  geom_line(data = data[data$Fold==2,], size = 1)+
#  geom_point(data = data[data$Fold==3,], size = 2)+
#  geom_line(data = data[data$Fold==3,], size = 1)+
#  geom_point(data = data[data$Fold==4,], size = 2)+
#  geom_line(data = data[data$Fold==4,], size = 1)+
  # D P-curve
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
  # aesthetics
  scale_color_continuous(type = 'viridis')+
  geom_abline(slope = 1, intercept = 0)+
  xlim(0,100)+
  ylim(0,100)+
  ggtitle(paste('ROC Curves on Validation Set - Model', model_name))+
  xlab('FPR')+
  ylab('TPR')

# to show cutoffs of each fold
for (fold in 0:4) {
  p <- ggplot(mapping = aes(x = D_FP/(D_FP+D_TN)*100, y = D_Rec*100, linetype = 'disorder only'), data = data[data$Fold==fold,])+
    # general P-curve
    #geom_point(data = data[data$Fold==best_fold,], size = 2)+
    #geom_line(data = data[data$Fold==best_fold,], size = 1)+
    #geom_text(mapping = aes(label = Cutoff), data = data[data$Fold==best_fold,], nudge_x = -4)+
    # D_N-curve
    geom_point(size = 2)+
    geom_line(size = 1)+
    geom_text(mapping = aes(label = Cutoff) , nudge_x = 6)+
    # aesthetics
    geom_abline(slope = 1, intercept = 0)+
    ggtitle(paste(paste('ROC Curves on Validation Set - Model', model_name), paste('Fold', toString(fold))))+
    xlab('FPR')+
    ylab('TPR')
  print(p)
}

#print(data[data$Prec == max(data$Prec) ,])






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
data_0 <- data.table(data_0, model = '0.: simple CNN')
data_1 <- read.table('../results/logs/validation_1_5_layers.txt', sep = '\t', header = TRUE)
data_1 <- data.table(data_1, model = '1.: larger CNN')
data_2 <- read.table('../results/logs/validation_2_FNN.txt', sep = '\t', header = TRUE)
data_2 <- data.table(data_2, model = '2.: FNN')
data_3 <- read.table('../results/logs/validation_3_d_only.txt', sep = '\t', header = TRUE)
data_3 <- data.table(data_3, model = '3.: Disorder only')
data_4 <- read.table('../results/logs/validation_2-1_new_oversampling.txt', sep = '\t', header = TRUE)
data_4 <- data.table(data_4, model = '2.1: Oversampling')
data_5 <- read.table('../results/logs/validation_2-2_dropout_0.3_new.txt', sep = '\t', header = TRUE)
data_5 <- data.table(data_5, model = '2.2: Dropout')
data_all = rbindlist(list(data_0, data_1, data_2, data_3, data_4, data_5), fill = TRUE)

ggplot(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = model))+  
  geom_point(data = data_all[data_all$model == '0.: simple CNN' & data_all$Fold==0,], size = 2)+
  geom_line(data = data_all[data_all$model == '0.: simple CNN' & data_all$Fold==0,], size = 1)+
  geom_point(data = data_all[data_all$model == '1.: larger CNN' & data_all$Fold==4,], size = 2)+
  geom_line(data = data_all[data_all$model == '1.: larger CNN' & data_all$Fold==4,], size = 1)+
  geom_point(data = data_all[data_all$model == '2.: FNN' & data_all$Fold==0,], size = 2)+
  geom_line(data = data_all[data_all$model == '2.: FNN' & data_all$Fold==0,], size = 1)+
  geom_point(data = data_all[data_all$model == '3.: Disorder only' & data_all$Fold==4,], size = 2)+
  geom_line(data = data_all[data_all$model == '3.: Disorder only' & data_all$Fold==4,], size = 1)+
  geom_point(data = data_all[data_all$model == '2.1: Oversampling' & data_all$Fold==0,], size = 2)+
  geom_line(data = data_all[data_all$model == '2.1: Oversampling' & data_all$Fold==0,], size = 1)+
  geom_point(data = data_all[data_all$model == '2.2: Dropout' & data_all$Fold==4,], size = 2)+
  geom_line(data = data_all[data_all$model == '2.2: Dropout' & data_all$Fold==4,], size = 1)+
  
  scale_color_discrete(type = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))+
  geom_abline(slope = 1, intercept = 0)+
  xlim(0,100)+
  ggtitle('ROC Curves of Different Models on Validation Set')+
  xlab('FPR')+
  ylab('TPR')





# mobidb and disorder val
data_d0 <- read.table('../results/logs/validation_mobidb_CNN_0.txt', sep = '\t', header = TRUE)
data_d0 <- data.table(data_d0, model = 'mobidb_CNN_0')
data_d1 <- read.table('../results/logs/validation_mobidb_D_CNN_0.txt', sep = '\t', header = TRUE)
data_d1 <- data.table(data_d1, model = 'mobidb_D_CNN_0')
data_d2 <- read.table('../results/logs/validation_mobidb_D_CNN_1.txt', sep = '\t', header = TRUE)
data_d2 <- data.table(data_d2, model = 'mobidb_D_CNN_1')
data_d3 <- read.table('../results/logs/validation_mobidb_D_FNN_0.txt', sep = '\t', header = TRUE)
data_d3 <- data.table(data_d3, model = 'mobidb_D_FNN_0')
data_d4 <- read.table('../results/logs/validation_mobidb_D_FNN_1.txt', sep = '\t', header = TRUE)
data_d4 <- data.table(data_d4, model = 'mobidb_D_FNN_1')
data_all = rbindlist(list(data_d0, data_d1, data_d2, data_d3, data_d4), fill = TRUE)

ggplot(mapping = aes(x = D_FN/(D_FN+D_TP)*100, y = D_NRec, color = model, linetype = 'disorder only,\nnegatives'))+  
  geom_point(data = data_all[data_all$model == 'mobidb_CNN_0' & data_all$Fold==2,], size = 2)+
  geom_line(data = data_all[data_all$model == 'mobidb_CNN_0' & data_all$Fold==2,], size = 1)+
  geom_point(data = data_all[data_all$model == 'mobidb_D_CNN_0' & data_all$Fold==2,], size = 2)+
  geom_line(data = data_all[data_all$model == 'mobidb_D_CNN_0' & data_all$Fold==2,], size = 1)+
  geom_point(data = data_all[data_all$model == 'mobidb_D_CNN_1' & data_all$Fold==2,], size = 2)+
  geom_line(data = data_all[data_all$model == 'mobidb_D_CNN_1' & data_all$Fold==2,], size = 1)+
  geom_point(data = data_all[data_all$model == 'mobidb_D_FNN_0' & data_all$Fold==4,], size = 2)+
  geom_line(data = data_all[data_all$model == 'mobidb_D_FNN_0' & data_all$Fold==4,], size = 1)+
  geom_point(data = data_all[data_all$model == 'mobidb_D_FNN_1' & data_all$Fold==2,], size = 2)+
  geom_line(data = data_all[data_all$model == 'mobidb_D_FNN_1' & data_all$Fold==2,], size = 1)+
  
  geom_point(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = model), data = data_all[data_all$model == 'mobidb_CNN_0' & data_all$Fold==2,], size = 2)+
  geom_line(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = model, linetype = 'whole dataset,\npositives'), data = data_all[data_all$model == 'mobidb_CNN_0' & data_all$Fold==2,], size = 1)+
  #geom_point(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = model), data = data_all[data_all$model == 'mobidb_D_CNN_0' & data_all$Fold==2,], size = 2)+
  #geom_line(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = model, linetype = 'whole dataset,\npositives'), data = data_all[data_all$model == 'mobidb_D_CNN_0' & data_all$Fold==2,], size = 1)+
  #geom_point(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = model), data = data_all[data_all$model == 'mobidb_D_CNN_1' & data_all$Fold==2,], size = 2)+
  #geom_line(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = model, linetype = 'whole dataset,\npositives'), data = data_all[data_all$model == 'mobidb_D_CNN_1' & data_all$Fold==2,], size = 1)+
  #geom_point(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = model), data = data_all[data_all$model == 'mobidb_D_FNN_0' & data_all$Fold==2,], size = 2)+
  #geom_line(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = model, linetype = 'whole dataset,\npositives'), data = data_all[data_all$model == 'mobidb_D_FNN_0' & data_all$Fold==2,], size = 1)+
  #geom_point(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = model), data = data_all[data_all$model == 'mobidb_D_FNN_1' & data_all$Fold==2,], size = 2)+
  #geom_line(mapping = aes(x = FP/(FP+TN)*100, y = Rec, color = model, linetype = 'whole dataset,\npositives'), data = data_all[data_all$model == 'mobidb_D_FNN_1' & data_all$Fold==2,], size = 1)+
  
  
  scale_color_discrete(type = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))+
  geom_abline(slope = 1, intercept = 0)+
  xlim(0,100)+
  ggtitle('ROC Curves of Different Models on Validation Set')+
  xlab('FPR')+
  ylab('TPR')


# dropout val
data_d0 <- read.table('../results/logs/validation_mobidb_D_CNN_0.txt', sep = '\t', header = TRUE)
data_d0 <- data.table(data_d0, model = 'mobidb_D_CNN_0')
data_d1 <- read.table('../results/logs/validation_mobidb_D_CNN_0_d2.txt', sep = '\t', header = TRUE)
data_d1 <- data.table(data_d1, model = 'mobidb_D_CNN_0_d2')
data_d2 <- read.table('../results/logs/validation_mobidb_D_CNN_0_d3.txt', sep = '\t', header = TRUE)
data_d2 <- data.table(data_d2, model = 'mobidb_D_CNN_0_d3')
data_all = rbindlist(list(data_d0, data_d1, data_d2), fill = TRUE)

ggplot(mapping = aes(x = D_FN/(D_FN+D_TP)*100, y = D_NRec, color = model, linetype = 'disorder only,\nnegatives'))+  
  geom_point(data = data_all[data_all$model == 'mobidb_D_CNN_0' & data_all$Fold==2,], size = 2)+
  geom_line(data = data_all[data_all$model == 'mobidb_D_CNN_0' & data_all$Fold==2,], size = 1)+
  geom_point(data = data_all[data_all$model == 'mobidb_D_CNN_0_d2' & data_all$Fold==4,], size = 2)+
  geom_line(data = data_all[data_all$model == 'mobidb_D_CNN_0_d2' & data_all$Fold==4,], size = 1)+
  geom_point(data = data_all[data_all$model == 'mobidb_D_CNN_0_d3' & data_all$Fold==3,], size = 2)+
  geom_line(data = data_all[data_all$model == 'mobidb_D_CNN_0_d3' & data_all$Fold==3,], size = 1)+
  
  
  scale_color_discrete(type = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))+
  geom_abline(slope = 1, intercept = 0)+
  xlim(0,100)+
  ggtitle('ROC Curves of Different Models on Validation Set')+
  xlab('FPR')+
  ylab('TPR')




