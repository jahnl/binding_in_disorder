library(ggplot2)
library(ggthemes)
library(data.table)

#color blind scales
# Fill
scale_fill_colorblind7 = function(.ColorList = 2L:9L, ...){
  scale_fill_discrete(..., type = colorblind_pal()(9)[.ColorList])
}
scale_fill_colorblind10 = function(.ColorList = 2L:13L, ...){
  scale_fill_discrete(..., type = colorblind_pal()(10)[.ColorList])
}
# Color
scale_color_colorblind7 = function(.ColorList = 2L:9L, ...){
  scale_color_discrete(..., type = colorblind_pal()(9)[.ColorList])
}
scale_color_colorblind10 = function(.ColorList = 2L:13L, ...){
  scale_fill_discrete(..., type = colorblind_pal()(10)[.ColorList])
}

################ disprot models ##############

performance <- data.table(read.table("../results/logs/performance_assessment.tsv", header = TRUE, sep = "\t"))
performance <- cbind(performance, model_name = c("0 simple_CNN", "1 larger_CNN", "2 FNN", "2.1 oversampling", "2.2 dropout_0.2", "2.2 dropout", "2.3 post-processing", "random baseline",
                                            "3 disorder_only", "random baseline", rep("4 multilabel", 3), rep("4.1 oversampling", 3), rep("random baseline", 3)))
performance <- cbind(performance, class_name = c(rep("-", 10), rep(c("1: protein-binding", "2: nuc-binding", "3: 'other'-binding"), 3)))

#binary predictors, all metrics
ggplot(data = rbind(performance[1:4], performance[6:8]))+
  geom_bar(mapping = aes(x = "Precision", y = Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = Precision - SE_Precision, ymax = Precision + SE_Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Recall", y = Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = Recall - SE_Recall, ymax = Recall + SE_Recall), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = Balanced.Acc. - SE_Balanced.Acc., ymax = Balanced.Acc. + SE_Balanced.Acc.), position = position_dodge2())+
  geom_bar(mapping = aes(x = "F1", y = F1, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "F1", ymin = F1 - SE_F1, ymax = F1 + SE_F1), position = position_dodge2())+
  geom_bar(mapping = aes(x = "MCC", y = MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = MCC - SE_MCC, ymax = MCC + SE_MCC), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Precision", y = Neg_Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Precision", ymin = Neg_Precision - SE_Neg_Precision, ymax = Neg_Precision + SE_Neg_Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Recall", y = Neg_Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Recall", ymin = Neg_Recall - SE_Neg_Recall, ymax = Neg_Recall + SE_Neg_Recall), position = position_dodge2())+
  ylab("Value")+
  xlab("")+
  ggtitle("Performance of the Binary Predictors")+
  scale_fill_colorblind7()+
  theme_bw()

# binary predictors, selection + text
ggplot(data = rbind(performance[1:4], performance[6:8]))+
  geom_bar(mapping = aes(x = "Precision", y = Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = Precision - SE_Precision, ymax = Precision + SE_Precision), position = position_dodge2())+
  geom_text(aes(x = "Precision", y = -0.02, label = round(Precision*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Recall", y = Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = Recall - SE_Recall, ymax = Recall + SE_Recall), position = position_dodge2())+
  geom_text(aes(x = "Recall", y = -0.02, label = round(Recall*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = Balanced.Acc. - SE_Balanced.Acc., ymax = Balanced.Acc. + SE_Balanced.Acc.), position = position_dodge2())+
  geom_text(aes(x = "Balanced.Acc.", y = -0.02, label = round(Balanced.Acc.*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "MCC", y = MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = MCC - SE_MCC, ymax = MCC + SE_MCC), position = position_dodge2())+
  geom_text(aes(x = "MCC", y = -0.02, label = round(MCC*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  ylab("Value")+
  xlab("")+
  ggtitle("Performance of the Binary Predictors")+
  scale_fill_colorblind7()+
  theme_bw()

#disorder only predictor, all metrics
ggplot(data = performance[9:10])+
  geom_bar(mapping = aes(x = "Precision", y = Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = Precision - SE_Precision, ymax = Precision + SE_Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Recall", y = Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = Recall - SE_Recall, ymax = Recall + SE_Recall), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = Balanced.Acc. - SE_Balanced.Acc., ymax = Balanced.Acc. + SE_Balanced.Acc.), position = position_dodge2())+
  geom_bar(mapping = aes(x = "F1", y = F1, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "F1", ymin = F1 - SE_F1, ymax = F1 + SE_F1), position = position_dodge2())+
  geom_bar(mapping = aes(x = "MCC", y = MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = MCC - SE_MCC, ymax = MCC + SE_MCC), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Precision", y = Neg_Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Precision", ymin = Neg_Precision - SE_Neg_Precision, ymax = Neg_Precision + SE_Neg_Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Recall", y = Neg_Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Recall", ymin = Neg_Recall - SE_Neg_Recall, ymax = Neg_Recall + SE_Neg_Recall), position = position_dodge2())+
  ylab("Value")+
  xlab("")+
  ylim(-0.008, 1)+
  ggtitle("Performance of the Binary Predictor Trained on Disordered Residues Only")+
  scale_fill_colorblind7()+
  theme_bw()

# disorder only predictor, selection + text
ggplot(data = performance[9:10])+
  geom_bar(mapping = aes(x = "Precision", y = Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = Precision - SE_Precision, ymax = Precision + SE_Precision), position = position_dodge2())+
  geom_text(aes(x = "Precision", y = -0.02, label = round(Precision*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Recall", y = Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = Recall - SE_Recall, ymax = Recall + SE_Recall), position = position_dodge2())+
  geom_text(aes(x = "Recall", y = -0.02, label = round(Recall*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = Balanced.Acc. - SE_Balanced.Acc., ymax = Balanced.Acc. + SE_Balanced.Acc.), position = position_dodge2())+
  geom_text(aes(x = "Balanced.Acc.", y = -0.02, label = round(Balanced.Acc.*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "MCC", y = MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = MCC - SE_MCC, ymax = MCC + SE_MCC), position = position_dodge2())+
  geom_text(aes(x = "MCC", y = -0.02, label = round(MCC*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  ylab("Value")+
  xlab("")+
  ylim(-0.02, 1)+
  ggtitle("Performance of the Binary Predictor Trained on Disordered Residues Only")+
  scale_fill_colorblind7()+
  theme_bw()

#multilabel predictors, all metrics
ggplot(data = performance[11:19])+
  geom_bar(mapping = aes(x = "Precision", y = Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = Precision - SE_Precision, ymax = Precision + SE_Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Recall", y = Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = Recall - SE_Recall, ymax = Recall + SE_Recall), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = Balanced.Acc. - SE_Balanced.Acc., ymax = Balanced.Acc. + SE_Balanced.Acc.), position = position_dodge2())+
  geom_bar(mapping = aes(x = "F1", y = F1, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "F1", ymin = F1 - SE_F1, ymax = F1 + SE_F1), position = position_dodge2())+
  geom_bar(mapping = aes(x = "MCC", y = MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = MCC - SE_MCC, ymax = MCC + SE_MCC), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Precision", y = Neg_Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Precision", ymin = Neg_Precision - SE_Neg_Precision, ymax = Neg_Precision + SE_Neg_Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Recall", y = Neg_Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Recall", ymin = Neg_Recall - SE_Neg_Recall, ymax = Neg_Recall + SE_Neg_Recall), position = position_dodge2())+
  ylab("Value")+
  xlab("")+
  #ylim(-0.01, 1)+
  ggtitle("Performance of the Multilabel Predictors")+
  scale_fill_colorblind7()+
  theme_bw()+
  facet_grid(. ~ class_name)

# multilabel predictors, selection + text
ggplot(data = performance[11:19])+
  geom_bar(mapping = aes(x = "Precision", y = Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = Precision - SE_Precision, ymax = Precision + SE_Precision), position = position_dodge2())+
  geom_text(aes(x = "Precision", y = -0.04, label = round(Precision*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Recall", y = Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = Recall - SE_Recall, ymax = Recall + SE_Recall), position = position_dodge2())+
  geom_text(aes(x = "Recall", y = -0.04, label = round(Recall*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = Balanced.Acc. - SE_Balanced.Acc., ymax = Balanced.Acc. + SE_Balanced.Acc.), position = position_dodge2())+
  geom_text(aes(x = "Balanced.Acc.", y = -0.04, label = round(Balanced.Acc.*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "MCC", y = MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = MCC - SE_MCC, ymax = MCC + SE_MCC), position = position_dodge2())+
  geom_text(aes(x = "MCC", y = -0.04, label = round(MCC*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  ylab("Value")+
  xlab("")+
  ggtitle("Performance of the Multilabel Predictors")+
  scale_fill_colorblind7()+
  theme_bw()+
  facet_grid(. ~ class_name)


# test set + bindEmbed

performance_t <- data.table(read.table("../results/logs/performance_assessment_test.tsv", header = TRUE, sep = "\t"))
performance_t <- cbind(performance_t, model_name = c("0 simple_CNN", "1 larger_CNN", "2 FNN", "2.1 oversampling", "2.2 dropout_0.2", "2.2 dropout_0.3", "2.3 post-processing", "random baseline",
                                                 "3 disorder_only", "random baseline", rep("4 multilabel", 3), rep("4.1 oversampling", 3), rep("random baseline", 3), "bindEmbed21DL"))

ggplot(data = rbind(performance_t[7], performance_t[20], performance_t[8]))+
  geom_bar(mapping = aes(x = "Precision", y = Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = Precision - SE_Precision, ymax = Precision + SE_Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Recall", y = Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = Recall - SE_Recall, ymax = Recall + SE_Recall), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = Balanced.Acc. - SE_Balanced.Acc., ymax = Balanced.Acc. + SE_Balanced.Acc.), position = position_dodge2())+
  geom_bar(mapping = aes(x = "F1", y = F1, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "F1", ymin = F1 - SE_F1, ymax = F1 + SE_F1), position = position_dodge2())+
  geom_bar(mapping = aes(x = "MCC", y = MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = MCC - SE_MCC, ymax = MCC + SE_MCC), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Precision", y = Neg_Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Precision", ymin = Neg_Precision - SE_Neg_Precision, ymax = Neg_Precision + SE_Neg_Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Recall", y = Neg_Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Recall", ymin = Neg_Recall - SE_Neg_Recall, ymax = Neg_Recall + SE_Neg_Recall), position = position_dodge2())+
  ylab("Value")+
  xlab("")+
  ggtitle("Performance on Test Set")+
  scale_fill_colorblind7()+
  theme_bw()
# selection
ggplot(data = rbind(performance_t[7], performance_t[20], performance_t[8]))+
  geom_bar(mapping = aes(x = "Precision", y = Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = Precision - SE_Precision, ymax = Precision + SE_Precision), position = position_dodge2())+
  geom_text(aes(x = "Precision", y = -0.02, label = round(Precision*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Recall", y = Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = Recall - SE_Recall, ymax = Recall + SE_Recall), position = position_dodge2())+
  geom_text(aes(x = "Recall", y = -0.02, label = round(Recall*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = Balanced.Acc. - SE_Balanced.Acc., ymax = Balanced.Acc. + SE_Balanced.Acc.), position = position_dodge2())+
  geom_text(aes(x = "Balanced.Acc.", y = -0.02, label = round(Balanced.Acc.*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "MCC", y = MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = MCC - SE_MCC, ymax = MCC + SE_MCC), position = position_dodge2())+
  geom_text(aes(x = "MCC", y = -0.02, label = round(MCC*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  ylab("Value")+
  xlab("")+
  ylim(-0.05, 1)+
  ggtitle("Performance on Test Set")+
  scale_fill_colorblind7()+
  theme_bw()


######### mobidb models ##############

performance <- data.table(read.table("../results/logs/performance_assessment_mobidb_2.tsv", header = TRUE, sep = "\t"))
performance <- cbind(performance, model_name = c("00 mobidb_2_CNN_0", "1. CNN_all", "02 mobidb_2_CNN_2",
                                                 "03 mobidb_2_FNN_0", "04 mobidb_2_FNN_1", "05 mobidb_2_FNN_2", "06 mobidb_2_FNN_3", "07 mobidb_2_FNN_4", "3. FNN_all",
                                                 "09 mobidb_2_D_CNN_0", "10 mobidb_2_D_CNN_1", "2. CNN_disorder", 
                                                 "12 mobidb_2_D_FNN_0", "13 mobidb_2_D_FNN_1", "14 mobidb_2_D_FNN_2", "15 mobidb_2_D_FNN_3", "4. FNN_disorder",
                                                 "17 AAindex_baseline", "5. AAindex_disorder",
                                                 "random_baseline", "random_disorder"))
performance <- performance[c(1:8, 12, 10:11, 9, 13:21)] # make CNN_disorder (12) and FNN_all (9) switch places due to plotting problems with error-bars

performance_test <- data.table(read.table("../results/logs/performance_assessment_test_mobidb_2.tsv", header = TRUE, sep = "\t"))
performance_test <- cbind(performance_test, model_name = c("00 mobidb_2_CNN_0", "1. CNN_all", "02 mobidb_2_CNN_2",
                                                           "03 mobidb_2_FNN_0", "04 mobidb_2_FNN_1", "05 mobidb_2_FNN_2", "06 mobidb_2_FNN_3", "07 mobidb_2_FNN_4", "3. FNN_all",
                                                           "09 mobidb_2_D_CNN_0", "10 mobidb_2_D_CNN_1", "2. CNN_disorder", 
                                                           "12 mobidb_2_D_FNN_0", "13 mobidb_2_D_FNN_1", "14 mobidb_2_D_FNN_2", "15 mobidb_2_D_FNN_3", "4. FNN_disorder",
                                                           "17 AAindex_baseline", "5. AAindex_disorder",
                                                           "random_baseline", "random_disorder",
                                                           "ref: ANCHOR2", "ref: DeepDISOBind"))
performance_test <- performance_test[c(1:8, 12, 10:11, 9, 13:23)]

#### whole protein prediction ######
ggplot(data = rbind(performance[1:9], performance[18]))+
  geom_bar(mapping = aes(x = "Precision", y = Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = Precision - SE_Precision, ymax = Precision + SE_Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Recall", y = Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = Recall - SE_Recall, ymax = Recall + SE_Recall), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = Balanced.Acc. - SE_Balanced.Acc., ymax = Balanced.Acc. + SE_Balanced.Acc.), position = position_dodge2())+
  geom_bar(mapping = aes(x = "F1", y = F1, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "F1", ymin = F1 - SE_F1, ymax = F1 + SE_F1), position = position_dodge2())+
  geom_bar(mapping = aes(x = "MCC", y = MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = MCC - SE_MCC, ymax = MCC + SE_MCC), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Precision", y = Neg_Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Precision", ymin = Neg_Precision - SE_Neg_Precision, ymax = Neg_Precision + SE_Neg_Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Recall", y = Neg_Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Recall", ymin = Neg_Recall - SE_Neg_Recall, ymax = Neg_Recall + SE_Neg_Recall), position = position_dodge2())+
  ylab("Value")+
  xlab("")+
  ggtitle("Performance of MobiDB Predictors on the Whole Protein")+
  scale_fill_colorblind7()+
  theme_bw()

# whole protein prediction, selection + text
ggplot(data = rbind(performance[1:9], performance[18]))+
  geom_bar(mapping = aes(x = "Precision", y = Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = Precision - SE_Precision, ymax = Precision + SE_Precision), position = position_dodge2())+
  geom_text(aes(x = "Precision", y = -0.02, label = round(Precision*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Recall", y = Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = Recall - SE_Recall, ymax = Recall + SE_Recall), position = position_dodge2())+
  geom_text(aes(x = "Recall", y = -0.02, label = round(Recall*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = Balanced.Acc. - SE_Balanced.Acc., ymax = Balanced.Acc. + SE_Balanced.Acc.), position = position_dodge2())+
  geom_text(aes(x = "Balanced.Acc.", y = -0.02, label = round(Balanced.Acc.*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "MCC", y = MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = MCC - SE_MCC, ymax = MCC + SE_MCC), position = position_dodge2())+
  geom_text(aes(x = "MCC", y = -0.02, label = round(MCC*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  ylab("Value")+
  xlab("")+
  ggtitle("Performance of MobiDB Predictors on the Whole Protein")+
  scale_fill_colorblind7()+
  theme_bw()


##### disorder only prediction, part X ####
best_of_val = rbind(performance[2], performance[9], performance[12], performance[17], performance[19], performance[21])
ggplot(data = best_of_val)+
  geom_bar(mapping = aes(x = "Precision", y = D.Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = D.Precision - SE_D.Precision, ymax = D.Precision + SE_D.Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Recall", y = D.Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = D.Recall - SE_D.Recall, ymax = D.Recall + SE_D.Recall), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = D.Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = D.Balanced.Acc. - SE_D.Balanced.Acc., ymax = D.Balanced.Acc. + SE_D.Balanced.Acc.), position = position_dodge2())+
  geom_bar(mapping = aes(x = "F1", y = D.F1, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "F1", ymin = D.F1 - SE_D.F1, ymax = D.F1 + SE_D.F1), position = position_dodge2())+
  geom_bar(mapping = aes(x = "MCC", y = D.MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = D.MCC - SE_D.MCC, ymax = D.MCC + SE_D.MCC), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Precision", y = D.Neg_Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Precision", ymin = D.Neg_Precision - SE_D.Neg_Precision, ymax = D.Neg_Precision + SE_D.Neg_Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Recall", y = D.Neg_Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Recall", ymin = D.Neg_Recall - SE_D.Neg_Recall, ymax = D.Neg_Recall + SE_D.Neg_Recall), position = position_dodge2())+
  ylab("Value")+
  xlab("")+
  ggtitle("Performance of MobiDB Predictors in Disordered Regions Only, Best Of")+
  scale_fill_colorblind10()+
  theme_bw()

# disorder only prediction, part X, selection + text
ggplot(data = best_of_val)+
  geom_bar(mapping = aes(x = "Precision", y = D.Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = D.Precision - SE_D.Precision, ymax = D.Precision + SE_D.Precision), position = position_dodge2())+
  geom_text(aes(x = "Precision", y = -0.02, label = round(D.Precision*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Recall", y = D.Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = D.Recall - SE_D.Recall, ymax = D.Recall + SE_D.Recall), position = position_dodge2())+
  geom_text(aes(x = "Recall", y = -0.02, label = round(D.Recall*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = D.Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = D.Balanced.Acc. - SE_D.Balanced.Acc., ymax = D.Balanced.Acc. + SE_D.Balanced.Acc.), position = position_dodge2())+
  geom_text(aes(x = "Balanced.Acc.", y = -0.02, label = round(D.Balanced.Acc.*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "MCC", y = D.MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = D.MCC - SE_D.MCC, ymax = D.MCC + SE_D.MCC), position = position_dodge2())+
  geom_text(aes(x = "MCC", y = -0.02, label = round(D.MCC*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  xlab("")+
  labs(fill='Model name')+
  scale_y_continuous("Performance", sec.axis = sec_axis(~ .))+
  ggtitle("Performance on the Validation Set")+
  scale_fill_colorblind10()+
  theme_bw()

# disorder only prediction TEST
best_of_test = rbind(performance_test[2], performance_test[9], performance_test[12], performance_test[17], performance_test[19], performance_test[21:23])
ggplot(data = best_of_test)+
  geom_bar(mapping = aes(x = "Precision", y = D.Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = D.Precision - SE_D.Precision, ymax = D.Precision + SE_D.Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Recall", y = D.Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = D.Recall - SE_D.Recall, ymax = D.Recall + SE_D.Recall), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = D.Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = D.Balanced.Acc. - SE_D.Balanced.Acc., ymax = D.Balanced.Acc. + SE_D.Balanced.Acc.), position = position_dodge2())+
  geom_bar(mapping = aes(x = "F1", y = D.F1, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "F1", ymin = D.F1 - SE_D.F1, ymax = D.F1 + SE_D.F1), position = position_dodge2())+
  geom_bar(mapping = aes(x = "MCC", y = D.MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = D.MCC - SE_D.MCC, ymax = D.MCC + SE_D.MCC), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Precision", y = D.Neg_Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Precision", ymin = D.Neg_Precision - SE_D.Neg_Precision, ymax = D.Neg_Precision + SE_D.Neg_Precision), position = position_dodge2())+
  geom_bar(mapping = aes(x = "Neg_Recall", y = D.Neg_Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Neg_Recall", ymin = D.Neg_Recall - SE_D.Neg_Recall, ymax = D.Neg_Recall + SE_D.Neg_Recall), position = position_dodge2())+
  ylab("Value")+
  xlab("")+
  ggtitle("Performance of Different Predictors in Disordered Regions On the Test Set")+
  scale_fill_colorblind10()+
  theme_bw()

# disorder only prediction TEST, selection
ggplot(data = best_of_test)+
  geom_bar(mapping = aes(x = "Precision", y = D.Precision, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Precision", ymin = D.Precision - SE_D.Precision, ymax = D.Precision + SE_D.Precision), position = position_dodge2())+
  geom_text(aes(x = "Precision", y = -0.02, label = round(D.Precision*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Recall", y = D.Recall, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Recall", ymin = D.Recall - SE_D.Recall, ymax = D.Recall + SE_D.Recall), position = position_dodge2())+
  geom_text(aes(x = "Recall", y = -0.02, label = round(D.Recall*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = D.Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = D.Balanced.Acc. - SE_D.Balanced.Acc., ymax = D.Balanced.Acc. + SE_D.Balanced.Acc.), position = position_dodge2())+
  geom_text(aes(x = "Balanced.Acc.", y = -0.02, label = round(D.Balanced.Acc.*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  geom_bar(mapping = aes(x = "MCC", y = D.MCC, fill = model_name), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = "MCC", ymin = D.MCC - SE_D.MCC, ymax = D.MCC + SE_D.MCC), position = position_dodge2())+
  geom_text(aes(x = "MCC", y = -0.02, label = round(D.MCC*100, 0)), position = position_dodge2(width = 0.9), size = 3.3)+
  scale_y_continuous("Performance", sec.axis = sec_axis(~ .))+
  xlab("")+
  labs(fill='Model name')+
  ggtitle("Performance on the Test Set")+
  scale_fill_colorblind10()+
  theme_bw()



# prediction: selection,  val and Test difference
val_perf <- data.table(best_of_val)
val_perf[] <- lapply(val_perf, function(x) as.numeric(x) )
val_perf$model_name <- c("1. CNN_all", "2. CNN_disorder", "3. FNN_all", "4. FNN_disorder", "5. AAindex_disorder", "random_disorder")
val_perf <- rbind(val_perf, NA, NA, fill=TRUE)
test_perf <- data.table(best_of_test)
test_perf$model_name[7:8] <- c("ref: ANCHOR2", "ref: DeepDISOBind")

ggplot(data = test_perf)+
  geom_bar(mapping = aes(x = "Balanced.Acc.", y = D.Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2(), alpha = 0.7)+
  geom_errorbar(mapping = aes(x = "Balanced.Acc.", ymin = D.Balanced.Acc. - SE_D.Balanced.Acc., ymax = D.Balanced.Acc. + SE_D.Balanced.Acc.), position = position_dodge2())+
  geom_bar(mapping = aes(x = "MCC", y = D.MCC, fill = model_name), stat = "identity", position = position_dodge2(), alpha = 0.7)+
  geom_errorbar(mapping = aes(x = "MCC", ymin = D.MCC - SE_D.MCC, ymax = D.MCC + SE_D.MCC), position = position_dodge2())+
  # val
  geom_bar(data = val_perf, mapping = aes(x = "Balanced.Acc.", y = D.Balanced.Acc., fill = model_name), stat = "identity", position = position_dodge2(), alpha = 0.7)+
  geom_errorbar(data = val_perf, mapping = aes(x = "Balanced.Acc.", ymin = D.Balanced.Acc. - SE_D.Balanced.Acc., ymax = D.Balanced.Acc. + SE_D.Balanced.Acc.), position = position_dodge2())+
  geom_bar(data = val_perf, mapping = aes(x = "MCC", y = D.MCC, fill = model_name), stat = "identity", position = position_dodge2(), alpha = 0.7)+
  geom_errorbar(data = val_perf, mapping = aes(x = "MCC", ymin = D.MCC - SE_D.MCC, ymax = D.MCC + SE_D.MCC), position = position_dodge2())+
  ylab("Value")+
  xlab("")+
  ggtitle("Difference of the Performance Between Validation and Test Set")+
  scale_fill_colorblind10()+
  theme_bw()

