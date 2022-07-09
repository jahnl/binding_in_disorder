library(ggplot2)
library(ggthemes)
library(data.table)

#color blind scales
# Fill
scale_fill_colorblind7 = function(.ColorList = 2L:9L, ...){
  scale_fill_discrete(..., type = colorblind_pal()(9)[.ColorList])
}
# Color
scale_color_colorblind7 = function(.ColorList = 2L:9L, ...){
  scale_color_discrete(..., type = colorblind_pal()(9)[.ColorList])
}


performance <- data.table(read.table("../results/logs/performance_assessment.tsv", header = TRUE, sep = "\t"))
performance <- cbind(performance, model_name = c("0 simple_CNN", "1 larger_CNN", "2 FNN", "2.1 oversampling", "2.2 dropout_0.2", "2.2 dropout_0.3", "2.2 dropout_0.3 post-processing", "random baseline",
                                            "3 disorder_only", "random baseline", rep("4 multilabel", 3), rep("4.1 oversampling", 3), rep("random baseline", 3)))
performance <- cbind(performance, class_name = c(rep("-", 10), rep(c("1: protein-binding", "2: nuc-binding", "3: 'other'-binding"), 3)))

#binary predictors, all metrics
ggplot(data = performance[1:8])+
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
ggplot(data = performance[1:8])+
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


