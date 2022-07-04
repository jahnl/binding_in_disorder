library(ggplot2)
library(data.table)

performance <- data.table(read.table("../results/logs/performance_assessment.tsv", header = TRUE, sep = "\t"))
performance <- cbind(performance, model_name = c("0 simple_CNN", "1 larger_CNN", "2 FNN", "2.1 oversampling", "2.2 dropout_0.2", "2.2 dropout_0.3",
                                            "3 d_only", rep("4 multilabel", 3), rep("4.1 oversampling", 3)))
# performance <- rbind(performance, c("random_baseline", ))

ggplot(data = performance[1:6])+
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
  ggtitle("Performance of the Binary Predictors")



ggplot(general_performance)+
  geom_bar(mapping = aes(x = variable, y = value, fill = model), stat = "identity", position = position_dodge2())+
  geom_errorbar(mapping = aes(x = variable, ymin = value - stdev, ymax = value + stdev), position = position_dodge2())+
  geom_text(aes(x = variable, y = -0.02, label = round(value, 2)),
            position = position_dodge2(width = 0.9), size = 3.5)+
  ggtitle("General Performance Assessment of the Final Models")+
  xlab("metric")+
  ylab("performance")+
  scale_fill_manual(name="Model", values=c("#66FF66", "#FF6600", "#CC0000", "#CC6699", "#9999CC"))+
  theme_bw()
