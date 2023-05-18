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
# this is the validation set! i want to do it for test, right?
cutoff_perf <- data.table(read.table('../results/logs/validation_mobidb_2_FNN_5.txt', header = TRUE, sep = "\t"))
cutoff_perf <- cutoff_perf[Fold == 4]

ggplot(data=cutoff_perf)+
  geom_line(mapping = aes(x=Cutoff, y=Prec, color='Precision'), size = 1)+
  geom_point(mapping=aes(x=Cutoff, y=Prec, color='Precision'), size = 2)+
  geom_line(mapping = aes(x=Cutoff, y=Rec, color='Recall'), size = 1)+
  geom_point(mapping=aes(x=Cutoff, y=Rec, color='Recall'), size = 2)+
  ylab('Performance [%]')+
  ggtitle('Precision-Recall-Curve on the Validation Set\nfor model FNN_all')
  
ggplot(data=cutoff_perf)+
  geom_line(mapping = aes(x=Cutoff, y=D_NPrec, color='Negative Precision'), size = 1)+
  geom_point(mapping=aes(x=Cutoff, y=D_NPrec, color='Negative Precision'), size = 2)+
  geom_line(mapping = aes(x=Cutoff, y=D_NRec, color='Negative Recall'), size = 1)+
  geom_point(mapping=aes(x=Cutoff, y=D_NRec, color='Negative Recall'), size = 2)+
  ylab('Performance [%]')+
  ggtitle('Negative Precision-Recall-Curve on the Validation Set\nfor model FNN_all')






