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

# this is for the validation set!

plot <- function(m){
  models_to_assess <- c('mobidb_2_CNN_1', 'mobidb_2_D_CNN_2', 'mobidb_2_FNN_5', 'mobidb_2_D_FNN_4', 'AAindex_D_baseline_2')
  names_simple <- c('CNN_all', 'CNN_disorder', 'FNN_all', 'FNN_disorder', 'AAindex_disorder')
  best_folds <- c(1, 3, 4, 1, 3)
  
  cutoff_perf <- data.table(read.table(paste('../results/logs/validation_',models_to_assess[m] , '.txt', sep = ''), header = TRUE, sep = "\t"))
  cutoff_perf <- cutoff_perf[Fold == best_folds[m]]
  
  print(
    ggplot(data=cutoff_perf)+
      geom_line(mapping = aes(x=Cutoff, y=Prec, color='Precision'), size = 1)+
      geom_point(mapping=aes(x=Cutoff, y=Prec, color='Precision'), size = 2)+
      geom_line(mapping = aes(x=Cutoff, y=Rec, color='Recall'), size = 1)+
      geom_point(mapping=aes(x=Cutoff, y=Rec, color='Recall'), size = 2)+
      ylab('Performance [%]')+
      ggtitle(names_simple[m])+
      labs(color='Performance Measure')+
      theme_bw()
  )
  print(
    ggplot(data=cutoff_perf)+
      geom_line(mapping = aes(x=Cutoff, y=D_NPrec, color='Negative Precision'), size = 1)+
      geom_point(mapping=aes(x=Cutoff, y=D_NPrec, color='Negative Precision'), size = 2)+
      geom_line(mapping = aes(x=Cutoff, y=D_NRec, color='Negative Recall'), size = 1)+
      geom_point(mapping=aes(x=Cutoff, y=D_NRec, color='Negative Recall'), size = 2)+
      ylab('Performance [%]')+
      ggtitle(names_simple[m])+
      labs(color='Performance Measure')+
      theme_bw()
  )
  
  
}



for (m in 1:5){
  print(m)
  plot(m)}






