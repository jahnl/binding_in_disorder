library(ggplot2)
library(stringr)
library(scales)

test <- readLines("../dataset/MobiDB_dataset/test_set_stats.txt")
train <- readLines("../dataset/MobiDB_dataset/train_set_stats.txt")
# parameters in lines: length-2, n_disordered-4, n_structured-6, n_D_binding-8, 
#                      n_D_nonbinding-10, binding_positioning_distr-12,
#                      D_region_length-14

extract_numerics <- function(x){
  numerics <- scan(text = substring(x, first=2, last = nchar(x)-1),
                      what = integer(), quiet = TRUE, sep =",")
  data.frame(numerics)
}

#length
test_length <- extract_numerics(test[2])
train_length <- extract_numerics(train[2])
ggplot()+
  geom_histogram(data = test_length, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "test"), alpha = 0.5)+
  geom_histogram(data = train_length, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "train"), alpha = 0.5)+
  xlim(0, 4000)+
  scale_x_log10()+
  xlab("protein length")+
  labs(fill="set")+
  ggtitle("Distribution of protein length in test vs train set")


#disordered region length
test_D_length <- extract_numerics(test[14])
train_D_length <- extract_numerics(train[14])
ggplot()+
  geom_histogram(data = test_D_length, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "test"), alpha = 0.5)+
  geom_histogram(data = train_D_length, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "train"), alpha = 0.5)+
  scale_x_log10()+
  xlab("disordered region length")+
  labs(fill="set")+
  ggtitle("Distribution of disordered region length in test vs train set")


# n disordered residues
test_d <- extract_numerics(test[4])
train_d <- extract_numerics(train[4])
ggplot()+
  geom_histogram(data = test_d, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "test"), alpha = 0.5)+
  geom_histogram(data = train_d, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "train"), alpha = 0.5)+
  scale_x_log10()+
  xlab("# disordered residues per protein")+
  labs(fill="set")+
  ggtitle("Distribution of the number of disordered residues per protein in test vs train set")

# percentage disordered residues
ggplot()+
  geom_histogram(data = test_d/test_length*100, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "test"), alpha = 0.5)+
  geom_histogram(data = train_d/train_length*100, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "train"), alpha = 0.5)+
  xlab("disordered residues per protein [%]")+
  labs(fill="set")+
  ggtitle("Distribution of the percentage of disordered residues per protein in test vs train set")

# n structured residues
test_s <- extract_numerics(test[6])
train_s <- extract_numerics(train[6])
ggplot()+
  geom_histogram(data = test_s, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "test"), alpha = 0.5)+
  geom_histogram(data = train_s, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "train"), alpha = 0.5)+
  scale_x_log10()+
  xlab("# structured residues per protein")+
  labs(fill="set")+
  ggtitle("Distribution of the number of structured residues per protein in test vs train set")

## percentage structured residues (aka just the inverse of percentage disorder)
#ggplot()+
#  geom_histogram(data = test_s/test_length*100, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "test"), alpha = 0.5)+
#  geom_histogram(data = train_s/train_length*100, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "train"), alpha = 0.5)+
#  xlab("structured residues per protein [%]")+
#  labs(fill="set")+
#  ggtitle("Distribution of the percentage of structured residues per protein in test vs train set")


# n binding residues in disorder
test_b <- extract_numerics(test[8])
train_b <- extract_numerics(train[8])
ggplot()+
  geom_histogram(data = test_b, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "test"), alpha = 0.5)+
  geom_histogram(data = train_b, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "train"), alpha = 0.5)+
  scale_x_sqrt()+
  scale_y_sqrt()+
  xlab("# disordered binding residues per protein")+
  labs(fill="set")+
  ggtitle("Distribution of the number of binding residues per protein in test vs train set")

# percentage binding residues
ggplot()+
  geom_histogram(data = test_b/test_length*100, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "test"), alpha = 0.5)+
  geom_histogram(data = train_b/train_length*100, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "train"), alpha = 0.5)+
  scale_y_sqrt()+
  xlab("binding residues per protein [%]")+
  labs(fill="set")+
  ggtitle("Distribution of the percentage of binding residues per protein in test vs train set")

ggplot()+
  geom_histogram(data = test_b/test_d*100, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "test"), alpha = 0.5)+
  geom_histogram(data = train_b/train_d*100, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "train"), alpha = 0.5)+
  scale_y_sqrt()+
  xlab("binding residues per disordered parts of a protein [%]")+
  labs(fill="set")+
  ggtitle("Distribution of the percentage of binding residues per disordered parts of a protein in test vs train set")


# n non-binding residues in disorder
test_nb <- extract_numerics(test[10])
train_nb <- extract_numerics(train[10])
ggplot()+
  geom_histogram(data = test_nb, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "test"), alpha = 0.5)+
  geom_histogram(data = train_nb, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "train"), alpha = 0.5)+
  #scale_x_sqrt()+
  scale_y_sqrt()+
  xlab("# disordered non-binding residues per protein")+
  labs(fill="set")+
  ggtitle("Distribution of the number of non-binding residues per protein in test vs train set")

# percentage non-binding residues
ggplot()+
  geom_histogram(data = test_nb/test_length*100, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "test"), alpha = 0.5)+
  geom_histogram(data = train_nb/train_length*100, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "train"), alpha = 0.5)+
  scale_y_sqrt()+
  xlab("non-binding residues per protein [%]")+
  labs(fill="set")+
  ggtitle("Distribution of the percentage of non-binding residues per protein in test vs train set")

ggplot()+
  geom_histogram(data = test_nb/test_d*100, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "test"), alpha = 0.5)+
  geom_histogram(data = train_nb/train_d*100, mapping = aes(y = stat(count / sum(count)), x = numerics, fill = "train"), alpha = 0.5)+
  scale_y_sqrt()+
  xlab("non-binding residues per disordered parts of a protein [%]")+
  labs(fill="set")+
  ggtitle("Distribution of the percentage of non-binding residues per disordered parts of a protein in test vs train set")


# binding positioning
test_p <- extract_numerics(test[12])
train_p <- extract_numerics(train[12])
ggplot()+
  geom_bar(data = test_p, mapping = aes(x = c(1,2,3,4,5), y = numerics/sum(numerics), fill = "test"), stat = "identity", alpha = 0.5)+
  geom_bar(data = train_p, mapping = aes(x = c(1,2,3,4,5), y = numerics/sum(numerics), fill = "train"), stat = "identity", alpha = 0.5)+
  xlab("one fifth of a disordered region")+
  ylab("percentage of occurence")+
  labs(fill="set")+
  ggtitle("Occurence of binding residues in the disordered regions in test vs train set")



###############score distribution###########
train_distr <- read.table("../dataset/MobiDB_dataset/train_set_score_distribution.tsv", header = TRUE)
test_distr <- read.table("../dataset/MobiDB_dataset/test_set_score_distribution.tsv", header = TRUE)
ggplot()+
  geom_histogram(data = train_distr, mapping = aes(y = stat(count / sum(count)), x = score, fill = "train"), alpha = 0.5)+
  geom_histogram(data = test_distr, mapping = aes(y = stat(count / sum(count)), x = score, fill = "test"), alpha = 0.5)+
  scale_x_log10()+
  annotation_logticks()+
  ggtitle("Score distribution of the datasets\nScore = protein_length + 5 * %_non_binding_in_disorder")
  
  