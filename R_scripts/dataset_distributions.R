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
  ylim(0, 0.12)+
  ggtitle("Score distribution of the datasets\nScore = protein_length + 5 * %_non_binding_in_disorder")
  

# remove some 
test_distr_2 <- subset(test_distr, !(protein %in% c("P81455",
                                                    "O34800",
                                                      "P15340",
                                                    "O14140",
                                                      "P22531",
                                                    "Q9KJ82",
                                                      "P80220",
                                                    "P0C079",
                                                      "Q9I322",
                                                    "Q9XES8",
                                                    "P12520",
                                                      
                                                    "O60829",
                                                    "Q8GT36",	
                                                    "Q9KMA5",
                                                      "P05318",
                                                    "P17096",
                                                    "P22943",
                                                    "A4ZNR2",
                                                      "Q9Y3M2",
                                                    "P0C0S5",
                                                    "A0A220GHA5",
                                                    "Q9V452",
                                                      "Q64364",
                                                      "G0SCY6",
                                                      "Q9BXW4",
                                                      
                                                      "P40019",
                                                    "Q88MI5",
                                                    "Q84MC7",
                                                      "P62696",
                                                      "O75817",
                                                    "Q9UKK9",
                                                      
                                                    "P55212",
                                                      "Q7B2Z6",
                                                    "P0AFC3",
                                                    "P07355",
                                                      "P0ACN7",
                                                      "P04183",
                                                      "P51946",
                                                      "O66858",
                                                    
                                                    # length only
                                                    "P0C8E0"	,
                                                    "Q8IVG9"	,
                                                    "P68005"	,
                                                    "P0C0Y1"	,
                                                    "Q9UUB7",
                                                    "Q7YUB9",
                                                    
                                                    "Q9Y6H6"	,
                                                    "A1B602",
                                                    "Q9Y3B4"	,
                                                    "Q9JLC3"	,
                                                    "O14352",
                                                    "Q9DBG9"	,
                                                    "Q96E14"	,
                                                    "E6PBU9",
                                                    
                                                    "P05452"	,
                                                    "P0ABJ3"	,
                                                    "P52565",
                                                    "Q9UI95",
                                                    "P77072",
                                                    "P08008"	,
                                                    "P53927",
                                                    "Q12483",
                                                    "Q71FK2",
                                                    "O00299"	,
                                                    "P0A2N1",
                                                    "P17931",
                                                    "P03259",
                                                    "Q9ZKK2"	,
                                                    "P46670",
                                                    "Q9VAN6"	,
                                                    "Q9UIV1"	,
                                                    "P06748",	
                                                    "P24005",
                                                    "Q9RFD6"))) 
ggplot()+
  geom_histogram(data = train_distr, mapping = aes(y = stat(count / sum(count)), x = score, fill = "train"), alpha = 0.5, bins = 30)+
  geom_histogram(data = test_distr_2, mapping = aes(y = stat(count / sum(count)), x = score, fill = "test"), alpha = 0.5, bins = 30)+
  scale_x_log10()+
  annotation_logticks()+
  ylim(0, 0.12)+
  ggtitle("Length distribution of the secondary datasets")


  