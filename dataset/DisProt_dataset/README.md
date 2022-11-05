# Construction of dataset
## General construction
1. Parse annotations from DisProt. Only consider proteins with any disorder annotation, remove binding annotations outside of disordered regions.
2. Extract proteins also overlapping with bindEmbed's development set --> 39 proteins (27 negatives - i.e., only disorder annotations, 12 positives - i.e., at least 1 binding region annotated)
3. Get remaining set --> 1,566 proteins (1,037 negatives, 529 positives)

## Construct training set
To ensure large enough data set size, the training set is less strictly redundancy reduced than the test set
1. Redundancy reduce positives using cd-hit at 40% sequence identity --> 487 proteins
2. Redundancy reduce negatives using cd-hit at 40% sequence identity --> 942 proteins
3. Redundancy reduce negatives against positives using cd-hit-2d at 40% sequence identity --> 903 proteins (65%)

## Construct test set (100 proteins)
1. Redundancy reduce positives using mmseqs-uniqueprot at H-val=0 --> 259 proteins
2. Redundancy reduce negatives using mmseqs-uniqueprot at H-val=0 --> 480 proteins
3. Redundancy reduce negatives against positives using mmseqs-uniqueprot at H-val=0 --> 220 proteins
4. Extract 38 negatives and 23 positives to mirror distribution of positives and negatives in training set
5. **Final test set: 100 proteins (65 negatives, 35 positives)**

## Finalize train set
1. Redundancy reduce training set against test set using mmseqs-uniqueprot at H-val=0
2. **Final training set: 754 proteins (477 negatives, 277 positives)**