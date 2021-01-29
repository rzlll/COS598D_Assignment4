# ADAPTIVE LEARNED BLOOM FILTER (ADA-BF)

This a PyTorch implementation of the Adaptive Learned Bloom Filter [1]. 

The python files include the implementation of the 1) Bloom filter (BF), partial implementation of 2) learned BF, and partial implementation of3) Ada-BF, and print the size of False Positives of the corresponding algorithm.

Your task is to complete function to search the optimal parameters for learned BF and  Ada-BF. Specfically, you are required to use cross validataion to find the best threshold for the ML classifier output scores for learned BF in `learned_Bloom_filter.py`; and find the best number of groups (*g* in [1]) and the best threshold of density ratio (*c* in [1]) in `Ada_BF.py`. 

**Reference**

[1] Dai, Z. and Shrivastava, A., 2019. Adaptive learned Bloom filter (Ada-BF): Efficient utilization of the classifier. arXiv preprint arXiv:1910.09131.

## How to run?

**Input arguments**: 
- `--data_path`: a csv file includes the items, scores and labels; `--size_of_Ada_BF`: size of Bloom filter;
- (for learned Bloom filter) `--threshold_min` and `--threshold_max` provide the range of the score threshold (between `threshold_min` and `threshold_max`). Items with score larger than the threshold are identified as keys;
- (for Ada-BF) `--num_group_min` and `--num_group_max` give the range of number of groups to divide (range of *g*); `--c_min` and `--c_max` provide the range of *c* where *c=m_j/m_{j+1}*

**Commands**:
- Run Bloom filter: `python Bloom_filter.py --data_path ./Datasets/URL_data.csv --size_of_BF 200000`
- Run learned Bloom filter and search the best threshold in [0.5, 0.95]: 
`python learned_Bloom_filter.py --data_path ./Datasets/URL_data.csv --size_of_LBF 200000  --threshold_min 0.5   --threshold_max 0.95`
- Run Ada-BF and search the best *g* in [8,12] and *c* in [1.6, 2.5]. Note *c>1*: 
`python Ada-BF.py --data_path ./Datasets/URL_data.csv --size_of_Ada_BF 200000  --num_group_min 8  --num_group_max 12  --c_min 1.6  --c_max 2.5`

**Your Tasks**:
1. Ipm
2.
Compare FPR of Bloom filter (BF), learned BF, and Ada-BF with memory budget using the provided two datasets in /Datasets/. Please refer to the figure 4 to report your results. 


