# ADAPTIVE LEARNED BLOOM FILTER (ADA-BF)

This a PyTorch implementation of the Adaptive Learned Bloom Filter[1]. The python files include the implementation of the Bloom filter, learned Bloom filter, Ada-BF and disjoint Ada-BF, and print the size of False Positives of the corresponding algorithm.

[1] Dai, Z. and Shrivastava, A., 2019. Adaptive learned Bloom filter (Ada-BF): Efficient utilization of the classifier. arXiv preprint arXiv:1910.09131.

**Input arguments**: 
- `--data_path`: a csv file includes the items, scores and labels; `--size_of_Ada_BF`: size of Bloom filter;
- (for learned Bloom filter) `--threshold_min` and `--threshold_max` provide the range of the score threshold (between `threshold_min` and `threshold_max`). Items with score larger than the threshold are identified as keys;
- (for Ada-BF) `--num_group_min` and `--num_group_max` give the range of number of groups to divide (range of *g*
); `--c_min` and `--c_max` provide the range of *c* where *c=m_j/m_{j+1}*

**Commands**:
- Run Bloom filter: `python Bloom_filter.py --data_path ./Datasets/URL_data.csv --size_of_BF 200000`
- Run learned Bloom filter: `python learned_Bloom_filter.py --data_path ./Datasets/URL_data.csv --size_of_LBF 200000  --threshold_min 0.5   --threshold_max 0.95`
- Run Ada-BF: `python Ada-BF.py --data_path ./Datasets/URL_data.csv --size_of_Ada_BF 200000  --num_group_min 8  --num_group_max 12  --c_min 1.6  --c_max 2.5`

**Your Tasks**:
1.
2.
Compare FPR of Bloom filter (BF), learned BF, and Ada-BF with memory budget using the provided two datasets in /Datasets/. Please refer to the figure 4 to report your results. 


