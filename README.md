# PTN: Perspective Transfer Network

This is the official implementation of the [CIKM 2022](https://www.cikm2022.org/) paper: [Few-Shot Relational Triple Extraction with Perspective Transfer Network](https://dl.acm.org/doi/10.1145/3511808.3557323).

# Updates
- 31/10/2022: We have released the organized code.
- 30/10/2022: We have released the datasets used in the paper.
- 29/10/2022: We have initialized the repository and will gradually release the contents.

# Datasets
- **<u>[FewRel](datasets/fewrel/)</u>**: We follow the original division of FewRel for
evaluating few-shot RTE, where the validation set of FewRel is considered as the testing set, and the testing set of **NYT10** is used as the validation set.
- **<u>[FewRel*](datasets/fewrel_star/)</u>**: We use it for few-shot RTE by randomly choosing 50 relations as the training set, 15 relations as the validation set, and 15 relations as the testing
set.
- **<u>[NYT10](datasets/nyt10/)</u>**: We filter sentences that contain only one relation, forming a dataset of 16 relations. Since this dataset contains few relations, we use the entire dataset as the testing set.
- **<u>[WebNLG](datasets/webnlg/)</u>**: We screen sentences in which the gold entities exactly appear, establishing a new dataset of 31 relations. To construct the testing set, we randomly sample sentences without replacement and classify the relations that appear in the sentences into the testing set until the number of relations reaches 16 (same with the number of relations in the above testing sets). The remaining relations serve as the validation set.

# How to Run
Run `main.py`. The arguments are presented below.
```
  --lr                  Learning rate
  --epoch               Number of epoch
  --batchsize           Batch size on training
  --batchsize_test      Batch size on testing
  --print_per_batch     Print results every XXX batches
  --numprocess          Number of process
  --start               Directory to load model
  --test                Set to True to inference
  --debug               Set to True to debug
  --dataset             Data directory
  --gpu                 gpuid
  --N                   N way
  --K                   K shot
  --seed                seed
  --na_prob             na prob
  --plm_path            pretrain language model path
  --train_iter          train iter
  --dev_iter            dev iter
  --test_iter           test iter
  --alpha               alpha
  --beta                beta
  --gamma               gamma
```

Take FewRel dataset for example, the expriments can be run as follows.

## 5-way 1-shot
```
python main.py --N 5 --K 1 --dataset fewrel
```

## 10-way 5-shot 30% NOTP
```
python main.py --N 10 --K 5 --dataset fewrel --na_prob 0.3
```

# Inference
You can evaluate an existing checkpoint by
```
python main.py --test --start {CHECKPOINT_PATH}
```

# Citation
If you use TPN in your work, please cite our paper:
```bibtex
@inproceedings{10.1145/3511808.3557323,
    author = {Fei, Junbo and Zeng, Weixin and Zhao, Xiang and Li, Xuanyi and Xiao, Weidong},
    title = {Few-Shot Relational Triple Extraction with Perspective Transfer Network},
    year = {2022},
    publisher = {Association for Computing Machinery},
    url = {https://doi.org/10.1145/3511808.3557323},
    doi = {10.1145/3511808.3557323},
    booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
    pages = {488–498},
    series = {CIKM '22}
}
```
