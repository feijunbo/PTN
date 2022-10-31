# PTN: Perspective Transfer Network

This is the official implementation of the [CIKM 2022](https://www.cikm2022.org/) paper: [Few-Shot Relational Triple Extraction with Perspective Transfer Network](https://dl.acm.org/doi/10.1145/3511808.3557323).

# Updates
- 30/10/2022: We have released the datasets used in the paper.
- 29/10/2022: We have initialized the repository and will gradually release the contents.

# Datasets
- **<u>[FewRel](datasets/fewrel/)</u>**: We follow the original division of FewRel for
evaluating few-shot RTE, where the validation set of FewRel is considered as the testing set, and the testing set of **NYT10** is used as the validation set.
- **<u>[FewRel*](datasets/fewrel_star/)</u>**: We use it for few-shot RTE by randomly choosing 50 relations as the training set, 15 relations as the validation set, and 15 relations as the testing
set.
- **<u>[NYT10](datasets/nyt10/)</u>**: We filter sentences that contain only one relation, forming a dataset of 16 relations. Since this dataset contains few relations, we use the entire dataset as the testing set.
- **<u>[WebNLG](datasets/webnlg/)</u>**: We screen sentences in which the gold entities exactly appear, establishing a new dataset of 31 relations. To construct the testing set, we randomly sample sentences without replacement and classify the relations that appear in the sentences into the testing set until the number of relations reaches 16 (same with the number of relations in the above testing sets). The remaining relations serve as the validation set.

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
