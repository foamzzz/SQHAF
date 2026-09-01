# SQHAF

This repository provides the implementation of SQHAF and the complete
leave-one-subject-out (LOSO) experiments for all evaluated methods.

## Contents

- `SQHAF.py`: Implementation of SQHAF.
- `EEGconformer.py`, `iistlf.py`, and `SUTLSSVEP.py`: Implementations of the corresponding comparison methods.
- `loso_experiment_0.5s.ipynb`, `loso_experiment_1.0s.ipynb`, `loso_experiment_1.5s.ipynb`, and `loso_experiment_2.0s.ipynb`: Complete LOSO experiment notebooks using different time windows.
- `metabci/`: The MetaBCI platform code used in the experiments.

## Datasets and Platform

The experiments use the following datasets and software platform:

- **Benchmark Dataset**: [A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces](https://bci.med.tsinghua.edu.cn/)
- **BETA Dataset**: [BETA: A Large Benchmark Database Toward SSVEP-BCI Application](https://bci.med.tsinghua.edu.cn/)
- **MetaBCI**: [https://github.com/TBC-TJU/MetaBCI](https://github.com/TBC-TJU/MetaBCI)

Please cite the corresponding references when using these datasets or MetaBCI.

## Citation

### MetaBCI

> Mei, J., Luo, R., Xu, L., Zhao, W., Wen, S., Wang, K., Xiao, X., Meng, J., Huang, Y., Tang, J., Cheng, L., Xu, M., and Ming, D. MetaBCI: An open-source platform for brain–computer interfaces. *Computers in Biology and Medicine*, 2024, 168, 107806. https://doi.org/10.1016/j.compbiomed.2023.107806

### Benchmark Dataset

> Wang, Y., Chen, X., Gao, X., and Gao, S. A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 2017, 25(10), 1746–1752. https://doi.org/10.1109/TNSRE.2016.2627556

### BETA Dataset

> Liu, B., Huang, X., Wang, Y., Chen, X., and Gao, X. BETA: A Large Benchmark Database Toward SSVEP-BCI Application. *Frontiers in Neuroscience*, 2020, 14, 627. https://doi.org/10.3389/fnins.2020.00627
