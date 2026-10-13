# SQ-HAF

**Source Quality-Guided Hierarchical Adaptation and Multi-Branch Fusion for Few-Shot Cross-Subject SSVEP Recognition**

This repository accompanies the SQ-HAF study on few-shot cross-subject recognition of steady-state visual evoked potentials (SSVEPs). The method addresses differences in transferability across source subjects and variability between subjects by combining source quality assessment, hierarchical adaptation, and multi-branch decision fusion.

## Method at a glance

SQ-HAF is described in the paper as a pipeline with the following components:

1. **Source quality assessment and selection** using target–source template similarity, with split-half template consistency (STC) as an additional confidence signal when enough labeled calibration trials are available.
2. **Within-subject adaptation** using frequency-neighborhood-regularized spatial filtering.
3. **Across-subject adaptation** using covariance alignment and generalized and source-specific templates.
4. **Multi-branch recognition** that fuses five-subband harmonic reference CCA scores with source-template scores.

See the [paper](https://doi.org/10.3390/s26185830) for the complete method, experimental protocol, and reported results.

## Paper

Lin, H.; He, Q.; Zhu, Y.; Si, J.; Wang, F. SQ-HAF: Source Quality-Guided Hierarchical Adaptation and Multi-Branch Fusion for Few-Shot Cross-Subject SSVEP Recognition. *Sensors* **2026**, *26*(18), 5830. [https://doi.org/10.3390/s26185830](https://doi.org/10.3390/s26185830)

## Datasets and platform

The experiments use the following public SSVEP datasets. Please obtain the data from the provider and follow its access and usage instructions.

- **Benchmark dataset** — available from the [Tsinghua University BCI Laboratory](https://bci.med.tsinghua.edu.cn/). See Wang et al. (2017) below.
- **BETA dataset** — available from the [Tsinghua University BCI Laboratory](https://bci.med.tsinghua.edu.cn/). See Liu et al. (2020) below.
- **MetaBCI** — the open-source BCI platform used in this work: [TBC-TJU/MetaBCI](https://github.com/TBC-TJU/MetaBCI). Refer to its repository for platform installation and usage documentation.

## References
1. Liu, B.; Huang, X.; Wang, Y.; Chen, X.; Gao, X. BETA: A Large Benchmark Database Toward SSVEP-BCI Application. *Frontiers in Neuroscience* **2020**, *14*, 627.
2. Wang, Y.; Chen, X.; Gao, X.; Gao, S. A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces. *IEEE Transactions on Neural Systems and Rehabilitation Engineering* **2017**, *25*, 1746–1752.
3. Mei, J.; Luo, R.; Xu, L.; Zhao, W.; Wen, S.; Wang, K.; Xiao, X.; Meng, J.; Huang, Y.; Tang, J.; et al. MetaBCI: An open-source platform for brain–computer interfaces. *Computers in Biology and Medicine* **2024**, *168*, 107806.
