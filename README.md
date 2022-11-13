# MOP-JCI
### Identifying Heterogeneous Treatment Effects in Multiple Outcomes using Joint Confidence Intervals

Heterogeneous treatment effects (HTEs) are commonly identified during randomized controlled trials (RCT). Identifying subgroups of patients with similar treatment effects is of high interest in clinical research to advance precision medicine. Often, multiple clinical outcomes are measured during an RCT, each having a potentially heterogeneous effect. Recently there has been high interest in identifying subgroups from HTEs, however, development has been lacking in settings where there are multiple outcomes. In this work, we propose a framework for partitioning the covariate space to identify subgroups across multiple outcomes based on the joint confidence intervals. We test our algorithm on synthetic and semi-synthetic data where there are two outcomes, and we demonstrate that our algorithm is robust in identifying subgroups across data with varying correlation and deviation.


* **make_data.py**: simulates synthetic and semi-synthetic datasets.
* **MOP_JCI.py**: recursively partitions outcomes using split conformal regression (SCR).
* **MOP_JCI_quantile.py**: recursively partitions outcomes using split conformal quantile regression (SCQR).
* **conf_prediction.py**: implementation of SCR and SCQR.
* **helper.py**: helper functions to generate confidence intervals using SCR and SCQR on varying ITE estimators.

#### *Licenses*
The license of the assets used in this paper are as follows:
* Robust recursive partitioning algorithm, and synthetic data: https://github.com/vanderschaarlab/mlforhealthlabpub/blob/main/LICENSE.md
* SCR: https://github.com/ryantibs/conformal/blob/master/LICENSE
* SCQR: https://github.com/yromano/cqr/blob/master/LICENSE
* IHDP dataset: Code zipped in the supplementary material of [1].

[1] Jennifer L. Hill. Bayesian nonparametric modeling for causal inference. Journal of Computational and Graphical Statistics, 20:217–240, 1 2011. ISSN 1061-8600. doi:10.1198/jcgs.2010.08162
