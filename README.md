# Datasets 
Models are trained on CoNLL 2003 relabelled dataset provided by [Liang et al](https://arxiv.org/abs/2006.15509).
Provided dataset was created using KB matching, for more details consider visiting [BOND](https://github.com/cliang1453/BOND).

KB-matching scores evaluated with original train part of CoNLL 2003:

| F1     | Precision | Recall | 
|--------|-----------|--------|
| 0.7097 | 0.8238    | 0.6233 |

Models were evaluated on both original test part of CoNLL 2003 dataset and corrected test part provided by [Wang et al](https://arxiv.org/pdf/1909.01441v1.pdf)

# Baseline
As baseline fine-tuned RoBERTa with document-level context was used. As expected, it effectively remembers all the incorrect annotations 
in dataset from KB matching. The results (**median** ± std) obtained from 5 runs with different RNG seeds (F1 / Precision / Recall) are as follows:

| original test                                         | corrected test                                         |
|-------------------------------------------------------|--------------------------------------------------------|
| **71.38** ± 0.56 / **81.50** ± 0.6 / **63.49** ± 0.54 | **71.94** ± 0.54 / **82.59** ± 0.59 / **63.72** ± 0.52 |

Experiment configs: `experiments/configs/baseline/*.json`

# BOND
Extremely volatile method that requires rigorous hyperparameter fine-tuning. The hyperparameters were tuned 
with RNG seed 42 and evaluated with seeds 1-5. While the best model reached the F1 score of **82.97**
(`experiments/relabel/bond.json`), after changing the seed results became not so prominent:

| original test                                          | corrected test                                         |
|--------------------------------------------------------|--------------------------------------------------------|
| **77.97** ± 1.67 / **79.92** ± 1.25 / **77.50** ± 2.73 | **78.93** ± 1.71 / **81.21** ± 1.12 / **78.36** ± 2.81 |

Which shows that model performance is greatly dependent on parameter initialization.

Experiment configs: `experiments/configs/bond/*.json`

# Co-regularization

Robust method for filtering out noisy annotations.


| original test                                          | corrected test                                         |
|--------------------------------------------------------|--------------------------------------------------------|
| **77.17** ± 0.23 / **89.64** ± 0.25 / **67.72** ± 0.32 | **77.77** ± 0.21 / **90.78** ± 0.21 / **67.95** ± 0.31 |


Experiment configs: `experiments/configs/coregularization/*.json`

# Results reproduction

Use TRIAGE to run any experiment config. Use `--help` option to get familiar 
with possible command-line arguments.

## Example: baseline

```bash
./init.sh
triage experiments/configs/baseline/*.json
```

## Acknowledgements
Most of the code is based on the work of Liang et al. [BOND: Bert-Assisted Open-Domain Named Entity Recognition with Distant Supervision](https://arxiv.org/abs/2006.15509) ([GitHub](https://github.com/cliang1453/BOND)).

Co-regularization technique was adapted from [wzhouad](https://github.com/wzhouad/NLL-IE).

`MarginalCRF` implementation was taken from [kajyuuen](https://github.com/kajyuuen/pytorch-partial-crf).

CrossWeigh dataset weighing was adapted from [ZihanwangKi](https://github.com/ZihanWangKi/CrossWeigh).