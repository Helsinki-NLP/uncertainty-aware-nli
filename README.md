# Uncertainty-Aware NLI with Stochastic Weight Averaging

This repository contains the code for running the experiments reported in our paper:

**Aarne Talman, Hande Celikkanat, Sami Virpioja, Markus Heinonen, Jörg Tiedemann. 2023. [Uncertainty-Aware Natural Language Inference with Stochastic Weight Averaging](https://openreview.net/forum?id=uygq9_N7TL). Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa).**


## How to Install and Run

Clone this repository and install the dependencies by running
```
git clone git@github.com:Helsinki-NLP/uncertainty-aware-nli.git
cd uncertainty-aware-nli
pip3 install -r requirements.txt
```

Download and prepare data by running:
```
./download_data.sh
```

See [train.sh](train.sh) and [experiment.sh](experiment.sh) for examples on how to run the code and modify those for your environment.

## Paper

Please cite our work:

```
@inproceedings{
    talman2023uncertaintyaware,
    title={Uncertainty-Aware Natural Language Inference with Stochastic Weight Averaging},
    author={Aarne Talman and Hande Celikkanat and Sami Virpioja and Markus Heinonen and J{\"o}rg Tiedemann},
    booktitle={The 24rd Nordic Conference on Computational Linguistics},
    year={2023},
    url={https://openreview.net/forum?id=uygq9_N7TL}
}
```