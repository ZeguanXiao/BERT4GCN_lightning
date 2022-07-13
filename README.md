<div align="center">

# BERT4GCN

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>


</div>

## Description

Official implementation of EMNLP 2021 paper [BERT4GCN: Using BERT Intermediate Layers to Augment GCN for Aspect-based Sentiment Classification](https://aclanthology.org/2021.emnlp-main.724)

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/ZeguanXiao/BERT4GCN_lightning
cd BERT4GCN_lightning

# [OPTIONAL] create conda environment
conda create -n bert4gcn python=3.8
conda activate bert4gcn

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
bash sctipts/build_env.sh
```

Parse dependency graph
```bash
bash scripts/preprocess.sh
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
bash scripts/schedule.sh
```

[Here](http://1.15.185.201:8080/xzg/BERT4GCN-Lightning/table?workspace=user-xzg) are experiments with above code.

## Citation

If the code is used in your research, please star our repo and cite our paper as follows:
```bash
@inproceedings{xiao-etal-2021-bert4gcn,
    title = "{BERT}4{GCN}: Using {BERT} Intermediate Layers to Augment {GCN} for Aspect-based Sentiment Classification",
    author = "Xiao, Zeguan  and
      Wu, Jiarun  and
      Chen, Qingliang  and
      Deng, Congjian",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.724",
    doi = "10.18653/v1/2021.emnlp-main.724",
    pages = "9193--9200",
    abstract = "Graph-based Aspect-based Sentiment Classification (ABSC) approaches have yielded state-of-the-art results, expecially when equipped with contextual word embedding from pre-training language models (PLMs). However, they ignore sequential features of the context and have not yet made the best of PLMs. In this paper, we propose a novel model, BERT4GCN, which integrates the grammatical sequential features from the PLM of BERT, and the syntactic knowledge from dependency graphs. BERT4GCN utilizes outputs from intermediate layers of BERT and positional information between words to augment GCN (Graph Convolutional Network) to better encode the dependency graphs for the downstream classification. Experimental results demonstrate that the proposed BERT4GCN outperforms all state-of-the-art baselines, justifying that augmenting GCN with the grammatical features from intermediate layers of BERT can significantly empower ABSC models.",
}
```

