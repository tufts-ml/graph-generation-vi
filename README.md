# Order Matters: Probabilistic Modeling of Node Sequencefor Graph Generation


This repository contains PyTorch implementation of the following paper: ["Order Matters: Probabilistic Modeling of Node Sequence for Graph Generation"](https://arxiv.org/abs/2106.06189)

## 0. Environment Setup
enviroment setup: "run conda install -f enviroment.yml"

installation of Graph automorphism library: https://web.cs.dal.ca/~peter/software/pynauty/html/install.html

## 1. Experiment

``` shell
#DGMG
sh experiments/DGMG_caveman_small.sh
sh experiments/DGMG_ENZYMES.sh
#GraphRNN
sh experiments/GraphRNN_caveman_small.sh
sh experiments/GraphRNN_Lung.sh
#Graphgen
sh experiments/Graphgen_citeseer_small.sh
sh experiments/Graphgen_ENZYMES.sh



```

## 2. Training
To list the arguments, run the following command:
```
python main.py -h
```

To train the given model on Lung dataset, run the following:

``` 
python main.py \
    --graph_tyep Lung                                  \
    --note <GraphRNN, DGMG, Graphgen>                  \
    --sample_size 16                                   \
    --gcn_type <gat, gcn, appnp>                       \
    --max_cr_iteration 5                               \
    --enable_gcn     
```    
   
   
   
         


To train the given model on ENZYMES dataset, run the following:

``` 
python main.py \
    --graph_tyep ENZYMES                               \
    --note <GraphRNN, DGMG, Graphgen>                  \
    --sample_size 16                                   \
    --gcn_type <gat, gcn, appnp>                       \
    --max_cr_iteration 5                               \
    --enable_gcn     
```    

To train the given model on caveman_small dataset, run the following:

``` 
python main.py \
    --graph_tyep caveman_small                         \
    --note <GraphRNN, DGMG, Graphgen>                  \
    --sample_size 16                                   \
    --gcn_type <gat, gcn, appnp>                       \
    --max_cr_iteration 5                               \
    --enable_gcn     
```    

To train the given model on citeseer_small dataset, run the following:

``` 
python main.py \
    --graph_tyep citeseer_small                        \
    --note <GraphRNN, DGMG, Graphgen>                  \
    --sample_size 16                                   \
    --gcn_type <gat, gcn, appnp>                       \
    --max_cr_iteration 5                               \
    --enable_gcn     
```    


