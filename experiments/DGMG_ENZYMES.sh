#!/bin/bash
cd ..
python main.py --graph_type ENZYMES --note DGMG --sample_size 16 --gcn_type gat --max_cr_iteration 5 --sample_arch "position" --enable_gcn --nobfs