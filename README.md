# Multi-view graph fusion RNN

This is a Pytorch implementation of MVGFRNN architecture as described in the paper 'Predicting Fine-grained Dengue Fever Risk Using Multi-View Graph Fusion RNNs with Approximation for Sensor-less Locations'
![](https://hackmd.io/_uploads/BJ2VkFySn.png)

## How to Run
- Use `process_dengue.py` split labeled/train/valid/test grids by a given labeled-grid ratio, and find the neighbors of unlabeled grids. The results will be saved in a new folder with the given name.
- Use `create_spatial_graph.py` and `create_temporal_graph.py` to generate multi-view graphs.
- Modify configurations in `config.py`.
- Train and evaluate the model in `main.py`.