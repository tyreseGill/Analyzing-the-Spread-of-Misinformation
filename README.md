# Real-Life Social Network Challenge: Misinformation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Matplotlib](https://img.shields.io/pypi/v/matplotlib?label=Matplotlib&color=orange&logo=plotly&logoColor=white)](https://matplotlib.org/)
[![NetworkX](https://img.shields.io/pypi/v/networkx?label=NetworkX&color=blue&logo=python&logoColor=white)](https://networkx.org/)
[![NumPy](https://img.shields.io/pypi/v/numpy?label=NumPy&color=blue&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/pypi/v/pandas?label=Pandas&color=blue&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

### Team Members: TBA

## Description
This project analyzes real-world Facebook Page–Page networks to identify structural weak points that enable misinformation to spread through social graphs. Using graph-theoretic tools and network science techniques, the project constructs a full misinformation-risk assessment and structural analysis to identify where misinformation is most likely to originate from and spread.

## Requirements

- Python 3.10+
- Numpy
- NetworkX
- Matplotlib
- Pandas

All dependencies are listed in `requirements.txt`


## Usage Instructions
1. Download python from the official website [https://www.python.org/downloads/](https://www.python.org/downloads/) if you have not already done so.
2. Clone/download a copy of this repository.
3. Open your terminal and navigate to the project folder containing "`page_rank.py`".
4. Create a virtual environment within the folder by typing in `python -m venv venv` and pressing enter.
    - Confirm that the `venv/` folder exists with: `ls` for Linux/macOs or `dir` for Windows.
5. Activate the environment
    - On Windows, this is done via: `venv\Scripts\Activate`.
    - On Linux/macOS, this is done via: `source venv/bin/activate`.
6. Install the necessary packages with into the environment: `pip install -r requirements.txt`.
7. Run the program by running the example commands below.


### Commands

- *Command*:
```bash
python ./main.py data/facebook_large/musae_facebook_edges.csv --bow_tie --title "Facebook Bow-Tie Structure" --sample_size 400
```
- *Output*:

![Bow Tie of FaceBook Structure](img/fb_bow_tie_structure.png)


#### Risk Assessment
- *Command*:

```bash
python ./main.py data/facebook_large/musae_facebook_edges.csv --risk_assessment --title "Facebook Risk-Distribution Structure" --sample_size 400
```

- *Output 1*:

![Bow Tie of FaceBook Structure](img/fb_bow_tie_structure.png)

- *Output 2*:
```bash
=========== Graph Overview ===========
nodes: 400
edges: 2325
average_degree: 11.625
average_clustering: 0.346
largest_component_size: 366

=========== Core-Periphery ===========
max_k_core: 21
num_core_nodes: 33
top_core_nodes: [14497, 12464, 10426, 6441, 2442, 4502, 16895, 18966, 2596, 16977]

=========== Structural Weakpoints ===========
num_articulation_points: 24
num_bridges: 32
top_bridge_nodes: [(6174, 2), (20550, 2), (481, 2), (10857, 2), (15419, 2), (11507, 2), (18427, 2), (52, 1), (6206, 1), (2124, 1)]

=========== Influence ===========
top_pagerank: [(21729, 0.007636074556628897), (19743, 0.0076359637048122484), (701, 0.007151247266633645), (11003, 0.00709037568021191), (18952, 0.006729666125127268), (10426, 0.006716618324470428), (21120, 0.006229129033156406), (1387, 0.0059877923586631986), (16895, 0.0059709942373022415), (11507, 0.005912269470727401)]
top_betweeness: [(11003, 0.116606900192846), (701, 0.08919439283582346), (18819, 0.05396264684833749), (19743, 0.052513600731307924), (21729, 0.05230511425234946), (20083, 0.04761004845657377), (11158, 0.047375221256313944), (11507, 0.046966725170307345), (22171, 0.03944941461419854), (11611, 0.03878644363999343)]
top_degree: [(21729, 56), (19743, 56), (10426, 53), (1387, 51), (16895, 50), (5458, 49), (15236, 46), (6441, 44), (14497, 43), (10379, 42)]

=========== Risk Analysis ===========
average_risk: 1.16
max_risk: 8
top_risk_nodes: [1654, 10426, 21120, 4809, 11003, 21729, 11507, 19743, 21955, 18216]

=========== Echo Chambers ===========
top_clustering_nodes: [(4275, 1.0), (8417, 1.0), (10500, 1.0), (20990, 1.0), (4760, 1.0), (4826, 1.0), (4907, 1.0), (1193, 1.0), (13525, 1.0), (3300, 1.0)]
```

## Summary

This project analyzes Facebook Page–Page networks to identify structural vulnerabilities that enable misinformation to spread. Using core–periphery decomposition, articulation-point detection, centrality measures, and clustering analysis, the network is partitioned into meaningful structural regions such as core hubs, inner and outer shells, tubes, tendrils, and disconnected nodes. A custom misinformation risk score integrates influence and structural weakness indicators to highlight the highest-risk pages. Radial bow-tie visualization reveals how misinformation can flow from a dense internal core, through key structural bridges (tubes), and outward into peripheral communities. Overall, the project demonstrates how network structure shapes the propagation potential of misinformation in real social graphs.
