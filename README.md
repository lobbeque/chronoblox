# Chronoblox: Chronophotographic Sequential Graph Visualization

![Layout](https://img.shields.io/badge/Layout-Python-informational?style=flat&logo=python&color=6144b3)&nbsp;&nbsp;![Interface](https://img.shields.io/badge/Interface-Javascript-informational?style=flat&logo=javascript&color=6144b3)&nbsp;&nbsp;

Since this project is still in development, this document remains in progress.

**Ressources** : [examples](https://lobbeque.github.io/chronoblox_examples/) | publication

## Chronoblox Layout

**Prerequisites** : You first need to install [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) and [graph-tool](https://graph-tool.skewed.de/)

**Other dependencies** : [gensim](https://pypi.org/project/gensim/) | [numpy](https://pypi.org/project/numpy/) | [pacmap](https://pypi.org/project/pacmap/) | [pecanpy](https://pypi.org/project/pecanpy/) | [scikit-learn](https://pypi.org/project/scikit-learn/)

### Launch the script

#### With no graph snapshot

```shell
cd ./layout
conda activate gt
python3 chronoblox.py
```

If you don't have your own snapshots, then by default Chronoblox will use the [board_directors](https://networks.skewed.de/net/board_directors) data set.

#### With your own graph snapshots

```shell
cd ./layout
conda activate gt
python3 chronoblox.py --snapshots ~/path/to/your/snapshots*
```

#### Graph snapshots' requirements

Chronoblox takes a sequence of graph snapshots as input data. Graph snapshots must be in graph-tool `.gt` format. The graphs must contain the following [graph properties](https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.PropertyMap.html#graph_tool.PropertyMap) and vertex properties:

* Graph properties
  * `phase` : a string that indicates the order of the  graph snapshots in the sequence. It can be a date or an Ord number. Graph snapshots will be sorted by `phase` during the embedding and visualization steps.
  * `name` : a that indicates the name of the sequence. It will be used to name the output files.
* Vertex properties
  * `vid` : a unique string identifier for each vertex whatever the phase.
  * `vmeta` : a qualitative metadata or a label attached to each vertex. Metadata will be used to highlight the node groups in the interface. If not completed the default will be _NA_.

### Other arguments

* _--threshold_ : a float used to filter the inter-temporal edges before the embedding (default = 0.1)
* _--scope_ : an int used to limit the temporal scope of the jaccard index while building the inter-temporal similarity matrix. By limiting the scope you will make Chronoblox more sensitive to local changes (default = 1; unlimited_scope = -1)

### Output files

Chronoblox layout produces two files : 

* `name_of_sequence-edges.csv` : this file contains informations related to inter-temporal and intra-temporal edges
* `name_of_sequence-blocks.csv` : this file contains informations related to the node groups

### Note

This version of Chronoblox only implements a [stochastic block model approach](https://graph-tool.skewed.de/static/doc/demos/inference/inference.html) in order to infer the node groups.


