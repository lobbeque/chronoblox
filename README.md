# Chronoblox

Chronoblox performs the chronophotography of a sequence of graphs. By using a single inter-temporal embedding space, Chronoblox lays out each snapshot in relation with the ones that precede or follow it. The motion of the whole sequence of graphs can thus be interpreted in terms micro to meso-structural evolution.

## Resources

* [examples](https://lobbeque.github.io/chronoblox_examples/)

## Layout

### Python dependencies

* [graph-tool](https://graph-tool.skewed.de/)
* [numpy](https://numpy.org/)
* [pacmap](https://github.com/YingfanWang/PaCMAP)
* [pecanpy](https://github.com/krishnanlab/PecanPy)
* [gensim.models](https://radimrehurek.com/gensim/models/word2vec.html)
* [sklearn.decomposition](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

### Run the script 

If you have installed graph-tool using [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) then first run `conda activate gt`.  
Then simply run `python3 chronoblox.py`.
If you don't specify the path to your own snapshots in argument such as `python3 chronoblox.py --snapshots ./path_to_your_snapshots.gt` then chronoblox will visualize the [board_directors](https://networks.skewed.de/net/board_directors) data set by default.

### Snapshots' requirements

Chronoblox takes a sequence of graph snapshots as input data. Graphs must be in graph-tool `.gt` format. The graphs must contain the following [graph properties](https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.PropertyMap.html#graph_tool.PropertyMap) :

* _phase_ : a string that indicates the order of the graph in the sequence, graphs will be ordered by _phase_
* _name_ : a string that indicates the name of the sequence

The vertices must contain the following vertex properties :

* _vid_ : a unique string identifier for each vertex whatever the phase.

The vertices can contain the following vertex properties :

* _vmeta_ : a metadata or a label attached to the vertices, the metadata will be projected on the chronophotographies in the interface

### Other arguments

* _--threshold_ : a float used to filter the inter-temporal edges before the embedding (default = 0.1)
* _--scope_ : an int used to limit the temporal scope of the jaccard index while building the inter-temporal similarity matrix. By limiting the scope you will make Chronoblox more sensitive to local changes (default = 1; unlimited_scope = -1)

### Output files

Chronoblox layout produces two files : 

* `name_of_sequence-edges.csv` : this file contains informations related to inter-temporal and intra-temporal edges
* `name_of_sequence-blocks.csv` : this file contains informations related to the node groups

### Note

This version of Chronoblox only implements a [stochastic block model approach](https://graph-tool.skewed.de/static/doc/demos/inference/inference.html) in order to infer the node groups.


