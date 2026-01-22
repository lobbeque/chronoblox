<div><img width="300" src="https://github.com/lobbeque/chronoblox/blob/master/images/vault.svg"></div>

# Chronoblox: Chronoblox: Chronophotographic Graph Visualization

![Layout](https://img.shields.io/badge/Layout-Python-informational?style=flat&logo=python&color=6144b3)&nbsp;&nbsp;![Interface](https://img.shields.io/badge/Interface-Javascript-informational?style=flat&logo=javascript&color=6144b3)&nbsp;&nbsp;

Chronoblox generates a chronophotography of graph snapshots by embedding all temporal phases into a single, shared similarity space. This unified spatialization allows users to simultaneously perceive both local and global structural dynamics.

**Ressources** : [examples](https://lobbeque.github.io/chronoblox_examples/) | [publication](https://arxiv.org/abs/2405.07506)

## Chronoblox Layout

**Prerequisites** : You first need to install [graph-tool](https://graph-tool.skewed.de/)

**Other dependencies** : [networkx](https://networkx.org/) | [numpy](https://pypi.org/project/numpy/) | [pacmap](https://pypi.org/project/pacmap/) 

### Launch the script

#### With no graph snapshot

```shell
cd ./layout
python chronoblox.py
```

If you don't have your own sequence of snapshots, then by default Chronoblox will use the [board_directors](https://networks.skewed.de/net/board_directors) data set.

#### With your own graph snapshots

```shell
cd ./layout
python chronoblox.py --snapshots ~/path/to/your/snapshots*
```

#### Graph snapshots' requirements

Chronoblox takes a sequence of graph snapshots as input data. Graph snapshots must be in graph-tool `.gt` format. The graphs must contain the following [graph properties](https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.PropertyMap.html#graph_tool.PropertyMap) and vertex properties:

* Graph properties
  * `graph_phase` : a string that indicates the order of the  graph snapshots in the sequence. It can be a date or an Ord number. Graph snapshots will be sorted by `phase` during the embedding and visualization steps.
  * `graph_name` : a that indicates the name of the sequence. It will be used to name the output files.
* Vertex properties
  * `vid` : a unique string identifier for each vertex whatever the phase.
  * `vmeta` : a qualitative metadata or a label attached to each vertex. Metadata will be used to highlight the node groups in the interface. If not completed the default will be _NA_.

### Output files

Chronoblox layout produces two files : 

* `name_of_sequence-edges.csv` : this file contains informations related to inter-temporal and intra-temporal edges
* `name_of_sequence-blocks.csv` : this file contains informations related to the node groups

## Chronoblox Interface

**Dependencies** : [p5.js](https://p5js.org/) | [d3.js](https://d3js.org/) | [fontawesome](https://fontawesome.com/)

### Launch the interface

Go to `./layout` and open the file `chronoblox.html` with your favorite IDE.
In the function `preload()` :

```javascript
    function preload() {
      nodes = loadTable('../layout/board_directors_sbm_blocks.csv','csv','header')
      edges = loadTable('../layout/board_directors_sbm_edges.csv','csv','header')
      neuekabel = loadFont('fonts/NeueKabel.otf');
      neuekabelbold = loadFont('fonts/NeueKabel-Bold.otf');
      setShader("night")
    }
```

Replace the default blocks and edges paths with your own paths and files produced by the layout, such as :

```javascript
    function preload() {
      nodes = loadTable('~/path/to/the/blocks.csv','csv','header')
      edges = loadTable('~/path/to/the/edges.csv','csv','header')
      neuekabel = loadFont('fonts/NeueKabel.otf');
      neuekabelbold = loadFont('fonts/NeueKabel-Bold.otf');
      setShader("night")
    }
```
Then open the file `chronoblox.html` in your web browser.

## Funding

The research that was conducted to create the first version of this tool was funded by CNRS and the ERC Consolidator Grant [Socsemics](https://socsemics.huma-num.fr/) (grant #772743).

<div><img width="300" src="https://github.com/lobbeque/chronoblox/blob/master/images/socsemics.png"></div>






