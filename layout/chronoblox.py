#######
# chronoblox.py
# description : Chronoblox performs the chronophotography of a sequence of snapshots
# licence : AGPL + CECILL v3
# paper : https://arxiv.org/abs/2405.07506
#######

# conda activate gt


####
## import
####


from contextlib import suppress

import argparse
import glob
import graph_tool.all as gt
import functools
import numpy as np  
import pacmap
import networkx as nx
import math

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from community import community_louvain

from node2vec import Node2Vec

####
## args
####


parser = argparse.ArgumentParser( description='Process the chronophotography of a temporal sequence of meta-graph snapshots'
	                            , usage='conda must be activated before using Chronoblox => conda activate gt',)
parser.add_argument('--snapshots'
				   , nargs=1
				   , default='board_directors', help='path to the snapshots (default: load the board_directors data set)')
parser.add_argument('--temporal_scope'
	               , type=int
	               , default=1, help='limit the temporal scope when computing jaccard index between node groups; unlimited scope is -1')
parser.add_argument('--similarity_threshold'
	               , type=float
	               , default=0.1, help='filter pointless inter-temporal similarity edges before embedding')
parser.add_argument('--hhi_filter'
	               , choices=['true','false']
	               , default=['true'], help='filter pointless synchronic edges with a hhi filter')
parser.add_argument('--grouping_strategy'
	               , choices=['none','sbm','louvain','by_label']
	               , default='none', help='choose a strategy to group individual nodes')
parser.add_argument('--group_size'
	               , type=int
	               , default=-1, help='filter small node groups; no filter is -1')
parser.add_argument('--group_metadata_strategy'
	               , choices=['majority','most_frequent']
	               , default=['majority'], help='choose a strategy to aggregate individual metadata at the node groups level')
parser.add_argument('--group_label_strategy'
	               , choices=['most_frequent','most_central']
	               , default='most_central', help='choose a strategy to label node groups')

args = parser.parse_args()


####
## get the input snapshots
#### 


def getGraphPhase(snapshot) :
	with suppress(KeyError): return snapshot.gp["phase"]

def getGraphName(snapshot) :
	with suppress(KeyError): return snapshot.gp["name"]

def loadSnapshot(path) :
	return gt.load_graph(path)

snapshots = []
graph_name = ""

if args.snapshots != "board_directors" :
	# load your own snapshots
	print("\nloading your own snapshots ...")
	paths = glob.glob(args.snapshots[0] + "*.gt")
	snapshots = list(map(lambda p: loadSnapshot(p), paths))
	snapshots.sort(key=getGraphPhase)
	graph_name = getGraphName(snapshots[0])
	print("\nsnapshots loaded")
else :
	# load the board_directors data set
	print("\nloading default data set ...")
	phases = ["2002-06-01","2003-06-01","2004-06-01","2005-06-01","2006-06-01","2007-06-01","2008-06-01","2009-06-01","2010-06-01","2011-06-01"]
	cpt = 0
	for phase in phases :
		g = gt.collection.ns["board_directors/net1m_" + phase]
		g.gp["phase"] = g.new_graph_property("string")
		g.gp["phase"] = phase
		snapshots.append(g)
		cpt += 1
	graph_name = "board_directors"
	print("\nboard_directors loaded")


####
## output files
#### 


def toEdgeFile(file,s,t,w,phase,edge_type,component_id) :
	file.write(s + ',' + t + ',' + w + ',' + phase  + ',' + edge_type + ',' + component_id + '\n')

output_edges = open("./" + graph_name + "_" + args.grouping_strategy + "_edges.csv", "w")
output_edges.write("source,target,weight,phase,type,sync_component\n")	

def toBlockFile(file,b_id,phase,size,sync_component_id,diac_component_id,lineage_size,meta,label,x,y,z) :
	file.write(b_id + ',' + phase + ',' + size + ',' + sync_component_id +',' + diac_component_id + ',' + lineage_size + ',' + meta + ',' + label + ',' + x + ',' + y + ',' + z + '\n')	

output_blocks = open("./" + graph_name + "_" + args.grouping_strategy + "_blocks.csv", "w")
output_blocks.write("id,phase,size,sync_component,diac_component,lineage_size,meta,label,x,y,z\n")


####
## functions
####


def groupEdges (edges) :
	# group the edges by source and target
	grouped = {}
	for e in edges.keys() :
		s = e[0]
		t = e[1]
		w = edges[e]['w']
		if s in grouped.keys() :
			grouped[s].append([s,t,w,'out'])
		else :
			grouped[s] = [[s,t,w,'out']]
		if t in grouped.keys() :
			grouped[t].append([s,t,w,'in'])
		else :
			grouped[t] = [[s,t,w,'in']]	
	return grouped.values()	

def hhiFilter(edges) :
	# compute the Herfindahl-Hirschman Index
	si = functools.reduce(lambda acc,edge: acc + edge[2], edges, 0)
	hhi = functools.reduce(lambda acc,edge: acc + (edge[2] / si)**2, edges, 0)
	edges.sort(key=lambda e:e[2], reverse=True)
	return edges[:round(1/hhi)]	

def intersect(l1, l2) :
	# compute the intersection of two lists
	return list(set(l1) & set(l2))

def union(l1, l2) :
	# compute the union of two lists
	return set(l1).union(set(l2))	

def jaccard(a,b) :
	# compute the Jaccard Index 
	return len(intersect(a,b))/len(union(a,b))	

def cosine(a,b) :
	# compute the Cosine similarity
	return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def squaredEuclidianDistance(a,b) :
	# compute the squared euclidian distance
	dist = (((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))
	return dist

def euclidianDistance (a,b) :
	# compute the euclidian distance as a similarity fonction [0,1]
	sim = 1 / (1 + (squaredEuclidianDistance(a,b) ** 0.5))
	return sim

def rbfKernel(a,b) :	
	dist = squaredEuclidianDistance(a,b)
	sigma = 1
	gamma = 1 / (2 * (sigma ** 2))
	return math.exp(-1 * gamma * dist)

def predicatePartition (l1,ll) :
	# split a list in two regarding the intersect predicate	
	connected = []
	not_connected = []
	for l2 in ll :
		if not intersect(l1,l2) :
			not_connected += [l2]
		else :
			connected = list(set(connected + l2))
	return [connected,not_connected]	

def connectedComponents (components,cur,graph) :
	# find connected components from a list of edges
	if not graph :
		if not cur :
			return components
		else : 
			return components + [cur]
	else :
		parts = predicatePartition(cur,graph)
		if not parts[0] :
			components = components + [cur]
			return connectedComponents(components,graph[0],graph[1:])
		else :
			return connectedComponents(components,list(set(cur + parts[0])),parts[1])	


####
## process the sequence of graph snapshots
####


def getVertexId (snapshot,v) :
	with suppress(KeyError): return snapshot.vp.vid[v]

def getVertexVector (snapshot,v) :
	with suppress(KeyError): return snapshot.vp.vgroup[v]

def getVertexSize (snapshot,v) :
	with suppress(KeyError): return snapshot.vp.vsize[v]	

def getVertex (snapshot,target_id) :
	for v in snapshot.vertices() :
		v_id = getVertexId(snapshot,v)
		if v_id == target_id :
			return v

def getVertexMetaType (snapshot,v) :
	try:
		return snapshot.vp.vmeta.value_type()
	except Exception as e:
		return 'NA'	

def getVertexMeta (snapshot,v) :
	try:
		return snapshot.vp.vmeta[v]
	except Exception as e:
		return 'NA'

def getVertexLabel (snapshot,v) :
	try:
		return str(snapshot.vp.vlabel[v])
	except Exception as e:
		return str(snapshot.vp.vid[v])

def snapshotToSBMPartitions (snapshot) :
	# use the sbm method to create node groups
	return gt.minimize_blockmodel_dl(snapshot, state_args=dict(deg_corr=True,recs=[snapshot.ep.weight],rec_types=["discrete-geometric"]))

def snapshotToLouvainPartitions (snapshot) :
	# use the louvain method to create node groups
	gx = nx.Graph()
	for v in snapshot.vertices() :
		v_id = getVertexId(snapshot,v)
		gx.add_node(v_id)
	for e in snapshot.edges():
		s = snapshot.vp.vid[e.source()]
		t = snapshot.vp.vid[e.target()]
		gx.add_edge(s, t, weight=snapshot.ep.weight[e])
	partitions = community_louvain.best_partition(gx,weight='weight')	
	vlouvain = snapshot.new_vertex_property("int")
	snapshot.vp.vlouvain = vlouvain
	for v in snapshot.vertices() :
		v_id = getVertexId(snapshot,v)
		snapshot.vp.vlouvain[v] = partitions[v_id]
	node_grouping_labels = snapshot.vp.vlouvain
	state = gt.BlockState(snapshot)
	state.set_state(node_grouping_labels)
	return state	

def snapshotToLabelPartitions (snapshot) :
	# use existing node labels to create node groups
	node_grouping_labels = snapshot.vp.vgroup
	state = gt.BlockState(snapshot)
	state.set_state(node_grouping_labels)
	return state

def snapshotToIdentityPartitions (snapshot) :
	# consider each node as a self-contained group
	state = gt.BlockState(snapshot)
	state.set_state(snapshot.vp.vid)
	return state

def snapshotToPartition (snapshot,strategy) :
	# select the grouping strategy
	if   strategy == "none" :
		return snapshotToIdentityPartitions(snapshot)
	elif strategy == "sbm" :
		return snapshotToSBMPartitions(snapshot)
	elif strategy == "by_label":
		return snapshotToLabelPartitions(snapshot)
	elif strategy == "louvain":
		return snapshotToLouvainPartitions(snapshot)		

phases = []
blocks_to_meta   = {}
blocks_to_label  = {}
root_label_to_blocks = {}
vertex_to_vector = {}
vertex_to_size   = {}
blocks_to_synchronic_components = {}
sequence_of_blocks = {}
flow_size = {}

for snapshot in snapshots :

	# inferring a partition from a snapshot

	phase = getGraphPhase(snapshot)
	phases.append(phase)

	print("\n####")
	print(phase)
	print(snapshot)

	partition = snapshotToPartition(snapshot,args.grouping_strategy)

	blocks = {}

	for v in snapshot.vertices() :

		# [block] 1) aggregate the vertices at the block level

		v_id    = getVertexId(snapshot,v)
		v_meta  = getVertexMeta(snapshot,v)
		v_label = getVertexLabel(snapshot,v)
		v_root_label = v_label

		b_id = str(partition.get_blocks()[v]) + "_" + phase

		if args.grouping_strategy == "none" : 
			vertex_to_vector[b_id] = getVertexVector(snapshot,v)
			vertex_to_size[b_id]   = getVertexSize(snapshot,v)
			v_root_label = (v_label.split('_'))[0]
			if v_root_label in root_label_to_blocks.keys() :
				root_label_to_blocks[v_root_label].append(b_id)
			else :
				root_label_to_blocks[v_root_label] =[b_id]

		if b_id in blocks.keys() :
			blocks[b_id].append(v_id)
			blocks_to_meta[b_id].append(v_meta)
			blocks_to_label[b_id].append(v_label)
		else :
			blocks[b_id] = [v_id]
			blocks_to_meta[b_id]  = [v_meta]
			blocks_to_label[b_id] = [v_label]

	# [block] 2) maybe filter the small blocks

	if (args.grouping_strategy != "none") and (args.group_size > 0) :
		blocks = {k: v for k, v in blocks.items() if len(v) > args.group_size}

	for b_id in blocks.keys() :	
		sequence_of_blocks[b_id] = blocks[b_id]

	# [metadata] aggregate the metadata to the block level

	if (args.grouping_strategy == "none") :
		# if no grouping strategy, we don't need to aggregate the metadata
		for b_id in blocks.keys() :	
			 blocks_to_meta[b_id] = blocks_to_meta[b_id][0]
	else :
		for b_id in blocks.keys() :	
			metas = blocks_to_meta[b_id]
			freq_max = 0
			most_freq_meta = ''
			for meta in metas :
				if (metas.count(meta) > freq_max) :
					# we use a simple most frequent strategy
					freq_max = metas.count(meta)
					most_freq_metal = meta
			if (args.group_metadata_strategy == "most_frequent") :
				blocks_to_meta[b_id] = most_freq_meta
			if (args.group_metadata_strategy == "majority") : 
				if freq_max > (len(metas) / 2) :
					blocks_to_meta[b_id] = most_freq_meta
				else :
					blocks_to_meta[b_id] = "mixed"

	# [label] set a label to each block

	for b_id in blocks.keys() :
		if   (args.grouping_strategy == "none") :
			# if no grouping strategy, we don't need to aggregate the metadata
			blocks_to_label[b_id] = blocks_to_label[b_id][0]	
		elif (args.group_label_strategy == "most_frequent") :
			labels = blocks_to_label[b_id]
			freq_max = 0
			most_freq_label = 'na'
			for label in labels :
				if (labels.count(label) > freq_max) :
					freq_max = labels.count(label)
					most_freq_label = label				
			blocks_to_label[b_id] = most_freq_label
		elif (args.group_label_strategy == "most_central") :
			gf = gt.GraphView(snapshot, vfilt=lambda v: getVertexId(snapshot,v) in blocks[b_id])
			centrality,ep = gt.betweenness(gf)
			most_central_label = 'na'
			betweenness_max = 0
			for v in gf.vertices() :
				if centrality[v] > betweenness_max :
					betweenness_max = centrality[v]
					most_central_label = getVertexLabel(gf,v)
			blocks_to_label[b_id] = most_central_label

	# set up edges for embedding
	
	print(str(len(blocks.keys())) + ' blocks')

	meta_graph  = partition.get_bg()
	sync_matrix = partition.get_matrix().toarray()

	# [sync_edges] 1) get intra-temporal edges

	sync_edges = {}
	
	for e in meta_graph.edges() :
		s = meta_graph.vertex_index[e.source()]
		t = meta_graph.vertex_index[e.target()]
		# remove self edge
		if (s == t) or (sync_matrix[s][t] == 0):
			continue
		# ensure that edges are undirected
		edge = [str(s) + "_" + phase, str(t) + "_" + phase]
		edge.sort()
		edge = tuple(edge)
		w = sync_matrix[s][t]
		if edge in sync_edges.keys() :
			sync_edges[edge]['w'] += w
		else :
			sync_edges[edge] = {'w':w,'shhi':0} 

	# [sync_edges] 2) filter the intra-temporal edges by using a sHHI
	
	# https://en.wikipedia.org/wiki/Herfindahl%E2%80%93Hirschman_index

	grouped_sync_edges = groupEdges(sync_edges)		

	for edges in grouped_sync_edges :
		filtered_edges = hhiFilter(edges)
		for edge in filtered_edges :
			sync_edges[(edge[0],edge[1])]["shhi"] += 1

	# [sync_edges] 3) export the intra-temporal edges	

	sync_edges_list = []	

	for edge in sync_edges.keys() :
		if (sync_edges[edge]["shhi"] != 2) or (args.hhi_filter == "false") :
			# each edge must satisfy the sHHI test
			continue
		sync_edges_list.append([edge[0],edge[1]])
		# then we export the intra-temporal edges
		toEdgeFile(output_edges,str(edge[0]),str(edge[1]),str(sync_edges[edge]["w"]),phase,'sync','-1')

	# [sync_edges] 4) find synchronic connected components

	if len(sync_edges_list) > 0 :
		sync_connected_components = connectedComponents([],sync_edges_list[0],sync_edges_list[1:])
		cmp_cpt = 0
		for component in sync_connected_components :
			for b_id in component :
				# synchronic connected components' id will be transfered to the blocks
				blocks_to_synchronic_components[b_id] = cmp_cpt
			cmp_cpt += 1

temporal_gap = int(phases[1]) - int(phases[0])

		
####
## create the inter-temporal similarity matrix
####


print('\nbuild the inter-temporal similarity matrix')

def areInScope(bi_t,bj_t) :
	bi_t_idx = phases.index(bi_t)
	bj_t_idx = phases.index(bj_t)
	if (args.temporal_scope < 0) :
		return True
	else :
		return (abs(bi_t_idx - bj_t_idx) <= args.temporal_scope)

blocks_to_diachronic_components = {}
filtered_diac_edges    = {}
best_in_out_diac_edges = {}
mat = []

def isDirectAncestor(bi_t,bj_t) :
	bi_t_idx = phases.index(bi_t)
	bj_t_idx = phases.index(bj_t)
	return (bi_t_idx == (bj_t_idx - 1))

for bi in sequence_of_blocks.keys() :
	
	row = []
	bi_t = bi.split('_')[1]
	bi_v = sequence_of_blocks[bi]
	
	for bj in sequence_of_blocks.keys() :

		# [matrix] 1) for each inter-temporal pair of blocks 
		
		bj_t = bj.split('_')[1]
		bj_v = sequence_of_blocks[bj]
		
		# [matrix] 2) populate the similarity matrix 

		sim = 0

		if args.grouping_strategy == "none" :
			if len(vertex_to_vector[bi]) > 2 :
				sim = cosine(vertex_to_vector[bi],vertex_to_vector[bj])
			else :
				sim = rbfKernel(vertex_to_vector[bi],vertex_to_vector[bj])
				# sim = euclidianDistance(vertex_to_vector[bi],vertex_to_vector[bj])
		else :
			sim = jaccard(bi_v,bj_v)
		
		row.append(sim)
		
		# [flow size] compute the flow size for visualization
		
		if (sim > 0) and (isDirectAncestor(bi_t,bj_t)) :
			if bj in flow_size.keys() :
				flow_size[bj] += sim * len(bi_v)
			else :
				flow_size[bj] =  sim * len(bi_v)

			# [diac_edges] 1) put the (t-1,t) inter-temporal edges aside to compute the inter-temporal lineages 
			
			if (sim >= args.similarity_threshold) :
				# these edges will be the only visible in the interface 
				filtered_diac_edges[(bi,bj)] = {'w':sim,'shhi':0}

	mat.append(row)

# [diac_edges] 2) filter the visible inter-temporal edges by using a sHHI	

filtered_grouped_diac_edges = groupEdges(filtered_diac_edges)

for edges in filtered_grouped_diac_edges :
	filtered_edges = hhiFilter(edges)
	for edge in filtered_edges :
		filtered_diac_edges[(edge[0],edge[1])]["shhi"] += 1

# [diac_edges] 3) find diachronic connected components

diac_edges_list = []
for edge in filtered_diac_edges.keys() :
	if (filtered_diac_edges[edge]["shhi"] == 2) :
		diac_edges_list.append([edge[0],edge[1]])

# diachronic connected components' id will be transfered to the blocks

if args.grouping_strategy == "none" :
	diac_connected_components = list(root_label_to_blocks.values())
else :
	diac_connected_components = connectedComponents([],diac_edges_list[0],diac_edges_list[1:])

cmp_cpt = 0
for component in diac_connected_components :
	for block in component :
		blocks_to_diachronic_components[block] = cmp_cpt
	cmp_cpt += 1

# [diac_edges] 4) export the inter-temporal edges
if args.grouping_strategy == "none" :
	for component in diac_connected_components :
		for i in range(0,len(component)) :
			if i + 1 <= len(component) - 1 :
				date_inf = int((component[i]).split('_')[1])
				date_sup = int((component[i + 1]).split('_')[1])
				if (date_sup - date_inf) == temporal_gap :				
					component_id = blocks_to_diachronic_components[component[i]]
					toEdgeFile(output_edges,component[i],component[i + 1],'1',str(date_inf),'diac',str(component_id))
else :
	for edge in filtered_diac_edges.keys() :
		if (filtered_diac_edges[edge]["shhi"] == 2) :
			component_id = blocks_to_diachronic_components[edge[0]]
			toEdgeFile(output_edges,edge[0],edge[1],str(filtered_diac_edges[edge]["w"]),edge[0].split('_')[1],'diac',str(component_id))


####
## embed the inter-temporal similarity matrix
####


print('\nembed the sequence of meta-graphs')

mat = np.array(mat)
b_ids = list(sequence_of_blocks.keys())

# [embedding] 1) export the edges to a temporary file (mandatory by PecanPy)

inter_temporal_graph = nx.Graph()
inter_temporal_edges = []

for i in range(len(mat)) :
	for j in range(i,len(mat)) :
		if j != i :
			if mat[i][j] > 0 :
				bi = b_ids[i]
				bi_t = bi.split('_')[1]
				bj = b_ids[j]
				bj_t = bj.split('_')[1]
				# scope can be limited here
				if (areInScope(bi_t,bj_t)) :
					inter_temporal_edges.append((bi,bj,mat[i][j]))

inter_temporal_graph.add_weighted_edges_from(inter_temporal_edges)

# [embedding] 2) embed the inter_temporal_graph via node2vec

node2vec = Node2Vec(inter_temporal_graph, dimensions=20, walk_length=16, num_walks=100)
embedding = node2vec.fit(window=10, min_count=1)

####
## [chronophotographic projection]
####


print('\nproject the embedding with pacmap ...')	

# [projection] 1) use PaCMAP to project the embedding on 2D visualization space

projector = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 
projection_2D = projector.fit_transform(embedding.wv.vectors, init="pca")

xs = projection_2D[:, 0]
ys = projection_2D[:, 1]


###
# [PCA reduction]
###


print('\nreduce the 2D coordinates with PCA ...')

# [projection] 2) use a PCA to reduce the 2D PaCMAP coordinates for building an alluvial chart

pca = PCA(n_components=1)

projection_1D = pca.fit_transform(projection_2D)

zs = projection_1D[:, 0]


###
# [export the blocks]
###

for i in range(len(embedding.wv.index_to_key)) :
	
	block = embedding.wv.index_to_key[i]

	lineage_size = 0
	diac_component_id = -1
	sync_component_id = -1	
	phase = block.split('_')[1]
	if args.grouping_strategy == "none" :
	    b_size = vertex_to_size[block]
	else :
		b_size = len(sequence_of_blocks[block])

	# [blocks] 1) find the corresponding synchronic component
	
	if (block in blocks_to_synchronic_components) :
		sync_component_id = blocks_to_synchronic_components[block]
	
	# [blocks] 2) find the corresponding diachronic component
	
	if (block in blocks_to_diachronic_components) :
		diac_component_id = blocks_to_diachronic_components[block]	

	# [blocks] 3) find the corresponding diachronic component
	
	if (block in flow_size) :
		lineage_size = flow_size[block]		

	toBlockFile(output_blocks
		      , block
		      , phase
		      , str(b_size)
		      , str(sync_component_id)
		      , str(diac_component_id)
		      , str(lineage_size)
		      , blocks_to_meta[block]
		      , blocks_to_label[block]		      
		      , str(xs[i])
		      , str(ys[i])
		      , str(zs[i]))		

print('\nReady for visualization')
