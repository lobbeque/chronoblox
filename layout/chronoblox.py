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

import argparse
import glob
import graph_tool.all as gt
import functools
import numpy as np  
import pacmap
import networkx as nx
import math
import sys

from contextlib import suppress
from datetime import datetime
from community import community_louvain

####
## args
####


parser = argparse.ArgumentParser( description='Process the chronophotography of a sequence of graph snapshots'
	                            , usage='conda must be activated before using Chronoblox => conda activate gt',)
parser.add_argument('--snapshots'
				   , nargs=1
				   , default='board_directors', help='path to the snapshots (default: load the board_directors data set)')
parser.add_argument('--grouping_strategy'
	               , choices=['sbm','louvain']
	               , default='sbm', help='choose a strategy to group individual nodes')
parser.add_argument('--group_size'
	               , type=int
	               , default=30, help='filter small node groups; no filter is -1')
parser.add_argument('--group_metadata_strategy'
	               , choices=['majority','most_frequent','mean']
	               , default='majority', help='choose a strategy to aggregate individual metadata at the node groups level')
parser.add_argument('--group_label_strategy'
	               , choices=['most_frequent','most_central']
	               , default='most_central', help='choose a strategy to label node groups')

args = parser.parse_args()


####
## graph getters
#### 


def getGraphPhase(snapshot) :
	with suppress(KeyError): return snapshot.gp["graph_phase"]

def getGraphName(snapshot) :
	with suppress(KeyError): return snapshot.gp["graph_name"]

def loadSnapshot(path) :
	return gt.load_graph(path)


####
## vertex getters
####


def getVertexId (snapshot,v) :
	with suppress(KeyError): return str(snapshot.vp.vid[v])

def getVertexVector (snapshot,v) :
	with suppress(KeyError): return snapshot.vp.vgroup[v]

def getVertexSize (snapshot,v) :
	with suppress(KeyError): return snapshot.vp.vsize[v]	

def getVertex (snapshot,target_id) :
	for v in snapshot.vertices() :
		v_id = getVertexId(snapshot,v)
		if v_id == target_id :
			return v

def getVertexMeta (snapshot,v) :
	try:
		return snapshot.vp.vmeta[v]
	except Exception as e:
		return 'NA'	


####
## get the input snapshots
#### 


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
	for phase in phases :
		g = gt.collection.ns["board_directors/net1m_" + phase]
		g.gp["graph_phase"] = g.new_graph_property("string")
		g.gp["graph_phase"] = phase
		e_weight = g.new_edge_property("int")
		g.ep.weight = e_weight 
		for e in g.edges() :	
			g.ep.weight[e] = 1	
		v_meta = g.new_vertex_property("string")	
		g.vp.vmeta = v_meta 
		for v in g.vertices() :	
			if (g.vp.gender[v] == 1) :
				g.vp.vmeta[v] = "male"	
			else :
				g.vp.vmeta[v] = "female"	
		snapshots.append(g)
	graph_name = "board_directors"
	print("\nboard_directors loaded")


####
## output files
#### 


def toEdgeFile(file,s,t,w,phase,edge_type,component_id) :
	file.write(s + ',' + t + ',' + w + ',' + phase  + ',' + edge_type + ',' + component_id + '\n')

output_edges = open("./" + graph_name + "_" + args.grouping_strategy + "_edges.csv", "w")
output_edges.write("source,target,weight,phase,type,diac_component\n")	

def toBlockFile(file,b_id,phase,size,diac_component_id,meta,x,y) :
	file.write(b_id + ',' + phase + ',' + size +',' + diac_component_id + ',' + str(meta) + ',' + x + ',' + y + '\n')	

output_blocks = open("./" + graph_name + "_" + args.grouping_strategy + "_blocks.csv", "w")
output_blocks.write("id,phase,size,diac_component,meta,x,y\n")


####
## similarity functions
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

def hhiFilterEdges(edges) :
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
## partition functions
####


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
		s = getVertexId(snapshot,e.source())
		t = getVertexId(snapshot,e.target())
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

def snapshotToPartition (snapshot,strategy) :
	# select the grouping strategy
	if   strategy == "sbm" :
		return snapshotToSBMPartitions(snapshot)
	elif strategy == "louvain":
		return snapshotToLouvainPartitions(snapshot)		


####
## process the sequence of graph snapshots
####


phases = []
blocks_to_meta     = {}
sequence_of_blocks = {}

for snapshot in snapshots :

	phase = getGraphPhase(snapshot)
	phases.append(phase)

	print("\n####")
	print(phase)
	print(snapshot)

	# [block] 1) inferring a partition from a snapshot

	partition = snapshotToPartition(snapshot,args.grouping_strategy)

	blocks = {}

	for v in snapshot.vertices() :

		# [block] 2) aggregate the vertices at the block level

		v_id    = getVertexId(snapshot,v)
		v_meta  = getVertexMeta(snapshot,v)
		
		b_id = str(partition.get_blocks()[v]) + "_" + phase

		if b_id in blocks.keys() :
			blocks[b_id].append(v_id)
			blocks_to_meta[b_id].append(v_meta)
		else :
			blocks[b_id] = [v_id]
			blocks_to_meta[b_id]  = [v_meta]

	# [block] 3) maybe filter the small blocks

	if (args.group_size > 0) :
		blocks = {k: v for k, v in blocks.items() if len(v) > args.group_size}

	for b_id in blocks.keys() :	
		sequence_of_blocks[b_id] = blocks[b_id]

	# [metadata] aggregate the metadata to the block level

	for b_id in blocks.keys() :	
		metas = blocks_to_meta[b_id]
		if (args.group_metadata_strategy == "mean") :
			blocks_to_meta[b_id] = sum(metas) / len(metas)
		else :
			freq_max = 0
			most_freq_meta = ''
			for meta in metas :
				if (metas.count(meta) > freq_max) :
					# we use a simple most frequent strategy
					freq_max = metas.count(meta)
					most_freq_meta = meta
			if (args.group_metadata_strategy == "most_frequent") :
				blocks_to_meta[b_id] = most_freq_meta
			if (args.group_metadata_strategy == "majority") : 
				if freq_max > (len(metas) / 2) :
					blocks_to_meta[b_id] = most_freq_meta
				else :
					blocks_to_meta[b_id] = "mixed"

	# [sync_edges] 1) aggregate the edges at the block level

	meta_graph  = partition.get_bg()
	sync_matrix = partition.get_matrix().toarray()

	# [sync_edges] 2) get intra-temporal edges

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

	# [sync_edges] 3) filter the intra-temporal edges by using a sHHI	
	# https://en.wikipedia.org/wiki/Herfindahl%E2%80%93Hirschman_index

	grouped_sync_edges = groupEdges(sync_edges)		
	for edges in grouped_sync_edges :
		filtered_edges = hhiFilterEdges(edges)
		for edge in filtered_edges :
			sync_edges[(edge[0],edge[1])]["shhi"] += 1		

	# [sync_edges] 4) export the intra-temporal edges		

	sync_edges_list = []	
	for edge in sync_edges.keys() :
		if (sync_edges[edge]["shhi"] != 2) :
			# each edge must satisfy the sHHI test
			continue
		sync_edges_list.append([edge[0],edge[1]])
		# then we export the intra-temporal edges
		toEdgeFile(output_edges,str(edge[0]),str(edge[1]),str(sync_edges[edge]["w"]),phase,'sync','-1')	

		
####
## create the inter-temporal similarity matrix
####


print('\nbuild the inter-temporal similarity matrix')

blocks_to_diachronic_components = {}
filtered_diac_edges = {}
mat = []

def isDirectAncestor(bi_t,bj_t) :
	bi_t_idx = phases.index(bi_t)
	bj_t_idx = phases.index(bj_t)
	return (bi_t_idx == (bj_t_idx - 1))

for bi in sequence_of_blocks.keys() :
	
	row = []
	bi_t = bi.split('_')[1]
	bi_name = bi.split('_')[0]
	bi_v = sequence_of_blocks[bi]
	
	for bj in sequence_of_blocks.keys() :

		# [matrix] 1) for each inter-temporal pair of blocks 
		
		bj_t = bj.split('_')[1]
		bj_name = bj.split('_')[0]		
		bj_v = sequence_of_blocks[bj]
		
		# [matrix] 2) populate the similarity matrix 
		
		sim = jaccard(bi_v,bj_v)

		row.append(sim)
		
		if (sim > 0) and (isDirectAncestor(bi_t,bj_t)) :

			# [diac_edges] 1) put the (t-1,t) inter-temporal edges aside to compute the inter-temporal lineages 
			
			filtered_diac_edges[(bi,bj)] = {'w':sim,'shhi':0}

	mat.append(row)

# [diac_edges] 2) filter the visible inter-temporal edges by using a sHHI	

filtered_grouped_diac_edges = groupEdges(filtered_diac_edges)

for edges in filtered_grouped_diac_edges :
	filtered_edges = hhiFilterEdges(edges)
	for edge in filtered_edges :
		filtered_diac_edges[(edge[0],edge[1])]["shhi"] += 1

# [diac_edges] 3) find diachronic connected components

diac_edges_list = []
for edge in filtered_diac_edges.keys() :
	if (filtered_diac_edges[edge]["shhi"] == 2) :
		diac_edges_list.append([edge[0],edge[1]])

# diachronic connected components' id will be transfered to the blocks

diac_connected_components = connectedComponents([],diac_edges_list[0],diac_edges_list[1:])

cmp_cpt = 0
for component in diac_connected_components :
	for block in component :
		blocks_to_diachronic_components[block] = cmp_cpt
	cmp_cpt += 1

# [diac_edges] 4) export the inter-temporal edges
for edge in filtered_diac_edges.keys() :
	if (filtered_diac_edges[edge]["shhi"] == 2) :
		component_id = blocks_to_diachronic_components[edge[0]]
		toEdgeFile(output_edges,edge[0],edge[1],str(filtered_diac_edges[edge]["w"]),edge[0].split('_')[1],'diac',str(component_id))

####
## [chronophotographic projection]
####

def areInScope(bi_t,bj_t) :
	bi_t_idx = phases.index(bi_t)
	bj_t_idx = phases.index(bj_t)
	return (abs(bi_t_idx - bj_t_idx) <= 1)

print('\nembed and project the matrix with pacmap ...')	

# 1) prepare the matrix

b_ids = list(sequence_of_blocks.keys())
vectors = []

output_matrix = open("./" + graph_name + "_matrix.csv", "w")
output_matrix.write("source,target,weight\n")

for i in range(len(mat)) :
	bi = b_ids[i]	
	bi_t = bi.split('_')[1]
	vector = []
	for j in range(len(mat)) :
		bj = b_ids[j]
		bj_t = bj.split('_')[1]
		if (areInScope(bi_t,bj_t)) :
			vector.append(mat[i][j])
		else :
			vector.append(0)
		output_matrix.write(bi + ',' + bj + ',' + str(mat[i][j]) + '\n')
	vectors.append(vector)

# 2) prune the matrix values with hhi

def hhiFilterVector(vector) :
	si = sum(vector)
	hhi = functools.reduce(lambda acc,v: acc + (v / si)**2, vector, 0)
	sorted_vector = sorted(vector,reverse=True)
	idx = round(1/hhi)
	return sorted_vector[idx]

output_matrix_pruned = open("./" + graph_name + "_matrix_pruned.csv", "w")
output_matrix_pruned.write("source,target,weight\n")

for i in range(len(mat)) :
	bi  = b_ids[i]
	thr = hhiFilterVector(vectors[i])
	vector_pruned = []
	for j in range(len(mat)) :
		bj = b_ids[j]
		if vectors[i][j] > thr :
			vector_pruned.append(vectors[i][j])
			output_matrix_pruned.write(bi + ',' + bj + ',' + str(vectors[i][j]) + '\n')
		else :
			vector_pruned.append(0)
			output_matrix_pruned.write(bi + ',' + bj + ',' + str(0) + '\n')
	vectors[i] = vector_pruned

vectors = np.array(vectors)	

# 3) use PaCMAP to project the embedding on 2D visualization space
 
projector = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2)
projection_2D = projector.fit_transform(vectors, init="pca")

xs = projection_2D[:, 0]
ys = projection_2D[:, 1]

###
# [export the blocks]
###

for i in range(len(vectors)) :
	
	block = b_ids[i]
	diac_component_id = -1
	phase = block.split('_')[1]
	b_size = len(sequence_of_blocks[block])
	
	# [blocks] 2) find the corresponding diachronic component
	
	if (block in blocks_to_diachronic_components) :
		diac_component_id = blocks_to_diachronic_components[block]	

	# [blocks] 3) find the corresponding diachronic component

	toBlockFile(output_blocks
		      , block
		      , phase
		      , str(b_size)
		      , str(diac_component_id)
		      , blocks_to_meta[block]		      
		      , str(xs[i])
		      , str(ys[i]))		

print('\nReady for visualization')
