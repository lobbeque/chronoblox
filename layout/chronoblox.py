#######
# chronoblox.py
# description : Chronoblox performs the chronophotography of a sequence of snapshots
# licence : AGPL + CECILL v3
# author : quentin lobbé - quentin.lobbe@gmail.com - CSS Team, Marc Bloch Center, Berlin
# funding : ERC Socsemics
# paper :  
#######


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

from pecanpy import pecanpy as node2vec
from gensim.models import Word2Vec,KeyedVectors
from sklearn.decomposition import PCA


####
## args
####


parser = argparse.ArgumentParser( description='Process the chronophotography of a temporal sequence of meta-graph snapshots'
	                            , usage='conda must be activated before using Chronoblox => conda activate gt',)
parser.add_argument('--snapshots'
				   , nargs=1
				   , default='board_directors', help='path to the snapshots (default: load the board_directors data set)')
parser.add_argument('--scope'
	               , type=int, nargs=1
	               , default=1, help='limit the temporal scope of the Jaccard index; unlimited scope is -1')
parser.add_argument('--threshold'
	               , type=float, nargs=1
	               , default=0.1, help='filter pointless inter-temporal edges before embedding')
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
	paths = glob.glob(args.path_to_snapshots + "*.gt")
	snapshots = map(lambda p: loadSnapshot(p), paths)
	snapshots.sort(key=getGraphPhase)
	graph_name = getGraphName(snapshots[0])
	print("\nsnapshots loaded")
else :
	# load the board_directors data set
	print("\nloading default data set ...")
	phases = ["2002-06-01","2003-06-01","2004-06-01","2005-06-01","2006-06-01","2007-06-01","2008-06-01","2009-06-01","2010-06-01","2011-06-01"]
	cpt = 0
	for phase in phases :
		if cpt > 2 :
			continue
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

output_edges = open("./" + graph_name + "_edges.csv", "w")
output_edges.write("source,target,weight,phase,type,sync_component\n")	

def toBlockFile(file,b_id,phase,size,sync_component_id,diac_component_id,lineage_size,meta,x,y,z) :
	file.write(b_id + ',' + phase + ',' + size + ',' + sync_component_id +',' + diac_component_id + ',' + lineage_size + ',' + meta + ',' + x + ',' + y + ',' + z + '\n')	

output_blocks = open("./" + graph_name + "_blocks.csv", "w")
output_blocks.write("id,phase,size,sync_component,diac_component,lineage_size,meta,x,y,z\n")


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
			grouped[s].append([s,t,w])
		else :
			grouped[s] = [[s,t,w]]
		if t in grouped.keys() :
			grouped[t].append([s,t,w])
		else :
			grouped[t] = [[s,t,w]]	
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

def getVertexMeta (snapshot,v) :
	try:
		return snapshot.vp.vmeta[v]
	except Exception as e:
		return 'NA'

phases = []
blocks_to_meta = {}
blocks_to_synchronic_components = {}
sequence_of_blocks = {}
flow_size = {}

for snapshot in snapshots :

	# inferring partition and meta-graph with sbm

	phase = getGraphPhase(snapshot)
	phases.append(phase)

	print("\n####")
	print(phase)
	print(snapshot)

	partition = gt.minimize_blockmodel_dl(snapshot)

	blocks = {}

	for v in snapshot.vertices() :

		# [block] aggregate the vertices at the block level

		v_id   = getVertexId(snapshot,v)
		v_meta = getVertexMeta(snapshot,v)
		
		b_id = str(partition.get_blocks()[v]) + "_" + phase

		if b_id in blocks.keys() :
			blocks[b_id].append(v_id)
			blocks_to_meta[b_id].append(v_meta)
		else :
			blocks[b_id] = [v_id]
			blocks_to_meta[b_id] = [v_meta]

	for b_id in blocks.keys() :	
		sequence_of_blocks[b_id] = blocks[b_id]

	# [metadata] aggregate the metadata to the block level

	for b_id in blocks.keys() :	
		labels = blocks_to_meta[b_id]
		freq_max = 0
		most_freq_label = ''
		for label in labels :
			if (labels.count(label) > freq_max) :
				# we use a simple most frequent strategy
				freq_max = labels.count(label)
				most_freq_label = label
		blocks_to_meta[b_id] = most_freq_label				

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
		if (sync_edges[edge]["shhi"] == 2) :
			# each edge must satisfy the sHHI test
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

				
####
## create the inter-temporal similarity matrix
####


print('\nbuild the inter-temporal similarity matrix')

blocks_to_diachronic_components = {}
diac_edges = {}
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
		
		sim = jaccard(bi_v,bj_v)
		row.append(sim)
		
		# [flow size] compute the flow size for visualization
		
		if (sim > 0) and (isDirectAncestor(bi_t,bj_t)) :
			if bj in flow_size.keys() :
				flow_size[bj] += sim * len(bi_v)
			else :
				flow_size[bj] =  sim * len(bi_v)

			# [diac_edges] 1) put the (t-1,t) inter-temporal edges aside to compute the inter-temporal lineages 
			
			if (sim >= args.threshold) :
				# these edges will be the only visible in the interface 
				diac_edges[(bi,bj)] = {'w':sim,'shhi':0}

	mat.append(row)

# [diac_edges] 2) filter the visible inter-temporal edges by using a sHHI	

grouped_diac_edges = groupEdges(diac_edges)

for edges in grouped_diac_edges :
	filtered_edges = hhiFilter(edges)
	for edge in filtered_edges :
		diac_edges[(edge[0],edge[1])]["shhi"] += 1

# [diac_edges] 3) find diachronic connected components

diac_edges_list = []
for edge in diac_edges.keys() :
	if (diac_edges[edge]["shhi"] == 2) :
		diac_edges_list.append([edge[0],edge[1]])

# diachronic connected components' id will be transfered to the blocks

diac_connected_components = connectedComponents([],diac_edges_list[0],diac_edges_list[1:])

cmp_cpt = 0
for component in diac_connected_components :
	for block in component :
		blocks_to_diachronic_components[block] = cmp_cpt
	cmp_cpt += 1

# [diac_edges] 4) export the visible inter-temporal edges

for edge in diac_edges.keys() :
	if (diac_edges[edge]["shhi"] == 2) :
		component_id = blocks_to_diachronic_components[edge[0]]
		toEdgeFile(output_edges,edge[0],edge[1],str(diac_edges[edge]["w"]),edge[0].split('_')[1],'diac',str(component_id))


####
## embed the inter-temporal similarity matrix
####


print('\nembed the sequence of meta-graphs')

def areInScope(bi_t,bj_t) :
	bi_t_idx = phases.index(bi_t)
	bj_t_idx = phases.index(bj_t)
	if (args.scope < 0) :
		return True
	else :
		return (abs(bi_t_idx - bj_t_idx) <= args.scope)

mat = np.array(mat)
b_ids = list(sequence_of_blocks.keys())

# [embedding] 1) export the edges to a temporary file (mandatory by PecanPy)

embedded_edges = open("./edges_for_embedding.csv", "w")

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
					embedded_edges.write(bi + '\t' + bj + '\t' + str(mat[i][j]) + '\n')

# [embedding] 2) embed the matrix (full or filtered) via the node2vec approach

high_dimensional_graph = node2vec.SparseOTF(p=1, q=1, workers=4, verbose=True)
high_dimensional_graph.read_edg("./edges_for_embedding.csv", weighted=True, directed=False, delimiter="\t")
walks = high_dimensional_graph.simulate_walks(num_walks=50, walk_length=30)
embedding = Word2Vec(walks, vector_size=64, window=5, min_count=1, sg=1, workers=4, epochs=1)


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
		      , str(xs[i])
		      , str(ys[i])
		      , str(zs[i]))		

print('\nReady for visualization')