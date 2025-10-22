#######
# sandbox.py
# description : Test environment for evaluating Chronoblox
# licence : AGPL + CECILL v3
# paper : https://arxiv.org/abs/2405.07506
#######

####
## imports & args
####

import graph_tool.all as gt
import argparse
import random

parser = argparse.ArgumentParser(description='Test environment for evaluating Chronoblox')
parser.add_argument('--nb_nodes'
				   , type=int
				   , default=400
				   , help='number of nodes V in the environment')
parser.add_argument('--level_of_mixing'
				   , type=float
				   , default=0.1
				   , help='controls the level of node mixing between blocks, level ∈ [0,0.5[')
parser.add_argument('--reserve'
				   , type=bool
				   , default=False
				   , help='use a reserve R while mixing nodes between blocks to emulate newcomers in the environment, |R| = |V|')
parser.add_argument('--level_of_mixing_with_reserve'
				   , type=float
				   , default=0.05
				   , help='controls the level of mixing between V and R')
args = parser.parse_args()

####
## setting up
####

λ = args.level_of_mixing
ψ = args.level_of_mixing_with_reserve
nodes_labels = list(map(lambda v: "n" + str(v), range(0,args.nb_nodes)))
nodes_labels_reserve = list(map(lambda v: "n" + str(v), range(args.nb_nodes,2*args.nb_nodes)))
blocks_labels = ["A","B","C","D"]
mixing_constraints = {"A":"B","B":"A","C":"D","D":"C"}

def initGraph(scenario,phase) :
	g = gt.Graph()
	g.gp["graph_phase"] = g.new_graph_property("string")
	g.gp["graph_phase"] = phase
	g.gp["graph_name"] = g.new_graph_property("string")
	g.gp["graph_name"] = "scenario_" + scenario + "_" + str(λ) + ("_with_reserve" if args.reserve else "") + "_" + phase
	g.vp["vblock"] = g.new_vertex_property("string")
	g.vp["vid"] = g.new_vertex_property("string")
	# init V	
	for i in range(0,4) :
		for j in range(i*100,(i+1)*100) :
			v = g.add_vertex()
			g.vp.vblock[v] = blocks_labels[i]
			g.vp.vid[v] = nodes_labels[j]
	# init R
	for k in range(0,len(nodes_labels_reserve)) :
		v = g.add_vertex()
		g.vp.vblock[v] = "R"
		g.vp.vid[v] = nodes_labels_reserve[k]
	return g	

def copyGraph(g_old,scenario,phase) :
	g_new = gt.Graph(g_old)
	g_new.gp["graph_phase"] = phase
	g_new.gp["graph_name"] = "scenario_" + scenario + "_" + str(λ) + ("_with_reserve" if args.reserve else "") + "_" + phase
	return g_new

def saveGraph(g) :
	label_str_to_int = {"A":1,"B":2,"C":3,"D":4}
	g_new = gt.GraphView(g, vfilt=lambda v: g.vp.vblock[v] != "R")
	g_new.vp["vgroup"] = g_new.new_vertex_property("int")
	for v in g_new.vertices():
		g_new.vp.vgroup[v] = label_str_to_int[g_new.vp.vblock[v]]
	g_new.save("scenarios/" + g_new.gp["graph_name"] + ".gt")	

####
## manipulating graphs
####

def pickLeavingNodes(g,block,level) :
	nodes = []
	for v in g.vertices():
		if g.vp.vblock[v] == block :
			nodes.append(g.vp.vid[v])
	leaving_nodes = random.sample(nodes,int(abs(level*len(nodes))))
	return leaving_nodes	

def switchNodesWithReserve(g,level) :
	if args.reserve :
		nodes_V = []
		nodes_R = []	
		for v in g.vertices() :
			if g.vp.vblock[v] in blocks_labels :
				nodes_V.append(v)
			else :
				nodes_R.append(v)
		leaving_nodes_V = random.sample(nodes_V,int(abs(level*len(nodes_V))))
		leaving_nodes_R = random.sample(nodes_R,int(abs(level*len(nodes_R))))
		for i in range(0,len(leaving_nodes_V)) :
			# switch V and R : v goes in R, r becomes a member of v_block
			v = leaving_nodes_V[i]
			r = leaving_nodes_R[i]
			v_block = g.vp.vblock[v]
			g.vp.vblock[v] = "R"
			g.vp.vblock[r] = v_block
	return g

def mixNodesBetweenGroups(g,level,constraint=False) :
	for block in blocks_labels :
		new_blocks_labels = list(set(blocks_labels) - set([block]))
		leaving_nodes = pickLeavingNodes(g,block,level)
		for v in g.vertices() :
			if g.vp.vid[v] in leaving_nodes :
				if constraint :
					old_block = g.vp.vblock[v]
					g.vp.vblock[v] = mixing_constraints[old_block]
				else :
					g.vp.vblock[v] = random.choice(new_blocks_labels)
	return g
	
####
## scenario α 
####

print("\n####")
print("Scenario α")
steps = [0,0,0]

## phase 1

print("• t1")
g1 = initGraph("alpha","t1")
saveGraph(g1)

## phase 2

print("• t1 → t2")
g2 = copyGraph(g1,"alpha","t2")
g2 = switchNodesWithReserve(g2,ψ)
g2 = mixNodesBetweenGroups(g2,steps[0])
saveGraph(g2)

## phase 3

print("• t2 → t3")
g3 = copyGraph(g2,"alpha","t3")
g3 = switchNodesWithReserve(g3,ψ)
g3 = mixNodesBetweenGroups(g3,steps[1])
saveGraph(g3)

## phase 4

print("• t3 → t4")
g4 = copyGraph(g3,"alpha","t4")
g4 = switchNodesWithReserve(g4,ψ)
g4 = mixNodesBetweenGroups(g4,steps[2])
saveGraph(g4)

####
## scenario β 
####

print("\n####")
print("Scenario β")
steps = [λ,λ,λ]

## phase 1

print("• t1")
g1 = initGraph("beta","t1")
saveGraph(g1)

## phase 2

print("• t1 → t2")
g2 = copyGraph(g1,"beta","t2")
g2 = switchNodesWithReserve(g2,ψ)
g2 = mixNodesBetweenGroups(g2,steps[0])
saveGraph(g2)

## phase 3

print("• t2 → t3")
g3 = copyGraph(g2,"beta","t3")
g3 = switchNodesWithReserve(g3,ψ)
g3 = mixNodesBetweenGroups(g3,steps[1])
saveGraph(g3)

## phase 4

print("• t3 → t4")
g4 = copyGraph(g3,"beta","t4")
g4 = switchNodesWithReserve(g4,ψ)
g4 = mixNodesBetweenGroups(g4,steps[2])
saveGraph(g4)

####
## scenario γ
####

print("\n####")
print("Scenario γ")
steps = [1-λ,1-λ,1-λ]

## phase 1

print("• t1")
g1 = initGraph("gamma","t1")
saveGraph(g1)

## phase 2

print("• t1 → t2")
g2 = copyGraph(g1,"gamma","t2")
g2 = switchNodesWithReserve(g2,ψ)
g2 = mixNodesBetweenGroups(g2,steps[0])
saveGraph(g2)

## phase 3

print("• t2 → t3")
g3 = copyGraph(g2,"gamma","t3")
g3 = switchNodesWithReserve(g3,ψ)
g3 = mixNodesBetweenGroups(g3,steps[1])
saveGraph(g3)

## phase 4

print("• t3 → t4")
g4 = copyGraph(g3,"gamma","t4")
g4 = switchNodesWithReserve(g4,ψ)
g4 = mixNodesBetweenGroups(g4,steps[2])
saveGraph(g4)

####
## scenario δ
####

print("\n####")
print("Scenario δ")
steps = [λ,1-λ,λ]

## phase 1

print("• t1")
g1 = initGraph("delta","t1")
saveGraph(g1)

## phase 2

print("• t1 → t2")
g2 = copyGraph(g1,"delta","t2")
g2 = switchNodesWithReserve(g2,ψ)
g2 = mixNodesBetweenGroups(g2,steps[0])
saveGraph(g2)

## phase 3

print("• t2 → t3")
g3 = copyGraph(g2,"delta","t3")
g3 = switchNodesWithReserve(g3,ψ)
g3 = mixNodesBetweenGroups(g3,steps[1])
saveGraph(g3)

## phase 4

print("• t3 → t4")
g4 = copyGraph(g3,"delta","t4")
g4 = switchNodesWithReserve(g4,ψ)
g4 = mixNodesBetweenGroups(g4,steps[2])
saveGraph(g4)

####
## scenario Ɛ
####

print("\n####")
print("Scenario Ɛ")
steps = [λ,λ,λ]

## phase 1

print("• t1")
g1 = initGraph("epsilon","t1")
saveGraph(g1)

## phase 2

print("• t1 → t2")
g2 = copyGraph(g1,"epsilon","t2")
g2 = switchNodesWithReserve(g2,ψ)
g2 = mixNodesBetweenGroups(g2,steps[0],True)
saveGraph(g2)

## phase 3

print("• t2 → t3")
g3 = copyGraph(g2,"epsilon","t3")
g3 = switchNodesWithReserve(g3,ψ)
g3 = mixNodesBetweenGroups(g3,steps[1],True)
saveGraph(g3)

## phase 4

print("• t3 → t4")
g4 = copyGraph(g3,"epsilon","t4")
g4 = switchNodesWithReserve(g4,ψ)
g4 = mixNodesBetweenGroups(g4,steps[2],True)
saveGraph(g4)