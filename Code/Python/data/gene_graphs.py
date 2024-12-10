
import csv
import numpy as np
import pandas as pd
import h5py
import networkx as nx
import academictorrents as at
from data.utils import symbol_map, ncbi_to_hugo_map, ensp_to_hugo_map, randmap
import os
import itertools


class GeneInteractionGraph(object):
    """ This class manages the data pertaining to the relationships between genes.
        It has an nx_graph, and some helper functions.
    """

    def __init__(self, relabel_genes=True, datastore=None, randomize=False):
        
        if datastore is None:
            self.datastore = os.path.dirname(os.path.abspath(__file__))
        else:
            self.datastore = datastore
        self.load_data()
        self.nx_graph = nx.relabel.relabel_nodes(self.nx_graph, symbol_map(self.nx_graph.nodes))
        
        # Randomize
        self.randomize = randomize
        if self.randomize:
            print("Randomizing the graph")
            self.nx_graph = nx.relabel.relabel_nodes(self.nx_graph, randmap(self.nx_graph.nodes))

    def load_data(self):
        raise NotImplementedError

    def first_degree(self, gene):
        neighbors = set([gene])
        # If the node is not in the graph, we will just return that node
        try:
            neighbors = neighbors.union(set(self.nx_graph.neighbors(gene)))
        except Exception as e:
            # print(e)
            pass
        neighborhood = np.asarray(nx.to_numpy_matrix(self.nx_graph.subgraph(neighbors)))
        return neighbors, neighborhood

    def bfs_sample_neighbors(self, gene, num_neighbors, include_self=True):
        neighbors = nx.OrderedGraph()
        if include_self:
            neighbors.add_node(gene)
        bfs = nx.bfs_edges(self.nx_graph, gene)
        for u, v in bfs:
            if neighbors.number_of_nodes() == num_neighbors:
                break
            neighbors.add_node(v)

        for node in neighbors.nodes():
            for u, v, d in self.nx_graph.edges(node, data="weight"):
                if neighbors.has_node(u) and neighbors.has_node(v):
                    neighbors.add_weighted_edges_from([(u, v, d)])
        return neighbors

    def adj(self):
        return nx.to_numpy_matrix(self.nx_graph)


class RegNetGraph(GeneInteractionGraph):
    
    def __init__(self, graph_name="regnet", at_hash="e109e087a8fc8aec45bae3a74a193922ce27fc58", randomize=False, **kwargs):
        self.graph_name = graph_name
        self.at_hash = at_hash
        super(RegNetGraph, self).__init__(**kwargs)

    def load_data(self):
        
        savefile = os.path.join(self.datastore,"graphs", self.graph_name + ".adjlist.gz")
        
        if os.path.isfile(savefile):
            print(" loading from cache file" + savefile)
            self.nx_graph = nx.read_adjlist(savefile)
        else:
        
            self.nx_graph = nx.OrderedGraph(
                nx.readwrite.gpickle.read_gpickle(at.get(self.at_hash, datastore=self.datastore)))
            
            print(" writing graph")
            nx.write_adjlist(self.nx_graph, savefile)

class GeneManiaGraph(GeneInteractionGraph):

    def __init__(self, graph_name="genemania", at_hash="5adbacb0b7ea663ac4a7758d39250a1bd28c5b40", **kwargs):
        self.graph_name = graph_name
        self.at_hash = at_hash
        super(GeneManiaGraph, self).__init__(**kwargs)


    def load_data(self):
        
        savefile = os.path.join(self.datastore,"graphs", self.graph_name + ".adjlist.gz")
        
        if os.path.isfile(savefile):
            print(" loading from cache file" + savefile)
            self.nx_graph = nx.read_adjlist(savefile)
        else:
        
            self.nx_graph = nx.OrderedGraph(
                nx.readwrite.gpickle.read_gpickle(at.get(self.at_hash, datastore=self.datastore)))
            
            print(" writing graph")
            nx.write_adjlist(self.nx_graph, savefile)

class HumanNetV2Graph(GeneInteractionGraph):
    """
    More info on HumanNet V1 : http://www.functionalnet.org/humannet/about.html
    """

    def __init__(self, randomize=False, **kwargs):
        super(HumanNetV2Graph, self).__init__(**kwargs)

    def load_data(self):
        self.benchmark = self.datastore + "/graphs/HumanNet-XN.tsv"
        edgelist = pd.read_csv(self.benchmark, header=None, sep="\t", skiprows=1).values[:, :2].tolist()
        self.nx_graph = nx.OrderedGraph(edgelist)
        # Map nodes from ncbi to hugo names
        self.nx_graph = nx.relabel.relabel_nodes(self.nx_graph, ncbi_to_hugo_map(self.nx_graph.nodes, datastore=self.datastore))
        # Remove nodes which are not covered by the map
        for node in list(self.nx_graph.nodes):
            if isinstance(node, float):
                self.nx_graph.remove_node(node)


class FunCoupGraph(GeneInteractionGraph):
    """
    Class for loading and processing FunCoup into a NetworkX object
    Please download the data file - 'FC4.0_H.sapiens_full.gz' from
    http://funcoup.sbc.su.se/downloads/ and place it in the 
    graphs folder before instantiating this class
    """

    def __init__(self, graph_name='funcoup', randomize=False, **kwargs):
        self.graph_name = graph_name
        super(FunCoupGraph, self).__init__(**kwargs)

    def load_data(self):
        
        savefile = os.path.join(self.datastore,"graphs", self.graph_name + ".adjlist.gz")
        
        if os.path.isfile(savefile):
            print(" loading from cache file" + savefile)
            self.nx_graph = nx.read_adjlist(savefile)
        else:
            pkl_file = os.path.join(self.datastore,"graphs", self.graph_name + ".pkl")
            if not os.path.isfile(pkl_file):
                print(" creating graph")
                self._preprocess_and_pickle(save_name=pkl_file)
            self.nx_graph = nx.OrderedGraph(nx.read_gpickle(pkl_file))
                                
            print(" writing graph")
            nx.write_adjlist(self.nx_graph, savefile)

    def _preprocess_and_pickle(self, save_name):
        names_map_file = os.path.join(self.datastore,"graphs", 'ensembl_to_hugo.tsv')
        data_file = os.path.join(self.datastore,"graphs", 'FC4.0_H.sapiens_full.gz')

        names = pd.read_csv(names_map_file, sep='\t')
        names.columns = ['symbol', 'ensembl']
        names = names.dropna(subset=['ensembl']).drop_duplicates('ensembl')
        names = names.set_index('ensembl').squeeze()

        data = pd.read_csv(data_file, sep='\t', usecols=['#0:PFC', '1:FBS_max',
                                                         '2:Gene1', '3:Gene2'])
        data['2:Gene1'] = data['2:Gene1'].map(names)
        data['3:Gene2'] = data['3:Gene2'].map(names)
        data = data.dropna(subset=['2:Gene1', '3:Gene2'])

        graph = nx.from_pandas_edgelist(data, source='2:Gene1', target='3:Gene2',
                                        # edge_attr=['#0:PFC', '1:FBS_max'], # Uncomment to include edge attributes
                                        create_using=nx.OrderedGraph)
        nx.write_gpickle(graph, save_name)
