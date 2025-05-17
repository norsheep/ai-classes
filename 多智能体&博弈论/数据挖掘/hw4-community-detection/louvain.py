import random
from collections import defaultdict

from graph import WeightedDiGraph
from community import Community

from typing import Dict, Tuple


class Louvain:

    def __init__(self, graph: WeightedDiGraph):
        # Number of nodes in the original graph
        # NOTE: These attributes should NOT be modified.
        self.original_graph = graph
        self.n_nodes = graph.N
        self.n_passes = 0

        # "G" is a graph of communities (we refer to it as "metagraph")
        # initially each node is its own community,
        # so the metagraph is (structurally) the same as the original graph
        self.G: WeightedDiGraph = graph.copy()  # G存的是子图不是原图

        # each node is initially a community of its own
        # we initialize n_nodes communities
        # the community is a map (community_id -> Community)
        self.communities: Dict[int, Community] = self._init_communities(
            self.G, self.n_nodes)  # 初始化，id到社区对象的映射

        # each node in the original graph belongs to a community,
        # this is tracked by node2commid (node_id -> community_id)
        # it maps a node in the original graph to its community id.
        # 将原始图中的节点 ID 映射到其所属的社区 ID
        self.node2commid: Dict[int, int] = {
            node_id: node_id
            for node_id in range(self.n_nodes)
        }

        # We use "metanode" to refer to a node in the metagraph
        # Each metanode represents a community of nodes in the original graph
        # In phase 1 of Louvain method (partitioning),
        # different metanodes might be merged into a single community
        # This is tracked by metanode2commid (metanode_id -> community_id)
        # it maps a metanode in the metagraph to its community id.
        # Initially, each metanode is its own community.
        # 表示当前元图（metagraph）中每个元节点所属的社区。元图是抽象后的社区
        self.metanode2commid: Dict[int, int] = {
            node_id: node_id
            for node_id in range(self.n_nodes)
        }

        # NOTE: In the code below we will use node and metanode interchangeably
        # unless otherwise stated, a "node" refers to a metanode in the metagraph

    def _init_communities(self, graph: WeightedDiGraph,
                          n_communities: int) -> Dict[int, Community]:
        """
        Initializes n_communities communities associated with the given graph.
        Each community is initialized with a single node.
        """
        return {
            node_id: Community(id=node_id, graph=graph, nodes={node_id})
            for node_id in range(n_communities)
        }

    def get_community_of_node(self, node: int) -> Community:
        """Returns the community to which the given node belongs."""
        return self.communities[self.metanode2commid[node]]

    def delta_modularity(self, node: int, community: Community) -> float:
        """Computes Delta_Q(i -> C),
        i.e., the change in modularity if we move node i to community C.
        """

        delta_q = 0
        # TODO: (Task 1) Compute Delta_Q(i -> C), i.e.,
        # the change in modularity if we move node i to community C
        #
        # Hints:
        # 1. The formula for Delta_Q(i -> C) is given in the handout.
        # 2. Make use of the functions you have implemented in the `Community` class.
        # 3. You can get the sum of edge weights of the metagraph by `self.G.M`.

        ##################
        if self.G.M == 0:
            return 0
        sum_weights = community.node2comm_degree(node)  # 计算节点i到社区C的边权重之和
        d_in = community.graph.get_in_degree(node)  # 计算节点i的入度
        d_out = community.graph.get_out_degree(node)  # 计算节点i的出度
        d_tot_out = community.get_out_degree()  # 计算社区C的出度
        d_tot_in = community.get_in_degree()  # 计算社区C的入度
        delta_q = sum_weights / self.G.M - (d_in * d_tot_out +
                                            d_out * d_tot_in) / (self.G.M**2)
        ##################

        # End of TODO

        return delta_q

    def phase1(self):
        n_metanodes = self.G.N
        num_iter = 0
        modularity_gain = 0
        while True:
            num_iter += 1
            changed = False
            n_changed = 0

            # TODO (Optional): In practice, the order in which we visit
            # the nodes might affect the final result.
            # We visit the nodes in the order of their metanode ids,
            # you can also try different orders (e.g., randomly shuffling the nodes).
            # 访问顺序不同会导致结果不同，默认用metanode id的顺序，也可以尝试随机的
            node_iterator = list(range(n_metanodes))
            # random.shuffle(node_iterator)  # 随机打乱顺序(在给定的reference的准确率要更低)

            print(
                f'Before phase {num_iter}, the number of communities is: {len(set(self.metanode2commid.values()))}'
            )

            for metanode in node_iterator:
                # remove current node from its old community
                old_community = self.get_community_of_node(metanode)
                old_community.remove_node(metanode)

                # TODO (Task 2): Compute Delta_Q(C -> i) for old_community and metanode.
                # You should set the variable `delta_q_del` to the computed value.
                #
                # Hints:
                # 1. Delta_Q(C -> i) = -Delta_Q(i -> C)
                # 2. Only one line of code is required here.

                # 此时社区已经删除了当前节点，可以直接用已有的
                delta_q_del = -self.delta_modularity(metanode, old_community)

                # End of TODO

                best_modularity = 0
                best_community = old_community
                # Iterate over neighbors of the current node
                for nbr in sorted(self.G.get_neighbors(metanode)):
                    # get the community of the neighbor node
                    new_community = self.get_community_of_node(nbr)

                    # skip if the neighbor is in the same old community
                    if new_community.id == old_community.id:
                        continue

                    # TODO (Task 2):
                    # Compute Delta_Q = Delta_Q(C -> i) + Delta_Q(i -> C).
                    # Update best_modularity and best_community if the new community
                    # has a higher modularity gain.

                    ##################
                    # Your Code Here #
                    delta_q_add = self.delta_modularity(
                        metanode, new_community)
                    if delta_q_add + delta_q_del > best_modularity:
                        best_modularity = delta_q_add + delta_q_del
                        best_community = new_community
                        # md为什么communities会到0啊？？？？
                    ##################

                    # End of TODO

                # add current node to the best community
                self.metanode2commid[metanode] = best_community.id
                best_community.add_node(metanode)
                modularity_gain += best_modularity

                # Update the changed flag if the node has changed its community
                if best_community.id != old_community.id:
                    changed = True
                    n_changed += 1

            print(
                f"| Pass: {self.n_passes:3d} "
                f"| Phase 1 | Iter: {num_iter:3d} "
                f"| Nodes changed: {n_changed:5d} ({changed}) "
                f"| #Communities: {len(set(self.metanode2commid.values())):5d} "
                f"| Modularity gain: {modularity_gain:.4f} |")

            if not changed:
                break

    def _update_node2commid(self):
        """Reassign nodes to their new communities after phase 1."""

        for node in range(self.n_nodes):
            # id of the old community of current node (in original graph)
            metanode_id = self.node2commid[node]
            # the metanode might be merged into other communities
            # so we need to find the new community id of the metanode
            community_id = self.metanode2commid[
                metanode_id]  # new community id
            # reassign the node to the new community
            self.node2commid[node] = community_id

    def _reindex_communities(self):
        """
        Reindex communities to make the community ids contiguous.
        Some communities might have been removed during phase 1,
        so we rearrange community ids so that they start from 0 and are contiguous

        E.g., if the communities are [0, 1, 3, 4, 5, 7, 8, 9],
        we reindex them to [0, 1, 2, 3, 4, 5, 6, 7]

        NOTE: `node2commid`, `metanode2commid` and `communities` will be updated.
        """
        remaining_communities = set(self.metanode2commid.values())
        reindex = {}
        for new_id, old_id in enumerate(remaining_communities):
            reindex[old_id] = new_id

        # update node2commid and metanode2commid to new community ids
        for node in range(self.n_nodes):
            self.node2commid[node] = reindex[self.node2commid[node]]
        for meta_node in range(self.G.N):
            self.metanode2commid[meta_node] = reindex[
                self.metanode2commid[meta_node]]

        # update community dict, drop removed communities
        self.communities = {
            reindex[old_id]: comm
            for old_id, comm in self.communities.items() if old_id in reindex
        }
        # update community id for consistency
        for new_id, comm in self.communities.items():
            comm.id = new_id

        # Sorry about the spaghetti code here :(
        # Should have considered some better ways to store the communities.

    def phase2(self):
        print(f"| Pass: {self.n_passes:3d} | Phase 2 | Building new graph. |")
        # update node to their new communities
        self._update_node2commid()
        # reindex communities to make the community ids continuous
        self._reindex_communities()

        new_edges: Dict[Tuple[int, int], int] = defaultdict(int)  # 默认值为0
        # TODO: (Task 2) Create a new metagraph of the updated communities
        # fill in `new_edges` with new edges between communities with updated weights
        # The format of `new_edges` is {(src, dst): weight}
        #
        # Hints:
        # 1. The communities will be the new metanodes in the new graph.
        # 2. The edge weights between the communities are the sum of the edge weights
        #    between the nodes in the original graph.
        # 3. You can use `get_community_of_node()` to get the community of a node,
        #    or use `metanode2commid` to get its community id.

        ##################
        # Your Code Here #
        for node in range(self.G.N):
            comm_id = self.metanode2commid[node]
            for neighbor in self.G.get_out_neighbors(node):
                # 指出
                neighbor_comm_id = self.metanode2commid[neighbor]  # 邻居的社区

                new_edges[(comm_id,
                           neighbor_comm_id)] += self.G.get_edge_weight(
                               node, neighbor)
        # for (src, dst), weight in self.G.edges.items():
        #     # 自己指向自己的社区也要加
        #     c_src = self.metanode2commid[src]
        #     c_dst = self.metanode2commid[dst]
        #     new_edges[(c_src, c_dst)] += weight
        ##################
        # End of TODO

        edge_list = [(src, dst, weight)
                     for (src, dst), weight in new_edges.items()]

        return WeightedDiGraph(edge_list)

    def louvain(self):
        random.seed(0)
        self.n_passes = 0
        while True:
            self.n_passes += 1

            self.phase1()
            g = self.phase2()  # 这里的g是新的metagraph

            # check if the new metagraph is the same as the old one
            if g.edges == self.G.edges:
                break

            # update the metagraph
            self.G = g
            self.communities = self._init_communities(self.G,
                                                      self.G.N)  # 社区就会重新初始化
            self.metanode2commid = {
                node_id: node_id
                for node_id in range(self.G.N)
            }  # 会更新元节点的社区id到新的，但是原始图那个不会改，计算的时候也应该算元图的

        return self.node2commid

    def merge_communities(
        self,
        node2commid: Dict[int, int],
        n_expected_communities: int,
        labels: Dict[int, int],
    ):
        # TODO (Task 2): merge extra communities to reduce the number of communities.
        # Update the `node2commid` dictionary to merge the communities.
        #
        # Hints:
        # 1. You can use the provided labels for merging the communities,
        #    e.g., consider merging two communities if many nodes in the
        #    community share the same ground-truth label.混合两个有许多点的真值相同的社区
        # 2. Another strategy is to find and merge small communities
        #    according to the delta modularity before and after merging them.
        #
        # Other merging strategies are also acceptable. You are encouraged to
        # experiment with different strategies to improve the performance.

        ##################
        # Your Code Here #

        # 嵌套字典，统计每个社区中各个标签的节点数量
        while len(set(node2commid.values())) > n_expected_communities:
            commid2labels = defaultdict(lambda: defaultdict(int))
            for node, commid in node2commid.items():
                # labels[node]是ground-truth标签
                if node in labels:
                    # label没有提供完整的
                    commid2labels[commid][labels[node]] += 1

            # 确定每个社区的主导label
            com2domlabel = {}
            for commid, label_dict in commid2labels.items():
                com2domlabel[commid] = max(
                    label_dict.items(),
                    key=lambda x: x[1])[0]  # 按照vlaue排序，返回label

            # 按照主导标签分组社区
            # label2com = defaultdict(set)
            # for commid, label in com2domlabel.items():
            #     label2com[label].add(commid)

            # # 记录每个社区分配的新id（其实可以直接用label）
            # comm_map = {
            #     comm: new_id
            #     for new_id, comms in enumerate(label2com.values())
            #     for comm in comms
            # }
            # node2commid = {
            #     node: comm_map[commid]
            #     for node, commid in node2commid.items()
            # }

            node2commid = {
                node: com2domlabel[commid]
                for node, commid in node2commid.items()
            }
            # 这里和被注释掉的地方作用是相同的
        # print('After merge, the number of communities:', len(set(node2commid.values())))
        ##################
        # End of TODO

        return node2commid
