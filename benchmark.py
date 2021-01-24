#!/usr/bin/env python3
import logging
import random
from collections import Counter
from functools import partial
from itertools import product, chain
from statistics import variance
from string import ascii_lowercase, ascii_uppercase
from typing import Any, Set, Tuple, List, Callable

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from tqdm import tqdm

from evaluation import BetaDistributedSimilarity
from lib import WORKING_DIRECTORY, timeit, LTS, Mapping, trace, LPSimulator

LABELS_LC = set(ascii_lowercase)
LABELS_UC = set(ascii_uppercase)
MAPPING = {(a, b) for a, b in zip(ascii_lowercase, ascii_uppercase)}


def rand_flip(t: Tuple[Any], p: float) -> Tuple[Any]:
    """reverse tuple by given probability"""
    if random.random() <= p:
        return tuple(reversed(t))
    return t


def lts_degrees(lts: LTS) -> List[int]:
    """get state degrees of a LTS"""
    return list(Counter(chain(*((a, b) for a, _, b in lts.transitions))).values())


def rgraph_normal(n: int, avg_degree: float, seed=None) -> nx.DiGraph:
    """generate random graph with normal degree distribution"""
    return nx.generators.random_graphs.erdos_renyi_graph(n, avg_degree / (2 * n - 1), directed=True, seed=seed)


def rgraph_scale_free(n: int, avg_degree: float, seed=None) -> nx.DiGraph:
    """generate random graph with degree distribution following a power law"""
    g = nx.generators.barabasi_albert_graph(n, round(avg_degree / 2), seed=seed)
    g = nx.from_edgelist(map(partial(rand_flip, p=.5), g.edges), create_using=nx.DiGraph)
    return g


def label_graph_edges(g: nx.DiGraph, labels: Set[str]) -> nx.DiGraph:
    """randomly assign labels to all edges"""
    labels = list(labels)

    for a, b in g.edges:
        g[a][b]['label'] = random.choice(labels)

    return g


def map_graph_labels(g: nx.DiGraph, mapping: Mapping) -> nx.DiGraph:
    """re-label a graph with respect to the given mapping"""
    mapping = dict(mapping)

    for a, b in g.edges:
        g[a][b]['label'] = mapping[g[a][b]['label']]

    return g


def rand_rewire_graph(g: nx.DiGraph, p: float) -> Tuple[nx.DiGraph, Set[str]]:
    """
    Re-wire given share of edges randomly by either
      - re-assigning an edge to another node pair (50%)
      - swap labels of two edges (50%)
    """
    g = g.copy()
    nodes = list(g.nodes)
    labels_mod = set()

    for i in range(round(len(g.nodes) * p)):
        edges = list(g.edges)

        # swap edge labels
        if random.random() < .5:
            u, v = random.choice(edges)
            x, y = random.choice(edges)

            tmp = g[u][v]['label']
            g[u][v]['label'] = g[x][y]['label']
            g[x][y]['label'] = tmp

            labels_mod.add(tmp)

        # re-wire edge
        else:
            u, v = random.choice(edges)
            x, y = random.choice(nodes), random.choice(nodes)

            g.add_edge(x, y, label=g[u][v]['label'])
            g.remove_edge(u, v)

            labels_mod.add(g[x][y]['label'])

    return g, labels_mod


def lts_from_digraph(g: nx.DiGraph, initial_state: int, name: str) -> LTS:
    """generate a LTS from networkx graph"""
    # only keep reachable nodes
    reachable = set(nx.algorithms.traversal.dfs_preorder_nodes(g, initial_state))
    for n in list(g.nodes):
        if n not in reachable:
            g.remove_node(n)

    # due to pruning, initial state may be removed from graph
    if initial_state not in g.nodes:
        raise ValueError(f'inital state {initial_state} in not part of graph')

    def make_ident(id):
        return f'{name[0]}{id}'

    transitions = [(make_ident(a), g[a][b]['label'], make_ident(b)) for a, b in g.edges]
    initial_state = make_ident(initial_state)

    return LTS(transitions, initial_state=initial_state, name=name)


def generate_example(factory: Callable[[int, float], nx.DiGraph], n: int, d: float) -> Tuple[LTS, LTS, Mapping]:
    # generate random graph pair
    g_1 = label_graph_edges(factory(n, d), LABELS_LC)
    g_2, labels_changed = rand_rewire_graph(map_graph_labels(g_1, MAPPING), .2)

    ref_mapping = {(a, b) for a, b in MAPPING if a not in labels_changed and b not in labels_changed}

    #  chose a random initial state
    initial_state = random.choice(list(g_1.nodes))
    assert initial_state in g_2.nodes

    # build LTS'
    lts_1 = lts_from_digraph(g_1, initial_state, 'a')
    lts_2 = lts_from_digraph(g_2, initial_state, 'b')

    return lts_1, lts_2, ref_mapping


def _benchmark():
    """generate different problem instances and solve them using the Gurobi backend"""
    yield 'n', 'd', 'method', 'duration', 'error', 'avg_n', 'avg_e', 'avg_d', 'avg_v'

    ns = [2 ** e for e in range(3, 9)]
    ds = [4, 6]
    fs = [rgraph_normal, rgraph_scale_free]
    rs = list(range(3))
    logging.info(f'{ns} x {ds} x {fs} x {rs}')

    configurations = list(product(ns, ds, fs, rs))
    logging.info(f'run benchmark for {len(configurations)} examples')

    for n, d, factory, _ in tqdm(reversed(configurations)):
        # ensure that for each configuration one simulation succeeds
        while True:
            try:
                # generate example
                # ensure that at least 85% of the nodes remain in the example and the mapping is non trivial
                while True:
                    lts_1, lts_2, ref_mapping = generate_example(factory, n, d)

                    if all(map(lambda l: len(l.states) >= .85 * n, (lts_1, lts_2))) and 0 < len(ref_mapping) < 26:
                        break

                # prepare simulator
                similarities = BetaDistributedSimilarity(1., 12., LABELS_LC, LABELS_UC, ref_mapping, threshold=.3)
                simulator = LPSimulator(similarities, gap_penalty=.25, backend='GUROBI_CMD')

                # compute and assess mapping
                duration, mapping = timeit(lambda: trace(lts_1, lts_2, simulator))
                error = len(set.symmetric_difference(ref_mapping, mapping)) / max(len(ref_mapping), len(mapping))

                # LTS stats
                deg_1 = lts_degrees(lts_1)
                deg_2 = lts_degrees(lts_2)
                avg_n = (len(lts_1.states) + len(lts_2.states)) / 2
                avg_e = (len(lts_1.transitions) + len(lts_2.transitions)) / 2
                avg_d = (sum(deg_1) + sum(deg_2)) / (len(deg_1) + len(deg_2))
                avg_v = (variance(deg_1) + variance(deg_2)) / 2

                yield n, d, factory.__name__, duration, error, avg_n, avg_e, avg_d, avg_v
                break

            except Exception as error:
                logging.debug(f'{type(error)}: {error}')


def benchmark():
    """run benchmark and store results to a csv file"""
    generator = _benchmark()
    columns = next(generator)
    pd.DataFrame(generator, columns=columns).to_csv(WORKING_DIRECTORY.joinpath('bench_raw.csv'), index=False)


def visualize():
    """plot benchmark results"""
    raw = pd.read_csv(WORKING_DIRECTORY.joinpath('bench_raw.csv'))
    mmap = {
        'rgraph_normal': 'normal',
        'rgraph_scale_free': 'power law',
    }

    fig, (ax_1, ax_2) = plt.subplots(ncols=2, figsize=(10, 4), dpi=300, sharey='all')

    for m in mmap:
        dist = mmap[m]
        for d, g in raw[raw['method'] == m].groupby(by='d'):
            s = g.groupby(by='n').mean().set_index('avg_n').sort_index()['duration']
            s.name = f'{dist} ($\\left< k \\right> = {d}$)'
            s.plot(ax=ax_1, linestyle=':', marker='$\\bigcirc$', legend=True)

            s = g.groupby(by='n').mean().set_index('avg_e').sort_index()['duration']
            s.name = f'{dist} ($\\left< k \\right> = {d}$)'
            s.plot(ax=ax_2, linestyle=':', marker='$\\bigcirc$', legend=True)

    ax_1.set_xlabel('$|S|$')
    ax_2.set_xlabel('$|T|$')
    ax_1.set_ylabel('duration ($s$)')
    ax_2.set_ylabel('duration ($s$)')
    ax_1.loglog()
    ax_2.loglog()
    plt.tight_layout()
    plt.savefig(WORKING_DIRECTORY.joinpath('benchmark.pdf'), transparent=True)


if __name__ == '__main__':
    benchmark()
    visualize()
