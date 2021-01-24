#!/usr/bin/env python3

from itertools import product, repeat
from multiprocessing import Pool, cpu_count
from typing import Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import beta as beta_dist
from scipy.special import gamma

from examples import LTS_P, LTS_Q
from lib import WORKING_DIRECTORY, Mapping, SimilarityFunction, SimilarityTable, LPSimulator, trace, s_average, \
    merge, cr_ignore, cr_prune

REFERENCE_MAPPING = {('A', 'a'), ('B', 'b'), ('C', 'c')}


class BetaDistributedSimilarity(SimilarityTable):
    """Random similarity function that employs beta distribution"""

    def __init__(self, alpha: float, beta: float, labels_1: Set[str], labels_2: Set[str], equalities: Mapping,
                 threshold: float = 0.):
        """
        :param alpha: alpha value for beta distribution
        :param beta: beta value for beta distribution
        :param labels_1: source space
        :param labels_2: target space
        :param equalities: labels that are considered equal (1 - beta)
        :param threshold: every similarity below is considered to be zero
        """
        s = ((a, b, beta_dist(alpha, beta)) for a, b in product(labels_1, labels_2))
        s = ((a, b, 1 - w if (a, b) in equalities else w) for a, b, w in s)
        s = ((a, b, w if w > threshold else 0.) for a, b, w in s)
        super().__init__(s)


def compute_error(mapping: Mapping) -> float:
    """compute error by counting misplaced items with respect to reference mapping"""
    return len(set.symmetric_difference(REFERENCE_MAPPING, mapping))


def greedy_mapper(similarities: SimilarityFunction, threshold: float = 0.) -> Mapping:
    """reference mapping strategy using a greedy algorithm"""
    l_p = set(LTS_P.labels)
    l_q = set(LTS_Q.labels)
    mapping = set()

    while l_p and l_q:
        q, a, b = max((similarities[a, b], a, b) for a, b in product(l_p, l_q))
        l_p.remove(a)
        l_q.remove(b)
        if q >= threshold:
            mapping.add((a, b))

    return mapping


def _simulate(alpha, beta, gap_penalty=.25):
    similarities = BetaDistributedSimilarity(alpha, beta, LTS_P.labels, LTS_Q.labels, REFERENCE_MAPPING, .3)

    s_def = LPSimulator(similarities, gap_penalty)
    s_min = LPSimulator(similarities, gap_penalty, strategy=min)
    s_max = LPSimulator(similarities, gap_penalty, strategy=max)
    s_avg = LPSimulator(similarities, gap_penalty, strategy=s_average)

    m_pq = trace(LTS_P, LTS_Q, s_def)
    m_qp = trace(LTS_Q, LTS_P, s_def)

    return [
        greedy_mapper(similarities, .3),
        m_pq,
        {(b, a) for a, b in m_qp},
        merge(m_pq, m_qp, resolution=cr_prune),
        merge(m_pq, m_qp, resolution=cr_ignore),
        merge(m_pq, m_qp, strict=True, resolution=cr_prune),
        merge(m_pq, m_qp, strict=True, resolution=cr_ignore),
        trace(LTS_P, LTS_Q, s_min, bisimulation=True),
        trace(LTS_P, LTS_Q, s_max, bisimulation=True),
        trace(LTS_P, LTS_Q, s_avg, bisimulation=True),
    ]


def _simulate_n(alpha, beta, n):
    errors = np.zeros(10, dtype=np.uint32)

    for mappings in Pool(cpu_count()).starmap(_simulate, repeat((alpha, beta), times=n)):
        errors += np.fromiter(map(compute_error, mappings), dtype=np.uint32)

    return errors


def _simulations(n):
    for beta in reversed(range(6, 21, 2)):
        yield beta, *_simulate_n(1., beta, n)


def evaluate():
    """evaluate different mapping strategies on different noise levels"""
    n = 1000

    columns = [
        '$\\beta$',
        '$e_{greedy}$',
        '$e_{\\rightarrow}$',
        '$e_{\\leftarrow}$',
        '$e_{\\leftrightarrow_{\\cap, \\setminus}}$',
        '$e_{\\leftrightarrow_{\\cap, \\cup}}$',
        '$e_{\\leftrightarrow_{\\cup, \\setminus}}$',
        '$e_{\\leftrightarrow_{\\cup, \\cup}}$',
        '$e_{\\parallel_{min}}$',
        '$e_{\\parallel_{max}}$',
        '$e_{\\parallel_{avg}}$',
    ]

    errors = pd.DataFrame(
        _simulations(n),
        columns=columns
    ).set_index(columns[0])

    plt.rc('text', usetex=True)

    for name, cols in [('full', columns[1:]), ('selected', columns[1:6])]:
        fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
        errors[cols].plot.bar(ax=ax)
        ax.set_title(f'Error for $n = {{{n}}}$ runs ($\\alpha = 1$)')
        ax.set_ylabel('errors')
        plt.yscale('log')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(WORKING_DIRECTORY.joinpath(f'evaluation_{n}_{name}.pdf'), transparent=True)

    errors.to_csv(WORKING_DIRECTORY.joinpath(f'evaluation_{n}.csv'))


def plot_beta():
    """blot density function of beta distribution"""

    def beta(a, b):
        return (gamma(a) * gamma(b)) / gamma(a + b)

    def pdf(x, a, b):
        return ((x ** (a - 1.)) * ((1. - x) ** (b - 1.))) / beta(a, b)

    index = np.arange(0., 1., 0.0001)
    df = pd.DataFrame()
    df['$\\beta = 20$'] = pd.Series(pdf(index, 1., 20.), index=index)
    df['$\\beta = 12$'] = pd.Series(pdf(index, 1., 12.), index=index)
    df['$\\beta = 6$'] = pd.Series(pdf(index, 1., 6.), index=index)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=300)

    df.plot(ax=ax)

    ax.set_title('Density Function of Beta Distribution ($\\alpha = 1$)')
    ax.set_ylabel('Density')
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(WORKING_DIRECTORY.joinpath('beta_distr.pdf'), transparent=True)


if __name__ == '__main__':
    evaluate()
    plot_beta()
