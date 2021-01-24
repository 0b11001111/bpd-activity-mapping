#!/usr/bin/env python3
import logging
from functools import partial
from itertools import cycle, product

from graphviz import Digraph

from lib import WORKING_DIRECTORY, SimilarityTable, LTS, LPSimulator, trace, cr_ignore, cr_prune, merge, \
    s_average, equivalence_classes

# stolen from RWTH design guide
PALETTE = [
    '#57AB27',  # grün
    '#F6A800',  # orange
    '#CC071E',  # rot
    '#0098A1',  # türkis
    '#BDCD00',  # maigrun
    '#A11035',  # bordeaux
    '#612158',  # violett
    '#006165',  # petrol
    '#7A6FAC',  # lila
    '#00549F',  # blau
]

SIMILARITIES_1 = SimilarityTable([
    ('A', 'x', .0),
    ('A', 'a', .7),
    ('A', 'b', .0),
    ('A', 'c', .2),
    ('A', 'y', .1),
    ('B', 'x', .0),
    ('B', 'a', .1),
    ('B', 'b', .4),
    ('B', 'c', .2),
    ('B', 'y', .0),
    ('C', 'x', .1),
    ('C', 'a', .0),
    ('C', 'b', .0),
    ('C', 'c', .5),
    ('C', 'y', .1),
    ('Z', 'x', .0),
    ('Z', 'a', .0),
    ('Z', 'b', .1),
    ('Z', 'c', .1),
    ('Z', 'y', .0),
])
SIMILARITIES_2 = SimilarityTable([
    ('A', 'a', .7),
    ('A', 'b', .3),
    ('A', 'c', .1),
    ('B', 'a', .4),
    ('B', 'b', .6),
    ('B', 'c', .1),
    ('C', 'a', .1),
    ('C', 'b', .1),
    ('C', 'c', .1),
], default=.05)
SIMILARITIES_3 = SimilarityTable([
    ('A', 'a', .95),
    ('B', 'b', .95),
    ('C', 'c', .95),
], default=.05)
LTS_P = LTS(
    transitions={('p1', 'A', 'p2'), ('p2', 'C', 'p3'), ('p2', 'B', 'p4'), ('p3', 'Z', 'p4')},
    initial_state='p1',
    name='p'
)
LTS_Q = LTS(
    transitions={('q1', 'x', 'q2'), ('q2', 'a', 'q3'), ('q3', 'b', 'q4'), ('q3', 'c', 'q4'), ('q4', 'y', 'q5')},
    initial_state='q1',
    name='q'
)
LTS_S = LTS(
    transitions={('s1', 'x', 's2'), ('s2', 'b', 's3'), ('s2', 'c', 's3'), ('s3', 'a', 's4'), ('s4', 'y', 's5')},
    initial_state='s1',
    name='s'
)
LTS_I = LTS(
    transitions={('i1', 'A', 'i2'), ('i2', 'A', 'i3'), ('i3', 'A', 'i4'), ('i4', 'A', 'i5'), ('i1', 'B', 'ix'), },
    initial_state='i1',
    name='i'
)
LTS_J = LTS(
    transitions={('j1', 'c', 'j2'), ('j2', 'b', 'j3'), ('j3', 'b', 'j4'), ('j4', 'b', 'j5'), ('j1', 'a', 'jx'), },
    initial_state='j1',
    name='j'
)
LTS_X = LTS(
    transitions={('x1', 'A', 'x2')},
    initial_state='x1',
    name='x'
)
LTS_Y = LTS(
    transitions={('y1', 'a', 'y2'), *((f'y{i}', 'b', f'y{i + 1}') for i in range(2, 13))},
    initial_state='y1',
    name='y'
)
LTS_Z = LTS(
    transitions={('z1', 'a', 'z2'), *((f'z{i}', 'b', f'z{i + 1}') for i in range(2, 14))},
    initial_state='z1',
    name='z'
)
LTS_O = LTS(
    transitions={('o1', 'A', 'o2'), ('o2', 'B', 'o1')},
    initial_state='o1',
    name='o'
)
LTS_U = LTS(
    transitions={('u1', 'a', 'u2')},
    initial_state='u1',
    name='u'
)


def example_bisimulation_strategies():
    """illustrate differences between bisimulation strategies"""
    simulator = LPSimulator(SIMILARITIES_1, .25)
    print(f'simulate {LTS_P.name} by {LTS_Q.name} ({simulator})')
    print(repr(simulator.simulate(LTS_P, LTS_Q)))
    print()

    print(f'simulate {LTS_Q.name} by {LTS_P.name} ({simulator})')
    print(repr(simulator.simulate(LTS_Q, LTS_P)))
    print()

    print(f'bisimulate {LTS_P.name} by {LTS_Q.name} ({simulator})')
    print(repr(simulator.bisimulate(LTS_P, LTS_Q)))
    print()

    simulator = LPSimulator(SIMILARITIES_1, .25, strategy=max)
    print(f'bisimulate {LTS_P.name} by {LTS_Q.name} ({simulator})')
    print(repr(simulator.bisimulate(LTS_P, LTS_Q)))
    print()

    simulator = LPSimulator(SIMILARITIES_1, .25, strategy=s_average)
    print(f'bisimulate {LTS_P.name} by {LTS_Q.name} ({simulator})')
    print(repr(simulator.bisimulate(LTS_P, LTS_Q)))
    print()


def example_merging_strategies():
    """illustrate differences between mapping merge strategies"""
    m_1 = {('A', 'a'), ('B', 'b'), ('B', 'c'), ('D', 'e'), ('G', 'g')}
    m_2 = {('a', 'A'), ('b', 'B'), ('c', 'B'), ('f', 'D')}

    print(f'{m_1} U {m_2}')
    for strict, strategy in product([True, False], [cr_prune, cr_ignore]):
        m = merge(m_1, m_2, strict, strategy)
        print(f'strict: {strict}, strategy: {strategy.__name__}\t=> {m}')


def examples_original():
    """examples (labels changed) from original paper"""
    logging.getLogger().setLevel(logging.INFO)

    simulator = LPSimulator(SIMILARITIES_1, .25)

    examples = [
        (LTS_P, LTS_Q),  # example from fig 3
        (LTS_P, LTS_S),  # example from fig 4
    ]

    for i, (lts_1, lts_2) in enumerate(examples, start=1):
        m_1 = trace(lts_1, lts_2, simulator)
        m_2 = trace(lts_2, lts_1, simulator)

        logging.info(f'm_1 = {m_1}')
        logging.info(f'm_2 = {m_2}')
        logging.info(f'm_1 U m_2 = {merge(m_1, m_2)}')


def examples_paper():
    """examples used for seminar paper"""
    logging.getLogger().setLevel(logging.INFO)

    configurations = [
        ('I_J_L2', LTS_I, LTS_J, SIMILARITIES_2, .25),
        ('Q_P_L3', LTS_Q, LTS_P, SIMILARITIES_3, .25),
        ('O_U_L1', LTS_O, LTS_U, SIMILARITIES_1, .25),
        ('P_Q_L1', LTS_P, LTS_Q, SIMILARITIES_1, .25),
        ('Q_P_L1', LTS_Q, LTS_P, SIMILARITIES_1, .25),
        ('P_S_L1', LTS_P, LTS_S, SIMILARITIES_1, .25),
        ('S_P_L1', LTS_S, LTS_P, SIMILARITIES_1, .25),
        ('P_Q_L3', LTS_P, LTS_Q, SIMILARITIES_3, .25),
        ('Y_X_L1', LTS_Y, LTS_X, SIMILARITIES_1, .25),
        ('Z_X_L1', LTS_Z, LTS_X, SIMILARITIES_1, .25),
    ]

    for name, lts_1, lts_2, similarities, gap_penalty in configurations:
        out_dir = WORKING_DIRECTORY.joinpath(f'example_{name}')
        render = partial(Digraph.render, directory=out_dir, format='pdf', cleanup=True)

        out_dir.mkdir(exist_ok=True)

        simulator = LPSimulator(similarities, gap_penalty, lp_path=out_dir.joinpath('simulation.lp'))
        simulation = simulator.simulate(lts_1, lts_2)
        mapping, record = trace(lts_1, lts_2, simulator, record=True)
        logging.info(f'mapping {mapping}')

        # store setup
        with out_dir.joinpath('setup.txt').open(mode='wt') as f:
            print(f'Example: {name}\n', file=f)

            print(f'LTS_1: {repr(lts_1)}', file=f)
            print(f'LTS_2: {repr(lts_2)}', file=f)
            print(f'Simulator: {repr(simulator)}\n', file=f)

            print(f'L{repr(similarities)[1:]}\n', file=f)
            print(f'Q{repr(simulation)[1:]}\n', file=f)
            print(f'M = {mapping}', file=f)

        # render similarities
        with out_dir.joinpath('similarities.tex').open(mode='wt') as f:
            print('\\caption{Label Similarities $L$}', file=f)
            f.write(similarities.fmt(tablefmt='latex', numalign='decimal', floatfmt='.1f'))

        # render simulation
        with out_dir.joinpath('simulation.tex').open(mode='wt') as f:
            print(f'\\caption{{Simulation $Q$ of $LTS_{lts_1.name}$ by $LTS_{lts_2.name}$, p={gap_penalty}}}', file=f)
            f.write(simulation.fmt(tablefmt='latex', numalign='decimal', floatfmt='.4f'))

        # render call graph
        dot = Digraph(engine='dot')
        dot.attr(dpi='300')
        dot.attr('node', shape='box', style='filled', fillcolor='grey90')
        record.dot(dot)
        render(dot, 'callgraph')

        record.tex(out_dir)
        with out_dir.joinpath('caption_callgraph.tex').open(mode='wt') as f:
            print(
                f'\\caption{{Callgraph of ${{\\textsc{{Trace}}}}\\left('
                f'{lts_1.initial_state}, {lts_2.initial_state}'
                f'\\right)$}}',
                file=f)

        # render mapping
        edge_kwargs = {}
        for color, eq_class in zip(cycle(PALETTE), equivalence_classes(mapping)):
            for label in eq_class:
                edge_kwargs[label] = {'color': color, 'penwidth': '2'}

        dot = Digraph(engine='dot')
        dot.attr(compound='true', dpi='150')
        dot.attr('node', shape='circle', style='filled', fillcolor='grey70')
        dot.attr('edge', arrowhead='normal')

        # invisible dummy cluster
        with dot.subgraph(name='cluster_global') as gc:
            gc.attr(style='invis')
            gc.node('global_void', label='', style='invis')

        for i, lts in enumerate((lts_1, lts_2), start=1):
            with dot.subgraph(name=f'cluster_lts_{i}') as lc:
                lc.attr(label=lts.name)
                lc.attr(style='filled,rounded', color='grey95')
                lts.dot(lc, prefix=f'cluster_lts_{i}', void='global_void', **edge_kwargs)

        render(dot, 'mapping'.lower())

        with out_dir.joinpath('caption_mapping.tex').open(mode='wt') as f:
            m_str = ', '.join(map(lambda t: '(' + ','.join(t) + ')', mapping))
            print('\\caption{Mapping \\\\ $', file=f, end='')
            print(f'M = \\left\\lbrace {m_str} \\right\\rbrace', file=f, end='')
            print('$}', file=f)


if __name__ == '__main__':
    example_bisimulation_strategies()
    example_merging_strategies()
    examples_original()
    examples_paper()
