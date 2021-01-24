import logging
import os
from abc import ABCMeta, abstractmethod
from collections import UserDict
from itertools import product, count, chain, combinations
from math import isclose
from pathlib import Path
from time import monotonic
from typing import Any, Iterable, Tuple, Optional, Set, Callable, Union, FrozenSet

from graphviz import Digraph
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, lpSum as lp_sum, getSolver as get_solver
from tabulate import tabulate

BiNode = Tuple[str, str]
Labels = Tuple[str, str]
Mapping = Set[Labels]

WORKING_DIRECTORY = Path(os.getenv('WORKING_DIRECTORY', os.getcwd()))
logging.debug(f'working directory: {WORKING_DIRECTORY}')


def timeit(function: Callable[[], Any]) -> Tuple[float, Any]:
    """measure duration of execution of given function"""
    t_start = monotonic()
    result = function()
    t_end = monotonic()
    return t_end - t_start, result


class DefaultDict(UserDict):
    """Alternative to `collections.defaultdict` that respects index when creating default items"""

    def __init__(self, factory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._factory = factory

    def __missing__(self, key):
        self.data[key] = self._factory(key)
        return self.data[key]


class SimilarityFunction(metaclass=ABCMeta):
    """Abstract protocol for similarity functions"""

    @abstractmethod
    def lookup(self, item_1: str, item_2: str) -> float:
        """Lookup similarity of two items"""
        raise NotImplementedError

    def __getitem__(self, items):
        return self.lookup(*items)


class SimilarityTable(SimilarityFunction):
    """A similarity function using a finite lookup table"""

    @staticmethod
    def _make_index(item_1: str, item_2: str) -> BiNode:
        item_1, item_2 = sorted([item_1, item_2])
        return item_1, item_2

    def __init__(self, similarities: Iterable[Tuple[str, str, float]], default: float = 0.):
        """
        :param similarities: pairwise similarities of items
        :param default: default similarity of all items that are not in table
        """
        similarities = list(similarities)
        self._rows = sorted({l for (l, _, _) in similarities})
        self._cols = sorted({l for (_, l, _) in similarities})
        self._table = {self._make_index(*l): s for (*l, s) in similarities}
        self._default = default

    def __str__(self):
        return f'{type(self).__name__}(...)'

    def fmt(self, **kwargs) -> str:
        """Format as table, accepts key-value pairs for `tabulate.tabulate`"""
        return tabulate(
            ({col: self.lookup(row, col) for col in self._cols} for row in self._rows),
            headers='keys',
            showindex=self._rows,
            **kwargs
        )

    def __repr__(self):
        return self.fmt(numalign='decimal', floatfmt='.4f')

    def lookup(self, item_1: str, item_2: str) -> float:
        try:
            return self._table[self._make_index(item_1, item_2)]
        except KeyError:
            return self._default


class LTS:
    """Labelled Transition System"""

    __id_ctr = count(1, 1)

    def __init__(self, transitions: Iterable[Tuple[str, str, str]], initial_state: str, *, name=None):
        """
        :param transitions: labelled transitions: state --label-> state
        :param initial_state: initial state
        :param name: optional name for LTS
        """
        self._id = next(LTS.__id_ctr)
        self._name = name or f'LTS{self._id}'
        self._transitions = set(transitions)
        self._states = {s for (s, _, _) in transitions} | {s for (_, _, s) in transitions} | {initial_state}
        self._labels = {o for (_, o, _) in transitions}
        self._initial_state = initial_state

        self._lookup = DefaultDict(lambda _: dict())
        for (s_1, o, s_2) in transitions:
            self._lookup[s_1] = {o: s_2, **self._lookup[s_1]}

    def __getitem__(self, item):
        if isinstance(item, tuple):
            state, label, *void = item

            if void:
                raise KeyError(f'unexpected keys: {void}')

            return self._lookup[state][label]

        else:
            return self._lookup[item]

    def __str__(self):
        return f'{self._name}(S: {len(self._states)}, O: {len(self._labels)}, T: {len(self._transitions)}, ' \
               f's_in: {self._initial_state})'

    def __repr__(self) -> str:
        return f'{self._name}(S: {self._states}, O: {self._labels}, T={self._transitions}, s_in: {self._initial_state})'

    name = property(fget=lambda self: self._name)
    states = property(fget=lambda self: self._states)
    labels = property(fget=lambda self: self._labels)
    transitions = property(fget=lambda self: self._transitions)
    initial_state = property(fget=lambda self: self._initial_state)

    def dot(self, dot: Optional[Digraph] = None, prefix: str = '', void: Optional[str] = None,
            **edge_kwargs) -> Digraph:
        """
        Render DOT representation of LTS to given or fresh DOT object.

        :param dot: DOT object
        :param prefix: prefix used for state labels
        :param void: optional, invisible state for initial transition
        :param edge_kwargs: edge parameters
        :return: DOT object
        """
        dot = dot or Digraph(self.name)

        if void is None:
            void = f'{prefix}_void'
            dot.node(void, style='invis')

        for state in sorted(self.states):
            dot.node(f'{prefix}_{state}', label=state)

        dot.edge(void, f'{prefix}_{self.initial_state}')

        for s1, l, s2 in sorted(self.transitions):
            dot.edge(f'{prefix}_{s1}', f'{prefix}_{s2}', label=l, **edge_kwargs.get(l, {}))

        return dot


def s_average(a: float, b: float) -> float:
    """merging strategy for `Simulator`s that averages bisimilarities"""
    return (a + b) / 2


def s_first(a: float, _: float) -> float:
    """merging strategy for `Simulator`s that just takes the first value"""
    return a


def s_second(_: float, b: float) -> float:
    """merging strategy for `Simulator`s that just takes the second value"""
    return b


class Simulator(metaclass=ABCMeta):
    """Abstract protocol for (bi-)simulation algorithms"""

    def __init__(self, similarities: SimilarityFunction, gap_penalty: float,
                 strategy: Callable[[float, float], float] = min):
        """
        :param similarities: label similarity function
        :param gap_penalty: penalty for one LTS skipping during bisimulation
        :param strategy: strategy for merging bisimilarities of both simulaitons
        """
        self._similarities = similarities
        self._gap_penalty = gap_penalty
        self._strategy = strategy

    def __repr__(self):
        strategy = getattr(self._strategy, '__name__', '???')
        return f'{type(self).__name__}' \
               f'(similarity_function: {self.similarities}, gap_penalty: {self.gap_penalty}, strategy: {strategy})'

    @abstractmethod
    def simulate(self, lts_1: LTS, lts_2: LTS) -> SimilarityTable:
        """
        Let the simuator lts_2 simulate the simulatee lts_1

        :param lts_1: simulatee
        :param lts_2: simulator
        :return: pairwise state similarities
        """
        raise NotImplementedError

    def bisimulate(self, lts_1: LTS, lts_2: LTS) -> SimilarityTable:
        """
        Let both LTS simulate each other (equation 4)

        :param lts_1: labeled transition system I
        :param lts_2: labeled transition system II
        :param strategy: strategy for how to merge simulation coefficients, `min` by default
        :return: degree to what both LTS can simulate each other
        """
        t_1 = self.simulate(lts_1, lts_2)
        t_2 = self.simulate(lts_2, lts_1)

        return SimilarityTable(
            (s_i, s_j, self._strategy(t_1[s_i, s_j], t_2[s_j, s_i])) for s_i, s_j in product(lts_1.states, lts_2.states)
        )

    gap_penalty = property(fget=lambda self: self._gap_penalty)
    similarities = property(fget=lambda self: self._similarities)


class RecursiveSimulator(Simulator):
    """
    Recursive implementation of bisimulation for labeled transition systems as described by equation 1-3.

    NOTE: This method only works if the LTS is sufficiently small and cycle free!
    """

    def simulate(self, lts_1: LTS, lts_2: LTS) -> SimilarityTable:

        def _q(s_i: str, s_j: str) -> float:
            if not lts_1[s_i]:
                return 1.
            return max(w_1(s_i, s_j), w_2(s_i, s_j))

        # reduce the actual evaluations of `_q`
        q_cache = DefaultDict(lambda a: _q(*a))

        def q(s_i: str, s_j: str) -> float:
            """equation 1"""
            return q_cache[s_i, s_j]

        def w_1(s_i: str, s_j: str) -> float:
            """equation 2"""
            return max((self.gap_penalty * q(s_i, s_m) for s_m in lts_2[s_j].values()), default=.0)

        def w_2(s_i: str, s_j: str) -> float:
            """equation 3"""

            def _inner():
                for (a, s_k) in lts_1[s_i].items():
                    yield max(
                        max(((self.similarities[a, b] * q(s_k, s_m)) for (b, s_m) in lts_2[s_j].items()), default=.0),
                        self.gap_penalty * q(s_k, s_j)
                    )

            return sum(_inner()) / len(lts_1[s_i])

        try:
            q(lts_1.initial_state, lts_2.initial_state)

        except RecursionError:
            raise ValueError('recursive bisimulation only works for smaller, cycle free transition graphs')

        return SimilarityTable((s_i, s_j, s) for (s_i, s_j), s in q_cache.items())


class LPSimulator(Simulator):
    """
    Formulate (bi-)simulation of labelled transition system as an optimization problem aka linear program. See equations
    1-4.
    """

    def __init__(self, *args, backend: str = 'PULP_CBC_CMD', lp_path: Optional[Union[Path, str]] = None, **kwargs):
        """
        :param args: positional arguments for `Simulator`
        :param backend: which backend to use, e.g. `'GUROBI_CMD'`, defaults to `'PULP_CBC_CMD'`
        :param lp_path: optional directory where to store linear program file
        :param kwargs: kew word arguments for `Simulator`
        """
        super().__init__(*args, **kwargs)
        self._backend = backend
        self._lp_path = lp_path

    def _simulate(self, lts_1: LTS, lts_2: LTS) -> SimilarityTable:
        def t2s(t: Tuple[str, str, str]) -> str:
            """format a transition"""
            s_i, label, s_j = t
            return f'({s_i},{label},{s_j})'

        # namespace where all variables of the linear program live in
        # implies boundaries for all Q_ij and X_ej
        ns = DefaultDict(lambda name: LpVariable(str(name), 0., 1.))

        # derive the linear program from transition systems
        lp = LpProblem(f'simulate_{lts_1.name}_by_{lts_2.name}', LpMinimize)

        # expression to minimize: sum of all Q_ij (equation 5)
        lp += (
            lp_sum((ns[f'Q_{s_i}_{s_j}'] for s_i, s_j in product(lts_1.states, lts_2.states))),
            'EXPR__min'
        )

        # constraint: LTS1 skips (equation 6)
        for s_i, s_j in product(lts_1.states, lts_2.states):
            q_ij = ns[f'Q_{s_i}_{s_j}']
            for s_m in set(lts_2[s_j].values()):
                q_im = ns[f'Q_{s_i}_{s_m}']
                lp += q_ij >= self.gap_penalty * q_im, f'CSTR_1__({s_i}, {s_j}, {s_m})'

        # constraint: cases where LTS1 moves (equation 7)
        for s_i, s_j in product(lts_1.states, lts_2.states):
            q_ij = ns[f'Q_{s_i}_{s_j}']
            if n := len(lts_1[s_i]):
                lp += (
                    q_ij >= (1 / n) * lp_sum(ns[f'X_{t2s((s_i, k, v))}_{s_j}'] for (k, v) in lts_1[s_i].items()),
                    f'CSTR_2__({s_i},{s_j})'
                )

        # constraint: LTS2 skips (equation 8)
        for t_i, s_j in product(lts_1.transitions, lts_2.states):
            _, _, s_k = t_i
            q_kj = ns[f'Q_{s_k}_{s_j}']
            x_ej = ns[f'X_{t2s(t_i)}_{s_j}']
            lp += (
                x_ej >= self.gap_penalty * q_kj,
                f'CSTR_3__({t2s(t_i)},{s_j})'
            )

        # constraint: both LTS move & emit mapping (a,b) (equation 9)
        for t_i, t_j in product(lts_1.transitions, lts_2.transitions):
            s_i, a, s_k = t_i
            s_j, b, s_m = t_j
            x_ej = ns[f'X_{t2s(t_i)}_{s_j}']
            q_km = ns[f'Q_{s_k}_{s_m}']
            lp += (
                x_ej >= self.similarities[a, b] * q_km,
                f'CSTR_4__({t2s(t_i)}, {t2s(t_j)})'
            )

        # constraint: Q_ij is 1 for final states (equation 12)
        for s_i, s_j in product(lts_1.states, lts_2.states):
            if not lts_1[s_i]:
                q_ij = ns[f'Q_{s_i}_{s_j}']
                lp += (
                    q_ij == 1.,
                    f'CSTR_5__({s_i}, {s_j})'
                )

        # optionally store lp
        if path := self._lp_path:
            lp.writeLP(str(path))

        # search for a solution
        logging.info(f'start solver ({lp.numVariables()} variables, {lp.numConstraints()} constraints)')
        duration, *_ = timeit(lambda: lp.solve(get_solver(self._backend, msg=False, threads=16)))

        logging.info(f'status: {LpStatus[lp.status]} ({duration:.3f}s)')
        assignments = {v.name: v.varValue for v in lp.variables()}

        return SimilarityTable(
            (s_i, s_j, assignments[f'Q_{s_i}_{s_j}']) for s_i, s_j in product(lts_1.states, lts_2.states)
        )

    def simulate(self, lts_1: LTS, lts_2: LTS) -> SimilarityTable:
        logging.info(f'simulate {lts_1.name} by {lts_2.name}')
        duration, similarities = timeit(lambda: self._simulate(lts_1, lts_2))
        logging.info(f'complete simulation in {duration:.3f}s')
        return similarities


class _StepRecord:
    """Container for capturing one step of `map_labels`"""

    def __init__(self, node: BiNode):
        self.active_node = node
        self.active_successors = set()
        self.active_mappings = dict()

    def add(self, node_1: BiNode, node_2: BiNode, labels: Optional[Labels] = None):
        """Add new movement to record"""
        self.active_successors.add(node_2)
        self.active_mappings[node_1, node_2] = {labels, *self.active_mappings.get((node_1, node_2), set())}


class TraceRecord:
    """Container for capturing a full execution of `LPSimulator.simulate`"""

    def __init__(self, lts_1: LTS, lts_2: LTS, q: SimilarityFunction, l: SimilarityFunction, p: float):
        self._q = q
        self._l = l
        self._p = p
        self._lts_1 = lts_1
        self._lts_2 = lts_2
        self._nodes = dict()
        self._edges = DefaultDict(lambda _: DefaultDict(lambda _: []))
        self._staging: Optional[_StepRecord] = None

    def __enter__(self):
        if not self._staging:
            raise ValueError('no step is staging, call `open` first')
        return self._staging

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self, node: BiNode) -> 'TraceRecord':
        """"""
        if self._staging:
            self.close()

        self._staging = _StepRecord(node)
        return self

    def close(self):
        if step := self._staging:
            s_i, s_j = step.active_node

            self._nodes[s_i, s_j] = True

            successors = chain(
                ((s_i, s_m, self._p, None) for s_m in self._lts_2[s_j].values()),
                ((s_k, s_j, self._p, None) for s_k in self._lts_1[s_i].values()),
                ((s_k, s_m, self._l[a, b], (a, b)) for (a, s_k), (b, s_m) in
                 product(self._lts_1[s_i].items(), self._lts_2[s_j].items()))
            )

            successors = list(successors)

            for (s_k, s_m, w_val, l) in successors:
                active = ((a := step.active_mappings.get(((s_i, s_j), (s_k, s_m)))) is not None) and (l in a)
                self._nodes[(s_k, s_m)] = (s_k, s_m) in step.active_successors
                self._edges[(s_i, s_j)][(s_k, s_m)].append({'weight': w_val, 'active': active, 'labels': l})

            self._staging = None

        else:
            raise ValueError('no step is staging')

    def dot(self, dot: Digraph):
        """Render DOT representation of all steps"""
        for (s_i, s_j), active in sorted(self._nodes.items()):
            dot.node(
                f'{s_i}_{s_j}',
                label=f'Q({s_i},{s_j}) = {self._q[s_i, s_j]:.2E}',
                fontsize='12.0',
                **dict(style='filled', fillcolor='grey65') if active else {}
            )

        for (s_i, s_j), successors in self._edges.items():
            for (s_k, s_m), attributes in successors.items():
                for attr in attributes:
                    w_tex = attr['weight']
                    kwargs = {'label': f'{w_tex:.2f}', 'fontsize': '10.0'}

                    if labels := attr['labels']:
                        a, b = labels
                        kwargs['label'] += f'\n({a},{b})'

                    if attr['active']:
                        kwargs['penwidth'] = '3'

                    dot.edge(f'{s_i}_{s_j}', f'{s_k}_{s_m}', **kwargs)

    def tex(self, tex: Path):
        """Render TEX representation of all steps"""
        for i, ((s_i, s_j), successors) in enumerate(self._edges.items(), start=1):
            with tex.joinpath(f'trace_{i}.tex').open(mode='wt') as f:
                print(f'\\caption{{${{\\textsc{{Trace}}}}\\left({s_i},{s_j}\\right)$}}', file=f)

                n = len(self._lts_1[s_i])
                table = []

                for (s_k, s_m), attributes in successors.items():
                    for attr in attributes:
                        action = '?'
                        w_val = attr['weight']
                        w_tex = 'p'
                        c_val = 1.
                        c_tex = ''

                        if labels := attr['labels']:
                            action = f'${self._lts_1.name}$, ${self._lts_2.name}$ move'
                            w_tex = f'\\frac{{1}}{{n}} \\cdot L(\\text{{{labels[0]}}},\\text{{{labels[1]}}})'
                            c_val = 1. / n
                            c_tex = f'{c_val:.2f} \\cdot '

                        elif s_i == s_k:
                            action = f'${self._lts_1.name}$ skips'

                        elif s_j == s_m:
                            action = f'${self._lts_2.name}$ skips'
                            w_tex = '\\frac{{1}}{{n}} \\cdot p'
                            c_val = 1. / n
                            c_tex = f'{c_val:.2f} \\cdot '

                        table.append([
                            action,
                            f'${w_tex} \\cdot Q_{{{s_k},{s_m}}} = $',
                            f'${c_tex}{w_val:.2f} \\cdot {self._q[s_k, s_m]:.4f}$',
                            f'$= {self._q[s_k, s_m] * w_val * c_val:.4f}$',
                            '\\textbf{select}' if attr['active'] else 'discard',
                            f'$(\\text{{{labels[0]}}},\\text{{{labels[1]}}})$' if attr['active'] and (
                                labels := attr['labels']) else ''
                        ])

                table.append(['$\\Sigma$', '', '', f'$= {self._q[s_i, s_j]:.4f}$', '', ''])

                f.write(tabulate(
                    table,
                    tablefmt="latex_raw",
                    colalign=('right', 'right', 'right', 'left', 'right', 'center')
                ))


class DummyRecord:
    """Drop-in replacement for `TraceRecord`"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def open(self, *args, **kwargs):
        return self

    def add(self, *args, **kwargs):
        pass


def trace(lts_1: LTS, lts_2: LTS, simulator: Simulator, bisimulation: bool = False, record: bool = False) \
        -> Union[Mapping, Tuple[Mapping, TraceRecord]]:
    """
    Procedural implementation of the label mapping algorithm

    :param lts_1: simulatee LTS
    :param lts_2: simulator LTS
    :param simulator: algorithm used for (bi-)simulation
    :param bisimulation: whether or not to employ a full bisimulation
    :param record: whether or not to record steps of the algorithm
    :return: a label mapping and optionally a record
    """
    logging.info('compute mapping (procedural)')
    logging.info(str(lts_1))
    logging.info(str(lts_2))

    logging.info(f'Simulate {lts_1.name} by {lts_2.name}: {simulator}')
    q = simulator.bisimulate(lts_1, lts_2) if bisimulation else simulator.simulate(lts_1, lts_2)

    logging.debug(f'Q\n{repr(q)}')

    p = simulator.gap_penalty
    l = simulator.similarities

    record = TraceRecord(lts_1, lts_2, q, l, p) if record else DummyRecord()

    visited = set()
    mapping = set()
    stack = [(lts_1.initial_state, lts_2.initial_state)]

    while stack:
        s_i, s_j = stack.pop()

        # skip final states
        if isclose(q[s_i, s_j], 1.):
            continue

        # skip visited states
        if (s_i, s_j) in visited:
            continue

        visited.add((s_i, s_j))

        with record.open((s_i, s_j)) as step_record:
            # LTS 1 skips
            if any(isclose(q[s_i, s_j], q[s_i, (s_m := s)] * p) for s in lts_2[s_j].values()):
                stack.append((s_i, s_m))
                step_record.add((s_i, s_j), (s_i, s_m), None)

            # LTS 1 moves
            else:
                for (a, s_k) in lts_1[s_i].items():
                    _, successors, b = max((
                        # LTS 2 skips
                        (q[s_k, s_j] * p, (s_k, s_j), None),
                        # both LTS move
                        *((l[a, b] * q[s_k, s_m], (s_k, s_m), b) for (b, s_m) in lts_2[s_j].items())
                    ))

                    stack.append(successors)
                    if b is not None:
                        mapping.add((a, str(b)))

                    step_record.add((s_i, s_j), successors, (a, b) if b else None)

    if isinstance(record, TraceRecord):
        return mapping, record

    return mapping


ConflictResolution = Callable[[Mapping, Mapping], Mapping]


def cr_ignore(mapping: Mapping, _: Mapping) -> Mapping:
    """conflict resolution strategy that ignores conflicts"""
    return mapping


def cr_prune(mapping: Mapping, conflicts: Mapping) -> Mapping:
    """conflict resolution strategy that removes conflict elements from mapping"""
    return mapping - conflicts


def cr_panic(mapping: Mapping, conflicts: Mapping) -> Mapping:
    """conflict resolution strategy that crashes when there are conflicts"""
    if conflicts:
        raise ValueError(str(conflicts))
    return mapping


def merge(m_1: Mapping, m_2: Mapping, strict: bool = True, resolution: ConflictResolution = cr_prune,
          reverse_m_2: bool = True) -> Mapping:
    """
    Merge two mappings

    :param m_1: first mapping
    :param m_2: second mapping
    :param strict: whether or not merge by intersection or union
    :param resolution: conflict resolution strategy
    :param reverse_m_2: whether or not to reverse direction of second mapping
    :return: merged mappings
    """
    _merge = set.intersection if strict else set.union
    mapping = _merge(m_1, {(b, a) for a, b in m_2} if reverse_m_2 else m_2)
    conflicts = set(chain(*((a, b) for a, b in combinations(mapping, 2) if a[0] == b[0] or a[1] == b[1])))
    return resolution(mapping, conflicts)


def equivalence_classes(mapping: Mapping) -> Set[FrozenSet[str]]:
    """
    Compute equivalence classes within a mapping

    :param mapping: any mapping
    :return: equivalence classes
    """
    eq_classes = set(map(frozenset, mapping))

    for _ in range(len(eq_classes)):
        if len(eq_classes) < 2:
            break

        c_new = set()
        c_old = set()

        for cls_1, cls_2 in combinations(eq_classes, 2):
            if frozenset.intersection(cls_1, cls_2):
                c_new.add(frozenset.union(cls_1, cls_2))
            else:
                c_old.update({cls_1, cls_2})

        eq_classes = set.union(c_new, c_old)

        # short circuit if there's no progress
        if not c_new:
            break

    return eq_classes
