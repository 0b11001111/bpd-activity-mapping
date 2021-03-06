# Trace

This is an implementation of `Trace`, an activity mapping algorithm for business 
process diagrams (BPD) proposed by Becker et al [1]. The work was done for the 
seminar _Selected Topics in Process Mining_ [2] by the chair of _Process and Data
Science_ at _RWTH Aachen University_ in winter semester 2020/21. Although this 
implementation has no focus on performance, it should scale fairly good with the 
right solver backend chosen.

**Install Dependencies**

```bash
python -m pip install -r requirements.txt
```

**Configuration**

The example, evaluation and benchmark scripts will operate in `.` by default.
To change this behaviour, set `WORKING_DIRECTORY` by

```bash
export WORKING_DIRECTORY='/path/to/working/directory'
```

**Different Solver Backend**

We use the [Pulp](https://coin-or.github.io/pulp/) solver library that comes 
with its custom solver backend. However, for larger problem instances you may 
[chose a different solver backend](https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html) 
such as [gurobi](https://www.gurobi.com/), [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)
and many more.

---

| Avtivity mapping inferred from _q_ simulating _p_ | Callgraph re-constructed by `trace` |
|:-------------------------------------------------:|:-----------------------------------:|
| ![mapping](img/mapping.svg)                       | ![callgraph](img/callgraph.svg)     |

---

**License**

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

---

References

[1] J. Becker, D. Breuker, P. Delfmann, H.-A. Dietrich, and M. Steinhorst, “Identifying business process activity
mappings by optimizing behavioral similarity,” 2012.

[2] https://www.pads.rwth-aachen.de/go/id/rbxa
