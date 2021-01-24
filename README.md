# Trace

This is an implementation of `Trace`, an activity mapping algorithm for business 
process diagrams (BPD) proposed by Becker et al [1]. The work was done for the 
seminar _Selected Topics in Process Mining_ [2] by the chair of _Process and Data
Science_ in winter semester 2020/21. Although this implementation has no focus on 
performance, it should scale fairly good with the right solver backend chosen.

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


[1] J. Becker, D. Breuker, P. Delfmann, H.-A. Dietrich, and M. Steinhorst, “Identifying business process activity
mappings by optimizing behavioral similarity,” 2012.

[2] https://www.pads.rwth-aachen.de/go/id/rbxa
