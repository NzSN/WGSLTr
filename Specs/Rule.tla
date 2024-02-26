------ MODULE Rule ------
CONSTANTS NULL

LOCAL INSTANCE Sequences
LOCAL INSTANCE Naturals
LOCAL INSTANCE Tree WITH NULL <- NULL

Rule(L, F, R) == <<L, F, R>>
\* Unable to describe the type of F by TLA+
\* due to it's domain and codomain is
\* too huge.
\* Type of F should be [Tree -> Tree]
InRule(R) ==
    /\ InTree(R[1])
    /\ InTree(R[2])
SourcePattern(R) == R[1]
WhereClause(R)   == R[2]
TargetPattern(R) == R[3]

===========================
