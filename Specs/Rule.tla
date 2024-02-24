------ MODULE Rule ------
LOCAL INSTANCE Sequences
LOCAL INSTANCE Naturals
LOCAL INSTANCE Tree

Rule(L,R) == <<L,R>>
InRule(R) == InTree(R[1]) /\ InTree(R[2])
SourcePattern(R) == R[1]
TargetPattern(R) == R[2]

===========================
