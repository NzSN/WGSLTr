------ MODULE Rule ------
LOCAL INSTANCE Naturals
LOCAL INSTANCE Tree

Rules == {<<l, r>>: l \in Trees, r \in Trees}
RuleInst(l,r) == <<l,r>>
LeftPattern(r) == r[1]
RightPattern(r) == r[2]

===========================
