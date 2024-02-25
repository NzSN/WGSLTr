--------- MODULE Main ---------
CONSTANTS NULL

LOCAL INSTANCE Tree
LOCAL INSTANCE Rule
LOCAL INSTANCE WGSLTr


WhereExpr[t \in {SourceTree}] == TRUE
SourceTree == AddNode(AddNode(Singleton(0), 0, 1), 0, 2)
RuleConfig == Rule(AddNode(AddNode(Singleton(0), 0, 1), 0, 2),
                   WhereExpr,
                   AddNode(AddNode(Singleton(0), 0, 2), 0, 1))
Spec == WGSLTr!Spec(RuleConfig, SourceTree)

===============================
