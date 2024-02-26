--------- MODULE Main ---------
CONSTANTS NULL
VARIABLES state, trState, transformer

LOCAL INSTANCE Tree WITH NULL <- NULL
LOCAL INSTANCE Rule WITH NULL <- NULL

SourceTree   == AddNode(AddNode(Singleton(0), 0, 1), 0, 2)
SourceTree_1 == AddNode(AddNode(Singleton(0), 0, 1), 0, 3)
TargetTree   == AddNode(AddNode(Singleton(0), 0, 2), 0, 1)
Trees == {
    SourceTree,
    TargetTree
}
WhereExpr[t \in Trees] == TRUE

RuleConfig == Rule(SourceTree, WhereExpr, TargetTree)

Rules == {RuleConfig}

Compiler == INSTANCE WGSLTr WITH
    NULL <- NULL,
    state <- state,
    trState <- trState,
    transformer <- transformer,
    Trees <- Trees,
    Rules <- Rules

Spec == Compiler!Spec
===============================
