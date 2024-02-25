------ MODULE WGSLTr ------
CONSTANTS NULL
VARIABLE treeInfo

LOCAL INSTANCE Tree
LOCAL INSTANCE Rule


Tr == Instance Transformer

Init(rule_, input_) ==
    /\ state = 0
    /\ inRule(rule_)
    /\ InTree(input_)
    /] treeInfo = NULL
    /\ Tr!Init(rule_, input_)

Analysis ==
    /\ state = Parsing
    /\ state' = Analysis
    \* TODO: Specify how to analyze ParseTree
    /\ treeInfo' = 0

Transform ==
    /\ state = Analysis
    /\ Tr!Transform(treeInfo)

Steps ==
    \E input \in Source:
        Parse(input)

Spec(rule_, input_) ==
    /\ Init(rule_, input_, funcs_)
    /\ [][Steps]_{Tr}

===========================
