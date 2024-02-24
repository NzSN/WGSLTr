------ MODULE WGSLTr ------
CONSTANTS Source

INIT == 0
Parsing == 1
Analysis == 2
Transform == 3

LOCAL INSTANCE Tree
LOCAL INSTANCE Rule

Tr == Instance Transformer

TypeInvariant == /\ state = INIT

Init(Rule, input) ==
    /\ TypeInvariant
    /\ Tr!UseRule(Rule)
    /\ Tr!UseInput(input)

Parse(input) ==
    /\ state = INIT
    /\ state' = Parsing
    /\ Tr!Parse

Analysis ==
    /\ state = Parsing
    /\ state' = Analysis

Transform ==
    /\ state = Analysis
    /\ state = Transform
    /\ Tr!Trans

Steps ==
    \E input \in Source:
        Parse(input)

Spec ==
    /\ Init
    /\ [][Steps]_{Tr}

===========================
