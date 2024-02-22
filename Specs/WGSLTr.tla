------ MODULE WGSLTr ------

INIT == 0
Parsing == 1

Tr == Instance Transformer

TypeInvariant == /\ state = INIT

Init == /\ TypeInvariant

Parsing == /\ state = INIT
           /\ state' = Parsing

===========================
