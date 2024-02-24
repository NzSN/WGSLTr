---- MODULE TreeMain ----
EXTENDS TLC, Integers, FiniteSets, Sequences
CONSTANTS NULL

INSTANCE Tree WITH NULL <- NULL


Valid ==
    LET t == AddNode(AddNode(Singleton(0), 0, 1), 0, 2)
    IN  /\ Assert(~Ring(t), <<"CounterExample(2)", t>>)
        /\ Assert(SingleTree(t), <<"CounterExample(3)", t>>)
        /\ InTree(Delete(t, 2))
        /\ ~IsExists(Delete(t, 2), 2)
        /\ IsExists(AddNode(Delete(t, 2), 0, 2), 2)
        /\ GetRoot(t) = 0
=========================
