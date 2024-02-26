------ MODULE Transformer -------
\* This specification is also a specification of
\* how api layer of chameleon look like.
CONSTANTS NULL, Rules, Trees
VARIABLES transformer, state

LOCAL INSTANCE Tree
LOCAL INSTANCE Rule

TypeInvariant ==
    /\ transformer = [input |-> NULL,
                      output |-> NULL,
                      rule |-> NULL,
                      info |-> NULL]

InitStage == 0
TransformStage == 1

Init ==
    /\ state = InitStage
    /\ TypeInvariant

Transform(input) ==
    \* TODO: use input as output directly, due how to do transformation
    \*       is still not specified yet.
    /\ state = InitStage
    /\ state' = TransformStage
    /\ transformer' = [transformer EXCEPT
                       !.input = input,
                       !.output = input]

Done ==
  /\ state = TransformStage
  /\ UNCHANGED <<transformer, state>>

Steps ==
  \/ \E t \in Trees: Transform(t)
  \/ Done
Spec == Init /\ [][Steps]_<<transformer, state>>

=================================
