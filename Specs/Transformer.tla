------ MODULE Transformer -------
\* This specification is also a specification of
\* how api layer of chameleon look like.
\*
\* I don't specify this module in detail due to
\* it already specified in Chameleon.
CONSTANTS NULL, Rules, Trees
VARIABLES transformer, state

LOCAL INSTANCE Tree
LOCAL INSTANCE Rule

TypeInvariant ==
    /\ transformer = [input |-> NULL, output |-> NULL,
                      rule |-> NULL, info |-> NULL]

InitStage == 0
SetupStage == 1
TransformStage == 2

Init ==
    /\ state = InitStage
    /\ TypeInvariant

Setup(t) ==
  /\ state = InitStage
  /\ state' = SetupStage
  /\ transformer' = [transformer EXCEPT !.input = t]

Transform ==
    \* TODO: use input as output directly, due how to do transformation
    \*       is still not specified yet.
    /\ state = SetupStage
    /\ state' = TransformStage
    /\ transformer' = [transformer EXCEPT
                       !.output = transformer.input]

Done ==
  /\ state = TransformStage
  /\ UNCHANGED <<transformer, state>>

Steps ==
  \/ \E t \in Trees: Setup(t)
  \/ Transform
  \/ Done
Spec == Init /\ [][Steps]_<<transformer, state>>

=================================
