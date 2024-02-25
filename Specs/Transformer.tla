------ MODULE Transformer -------
\* This specification is also a specification of
\* how api layer of chameleon look like.
CONSTANTS NULL
VARIABLES transformer

LOCAL INSTANCE Tree
LOCAL INSTANCE Rule

TypeInvariant(rule_, input_) ==
    /\ InTree(input_)
    /\ InRule(rule_)
    /\ transformer = [input |-> input_,
                      output |-> NULL,
                      rule |-> rule_,
                      info |-> NULL,]

Init(rule_, input_) ==
    /\ state = 0
    /\ TypeInvariant(rule_, input_)

Transform(info_) ==
    \* TODO: use input as output directly, due how to do transformation
    \*       is still not specified yet.
    /\ [transformer' EXCEPT ![output] = transformer[input]

Steps(info_) == Transform(info_)
Spec(rule_, input_, info_) ==
    /\ Init(rule_, input_)
    /\ [][Steps(info_)]_{transformer}

=================================
