------ MODULE WGSLTr ------
CONSTANTS NULL, Rules, Trees
VARIABLE transformer, state, trState

LOCAL INSTANCE Tree WITH NULL <- NULL
LOCAL INSTANCE Rule WITH NULL <- NULL

Tr == INSTANCE Transformer WITH
  NULL <- NULL,
  transformer <- transformer,
  state <- trState,
  Rules <- Rules,
  Trees <- Trees

InitStage     == 0
AnalysisStage == 1
TransformStage == 2

TypeInvariant ==
  /\ Tr!TypeInvariant

Init ==
    /\ state = InitStage
    /\ Tr!Init
    /\ TypeInvariant

Analysis ==
    /\ state = InitStage
    /\ state' = AnalysisStage
    /\ UNCHANGED <<transformer, trState>>

Transform(t) ==
    /\ state = AnalysisStage
    /\ state' = TransformStage
    /\ Tr!Transform(t)

Done ==
  /\ state = TransformStage
  /\ Tr!Done
  /\ transformer.output \in Trees
  /\ UNCHANGED <<transformer, state, trState>>

Steps ==
  \/ Analysis
  \/ \E t \in Trees: Transform(t)
  \/ Done

Spec ==
    /\ Init
    /\ [][Steps]_<<transformer, state, trState>>

===========================
