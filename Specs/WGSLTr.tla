------ MODULE WGSLTr ------
CONSTANTS NULL, Rules, Trees
VARIABLE transformer, state, trState

LOCAL INSTANCE Tree WITH NULL <- NULL
LOCAL INSTANCE Rule WITH NULL <- NULL
LOCAL INSTANCE Analyzer WITH
  NULL <- NULL,
  Trees <- Trees

Tr == INSTANCE Transformer WITH
  NULL <- NULL,
  transformer <- transformer,
  state <- trState,
  Rules <- Rules,
  Trees <- Trees

InitStage     == 0
SetupStage    == 1
AnalysisStage == 2
TransformStage == 3

TypeInvariant ==
  /\ Tr!TypeInvariant

Init ==
    /\ state = InitStage
    /\ Tr!Init
    /\ TypeInvariant

\* Setup a tree to be transformed
\* and rules used to do transform.
Setup(t, r) ==
    /\ state = InitStage
    /\ state' = SetupStage
    /\ Tr!Setup(t, r)

\* TODO: Specify analysis
Analysis ==
    /\ state = SetupStage
    /\ state' = AnalysisStage
    /\ transformer.input # NULL
    /\ transformer' = [transformer EXCEPT
                       !.info = Analyze(@.input)]
    /\ UNCHANGED <<trState>>

Transform ==
    /\ state = AnalysisStage
    /\ state' = TransformStage
    /\ Tr!Transform

Done ==
  /\ state = TransformStage
  /\ transformer.output \in Trees
  /\ UNCHANGED <<transformer, state, trState>>

Steps ==
  \/ Analysis
  \/ \E t \in Trees:
      \E r \in Rules: Setup(t, r)
  \/ Transform
  \/ Done

Spec ==
    /\ Init
    /\ [][Steps]_<<transformer, state, trState>>

===========================
