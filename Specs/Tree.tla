---- MODULE Tree ----
(* N-Ary Tree with usual operations except delete nodes *)
(* but in WGSLTr it's not a problem cause only analysis *)
(* will WGSLTr to do on Trees. *)
CONSTANTS NULL

LOCAL INSTANCE TLC
LOCAL INSTANCE Naturals
LOCAL INSTANCE Sequences
LOCAL INSTANCE FiniteSets

LOCAL CODOMAIN(f) ==
    {f[x]: x \in DOMAIN f}
LOCAL Root(Nodes) == Nodes[1]
LOCAL TreeRelations(Nodes) ==
    \* DOMAIN is the set of all nodes
    \* that is parent, NULL indicate
    \* nodes that been deleted.
    \*
    \* Restrict maximum number of Children
    \* to make it enumerable.
    LET Children == (UNION {[1..n -> Nodes]: n \in 0..Cardinality(Nodes)})
    IN  [Nodes -> Children \union {NULL}]

LOCAL Descdent(Node, relation) ==
    LET D[N \in DOMAIN relation] ==
        IF relation[N] = <<>>
          THEN {} ELSE
          LET nodes == DOMAIN relation
              Childs == {c: c \in {relation[N][i]: i \in 1..Len(relation[N])} }
              InDirect == {D[c]: c \in Childs}
          IN (UNION InDirect) \union Childs
    IN D[Node]

\* 1. Every node can have infinite childs
\* 2. A child can have only one parent.
 \* 3. Root has no parent.
\* 4. Every nodes except Root is descdent of Root
LOCAL IsTree(relation) ==
    LET nodes == {n \in DOMAIN relation: relation[n] /= NULL}
        all_seqs == [1..Cardinality(nodes) -> nodes]
    IN
        \* Assert that a node unable to be a Parent of itself.
        /\ \A x \in nodes:
             \A i \in 1..Len(relation[x]): relation[x][i] /= x
        \* Assert that only one path from parent to it's child.
        /\ \A x \in nodes:
            \A i \in 1..Len(relation[x]):
            \A j \in 1..Len(relation[x]) \ {i}:
                relation[x][i] /= relation[x][j]
        \* Assert that every node has only one parent.
        /\ \A x \in nodes:
           \A y \in nodes \ {x}:
           \A i \in 1..Len(relation[x]):
           \A j \in 1..Len(relation[y]):
                relation[x][i] /= relation[y][j]
        \* Assert that root has no parent,
        /\ \E ordering \in all_seqs:
            \A x \in nodes:
            \A i \in 1..Len(relation[x]):
                Root(ordering) /= relation[x][i]
        \* Assert that no cycle
        /\ \A x \in nodes:
               \A y \in nodes \ {x}:
                 x \in CODOMAIN(relation[y]) =>
                   y \notin CODOMAIN(relation[x])
        \* Assert that every nodes except Root is descdent of Root
        \* which make sure a relation correspond to only one Tree.
        /\ \E ordering \in all_seqs:
           \A x \in nodes:
                \/ x = Root(ordering)
                \/ x \in Descdent(Root(ordering), relation)

InTree(T) == IsTree(T)

(*Operations*)
Singleton(n) == (n :> <<>>)
IsExists(T, Node) ==
    \E r \in DOMAIN T: Node = r /\ T[r] /= NULL
LOCAL IsExists_(T, Node) ==
    \E r \in DOMAIN T: Node = r


IsChild(T, Node, Child) ==
  IF InTree(T) /\ IsExists(T, Node) /\ IsExists(T, Child)
  THEN \E n \in DOMAIN T: \E i \in 1..Len(T[n]): T[n][i] = Child
  ELSE FALSE

AddNode(T, Node, New) ==
    IF InTree(T) /\ IsExists(T, Node)
    THEN LET New_Tree == T @@ (New :> <<>>)
         IN  IF IsExists_(T, New)
             THEN [New_Tree EXCEPT ![Node] = Append(New_Tree[Node], New),
                                   ![New] = <<>>]
             ELSE [New_Tree EXCEPT ![Node] = Append(New_Tree[Node], New)]
    ELSE T

Delete(T, Node) ==
    IF InTree(T) /\ IsExists(T, Node)
    THEN LET parent == CHOOSE n \in DOMAIN T: IsChild(T, n, Node)
             F[i \in 0..Len(T[parent])] ==
               IF i = 0
               THEN <<>>
               ELSE IF T[parent][i] /= Node
                    THEN Append(F[i-1], T[parent][i])
                    ELSE F[i-1]

         IN [T EXCEPT ![Node] = NULL,
                      ![parent] = F[Len(T[parent])]]
    ELSE T

\* TODO: Implement Replace for Tree in Specification
Replace(T, Node, T_NEW) == Assert(FALSE, "Need to implement")

GetChild(T, Node, N) ==
    IF IsExists(T, Node) /\ Len(T[Node]) >= N
    THEN T[Node][N]
    ELSE NULL

NumOfChild(T, Node) ==
    IF /\ IsExists(T, Node)
       /\ Len(T[Node]) > 0
    THEN Len(T[Node])
    ELSE 0

GetRoot(T) ==
    LET nodes == DOMAIN T
        RootSet == {n \in nodes: \A y \in nodes \ {n}: y \in Descdent(n, T)}
    IN IF Cardinality(RootSet) = 1 \* There should only one root
       THEN CHOOSE n \in nodes: \A y \in nodes \ {n}: y \in Descdent(n, T)
       ELSE NULL

(*Tree Property Predicates*)
UniqueParent(T) ==
      LET nodes == DOMAIN T
      IN  \A x \in nodes:
          \A y \in nodes\{x}:
            T[x] /= <<>> /\ T[y] /= <<>> =>
                \A i \in 1..Len(T[x]):
                \A j \in 1..Len(T[y]):
                    T[x][i] /= T[y][j]
Ring(T) ==
    IF CODOMAIN(T) = {<<>>}
    THEN FALSE
    ELSE UniqueParent(T) /\
         \A y \in Descdent(GetRoot(T), T):
           GetRoot(T) \in CODOMAIN(T[y])

SingleTree(T) ==
    LET nodes == DOMAIN T
        all_seqs == [1..Cardinality(nodes) -> nodes]
    IN \E ordering \in all_seqs:
       \A x \in nodes:
         \/ x = Root(ordering)
         \/ x \in Descdent(Root(ordering), T)
=====================
