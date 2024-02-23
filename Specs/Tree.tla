------ MODULE Tree ------
LOCAL INSTANCE Naturals

\* Treat tree as an object and due
\* to Tree with finite nodes isomorphic
\* with Naturals so I use Tree[i]
\* indicate a trees
MAXIMUM_INDEX == 5
Trees == {{{n}}: n \in 0..MAXIMUM_INDEX}
Tree[n \in 0..MAXIMUM_INDEX] == {{n}}

===========================
