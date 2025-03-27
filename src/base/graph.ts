export enum VertexState {
    UNDISCOVERED = 0,
    DISCOVERED = 1,
    CLOSED = 2,
}
export interface Vertex {
    edges: Vertex[];
    state: VertexState;
}


export function dfs(v_entry: Vertex,
                    convergent_cond: (v:Vertex) => boolean): Vertex[] {

    let vs: Vertex[] = [];
    dfsInternal(v_entry, convergent_cond, vs);
    return vs;
}

function dfsInternal(v_entry: Vertex,
                    convergent_cond: (v:Vertex) => boolean,
                    vertexs: Vertex[]) {

    /* Iterate all vertexs and close vertex once exits from the node */
    v_entry.state = VertexState.DISCOVERED;
    if (convergent_cond(v_entry)) {
        return v_entry;
    }
    for (let u of v_entry.edges) {
        if (u.state == VertexState.UNDISCOVERED) {
            let v = dfsInternal(u, convergent_cond, vertexs);
            if (v != null) {
                vertexs.push(v);
            }
        }
    }
    v_entry.state = VertexState.CLOSED;

    return null;
}
