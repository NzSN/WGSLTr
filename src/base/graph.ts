export enum VertexState {
    UNDISCOVERED,
    DISCOVERED,
    CLOSED,
}
export interface Vertex {
    edges: Vertex[];
    state: VertexState;
}


export function dfs(v_entry: Vertex,
                    convergent_cond: (v:Vertex) => boolean): Vertex | null {
    /* Iterate all vertexs and close
     * vertex once exits from the node */
    v_entry.state = VertexState.DISCOVERED;
    if (convergent_cond(v_entry)) {
        return v_entry;
    }
    for (let u of v_entry.edges) {
        if (u.state == VertexState.UNDISCOVERED) {
            let res = dfs(u, convergent_cond);
            if (res != null) {
                return res;
            }
        }
    }
    v_entry.state = VertexState.CLOSED;
    return null;
}
