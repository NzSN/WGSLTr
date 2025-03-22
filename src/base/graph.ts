

export enum VertexState {
    UNDISCOVERED,
    DISCOVERED,
}
export interface Vertex {
    edges: Vertex[];
    state: VertexState;
}


function dfs(v_entry: Vertex, cond: (v:Vertex) => boolean): Vertex | null {
    v_entry.state = VertexState.DISCOVERED;

    for (let vertex of v_entry.edges) {
        vertex.state = VertexState.DISCOVERED;
        if (cond(vertex)) {
            return vertex;
        }
        let r = dfs(vertex, cond);
        if (r != null) {
            return r;
        }
    }
    return null;
}
