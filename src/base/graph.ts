export enum VertexState {
    UNDISCOVERED,
    DISCOVERED,
}
export interface Vertex {
    edges: Vertex[];
    state: VertexState;
}

function dfs(v_entry: Vertex, cond: (v:Vertex) => boolean): Vertex {
    v_entry.state = VertexState.DISCOVERED;

    for (let vertex of v_entry.edges) {
        if (vertex.state == VertexState.DISCOVERED) {
            continue;
        }
        vertex.state = VertexState.DISCOVERED;
        if (cond(vertex)) {
            return vertex;
        }
        return dfs(vertex, cond);
    }

    return v_entry
}
