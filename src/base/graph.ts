

export enum VertexState {
    UNDISCOVERED,
    DISCOVERED,
}
export interface Vertex {
    edges: Vertex[];
    state: VertexState;
}

function dfs(v_entry: Vertex, cond: (v:Vertex) => boolean) {
    v_entry.state = VertexState.DISCOVERED;

    for (let vertex of v_entry.edges) {
        vertex.state = VertexState.DISCOVERED;
        if (cond(vertex)) {
            break;
        }
        dfs(vertex, cond);
    }
}
