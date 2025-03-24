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
    let stack: Vertex[] = [v_entry];
    while (stack.length > 0) {
        let v = stack.pop()!;
        if (convergent_cond(v)) {
            return v;
        }
        if (v.state == VertexState.UNDISCOVERED) {
            v.state = VertexState.DISCOVERED;
        }
        for (let u of v.edges) {
            if (u.state != VertexState.CLOSED) {
                stack.push(u);
            }
        }
    }
    return null;
}
