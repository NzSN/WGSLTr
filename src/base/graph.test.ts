import { dfs, Vertex, VertexState } from './graph';

class TrivialVertex implements Vertex {
    public edges: TrivialVertex[] = [];
    public state: VertexState = VertexState.UNDISCOVERED;

    public setEdge(v: TrivialVertex) {
        this.edges.push(v);
    }

    public counter: number = 0;
}

describe("Graph Unittests", () => {
    test("DFS", () => {
        let Vs = [new TrivialVertex(),
                  new TrivialVertex(),
                  new TrivialVertex(),
                  new TrivialVertex(),
                  new TrivialVertex(),
                  new TrivialVertex(),];
        let k = 1
        Vs.forEach((v) => {
            if (k < Vs.length) {
                v.setEdge(Vs[k++])
            }
        })

        dfs(Vs[0], (v) => {
            (v as TrivialVertex).counter += 1;
            return false;
        });

        Vs.forEach((v) => expect(v.counter == 1).toBeTruthy());
    })

    test("DFS Circular", () => {
        let Vs = [new TrivialVertex(),
                  new TrivialVertex(),
                  new TrivialVertex(),
                  new TrivialVertex(),
                  new TrivialVertex(),
                  new TrivialVertex(),];
        let k = 1
        Vs.forEach((v) => {
            if (k < Vs.length) {
                v.setEdge(Vs[k++]);
            } else {
                v.setEdge(Vs[0]);
            }
        })

        dfs(Vs[0], (v) => {
            v.state = VertexState.CLOSED;
            (v as TrivialVertex).counter += 1;
            return false;
        });

        Vs.forEach((v) => expect(v.counter == 1).toBeTruthy());
    })

    test("DFS Circular Detect", () => {
      let Vs = [new TrivialVertex(),
                new TrivialVertex(),
                new TrivialVertex(),
                new TrivialVertex(),
                new TrivialVertex(),
                new TrivialVertex(),];
        let k = 1
        Vs.forEach((v) => {
            if (k < Vs.length) {
                v.setEdge(Vs[k++]);
            } else {
                v.setEdge(Vs[0]);
            }
        })

        let detected = false;
        let vertex = dfs(Vs[0], (v) => {
            const circular_node =
                v.edges.find((v) => v.state == VertexState.DISCOVERED)
            if (circular_node != undefined) {
                detected = true;
                return true;
            }
            return false;
        });

        expect(vertex != null).toBeTruthy();
        expect(vertex == Vs[Vs.length - 1]).toBeTruthy();
        expect(detected).toBeTruthy();
    })
})
