import { dfs, dfsCircularDetect, Vertex, VertexState } from './graph';

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

    test("Circular Detect Case 1", () => {
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

    test("Circular Detect Case 2", () => {
        let Vs = [new TrivialVertex(),
                  new TrivialVertex(),
                  new TrivialVertex()];

        Vs[0].setEdge(Vs[1]);
        Vs[0].setEdge(Vs[2]);
        Vs[1].setEdge(Vs[2]);

        let detected = true;
        detected = dfs(Vs[0], (v) => {
            const circular_node =
                v.edges.find((v) => v.state == VertexState.DISCOVERED)
            if (circular_node != undefined) {
                detected = true;
                return true;
            }
            return false;
        }) != undefined;


        expect(!detected).toBeTruthy();
    });
})
