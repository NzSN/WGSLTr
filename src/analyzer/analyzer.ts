import { Module, Symbol } from "../module";
import { Searcher } from "../parser/parser";
import { Node } from "web-tree-sitter";
import { strict as assert } from 'assert';
import { VertexState, dfs } from "../base/graph";

export class Analyzer {

    /* Unimported Override may break module semantic of Extended WGSL.
     * For the purposes of gurantee of Module semantic all declarations
     * of override variables and all primary expressions that reference
     * to thoses declared overrides need to be tracked so that able to
     * recognized unresolved references to override and report as exception. */
    public static analyzeOverrides(mod: Module) {
        let override_searcher: Searcher = new Searcher(
            mod.rootNode, 'global_constant_decl');

        const overrides: Node[] =
            override_searcher.searching_all(
                mod.rootNode.walk(),
                (n:Node) => {
                    return n.type == 'global_constant_decl';
                }).filter((n: Node) => {
                    return n.children.find(
                        (n:(Node | null)) => n != null && n.type == 'override');
                });
        const override_idents: Symbol[] =
            overrides.map((o: Node) => {
                let override_token_searcher: Searcher = new Searcher(
                    o, 'override');
                let ident_searcher: Searcher = new Searcher(
                    o, 'ident');

                let override_token: Node | null =
                    override_token_searcher.searching_next(o.walk());
                assert(override_token != null);

                let idents: Node[] =
                    ident_searcher.searching_all(o.walk())
                        .filter((ident_node: Node) => {
                            return ident_node.startIndex >
                                (override_token as Node).endIndex;
                        });
                assert(idents.length > 0);
                let ident: Node =
                    idents.sort(
                        (l:Node, r:Node) => l.startIndex - r.startIndex)[0];

                return ident.text;
            });
        Module.overrides.set(mod.ident, override_idents);
        Module.override_list = Module.override_list.concat(override_idents);
    }


    /* Figure out all Modules that contain an import statement that
     * import an Modules that is an ancestor of the Module. */
    public static circularDepDetect(mod: Module) {
        dfs(mod, (m) => {
            const is_circular_point = m.edges.find((v) => {
                return v.state == VertexState.DISCOVERED
            }) != undefined;

            if (is_circular_point) {
                mod.addCircularPoint(m as Module);
            }

            return is_circular_point;
        });
    }

    public static analyze(mod: Module) {
        Analyzer.analyzeOverrides(mod);
        Analyzer.circularDepDetect(mod);
    }
}
