import { Module, Symbol } from "../module";
import { Searcher } from "../parser/parser";
import { Node } from "web-tree-sitter";
import { strict as assert } from 'assert';

export class Analyzer {

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

    public static analyze(mod: Module) {
        Analyzer.analyzeOverrides(mod);
    }

    public static verify(mod: Module) {

    }
}
