import { Node } from "web-tree-sitter";
import { Module } from "../module";
import { Searcher } from "../parser/parser";


export class Semantic {
    public static overrideVerify(mod: Module, n: Node): void | never {
        if (n.type != 'primary_expression') {
            return;
        }
        // Find out an identifier that refer to an override symbol.
        let ident_searcher = new Searcher(n, 'ident');
        let ident_node = ident_searcher.searching_next(n.walk());
        if (ident_node == null) {
            return;
        }
        let var_ident = ident_node.text;
        let is_refer_to_override = Module.override_list.find(
            (s) => s == var_ident) != undefined;
        if (is_refer_to_override) {
            const overrides_def_in_current_mod =
                Module.overrides.get(mod.ident);

            const is_defined =
                mod.allExternalSymbols.find(
                    (s) => s == var_ident) != undefined ||
                (overrides_def_in_current_mod != undefined &&
                    overrides_def_in_current_mod.find(
                        (s) => s == var_ident) != undefined);

            if (!is_defined) {
                throw new Error(`Unresolved override reference: ${var_ident}`);
            }
        }
    }

    public static verify(mod: Module, n: Node): void | never {
        this.overrideVerify(mod, n);
    }
}
