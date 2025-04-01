import { readFileSync } from 'node:fs';
import { strict as assert } from 'assert';
import { WASM_MODULE_PATHS } from '../constants';
import { Parser, Language, Tree, TreeCursor, Node } from 'web-tree-sitter';
import { Module } from '../module';
import { relativeModPath } from './utility';
import { Analyzer } from '../analyzer/analyzer';
import { mod_group } from '../module_group';
import Path from 'path';
import { Subject } from '../base/observer';

type WGSLNodeType = string;

export class WGSLParser extends Subject<Module>  {
    private parser_: Parser | null = null;
    private typescript_lang: Language | null = null;

    constructor() {
        super();
    }

    public async parse(source: string): Promise<Tree | null> {
        if (this.parser_ == null) {
            await Parser.init();
            this.parser_ = new Parser();

            this.typescript_lang =
                await Language.load(WASM_MODULE_PATHS.TreeSitterWGSL);
            this.parser_.setLanguage(this.typescript_lang);
        }
        return this.parser_.parse(source);
    }

    private async parseAsModuleFromFileInternal(path: string): Promise<Module | null> {
        let outdated_mod: Module | null = null;
        let need_rebuild_graph = false;

        const abs_path = Path.resolve(path);
        const mod_id = Module.getIdentByPath(abs_path);
        if (mod_group.identExists(mod_id)) {
            const mod = mod_group.search_by_id(mod_id) as Module;
            if (!(await mod.isOutOfDate())) {
                return mod;
            } else {
                // Module is out of date
                outdated_mod =
                    mod_group.search_by_id(mod_id) as Module;
                assert(outdated_mod != null);
                mod_group.destruct_by_id(mod_id);
                need_rebuild_graph = true;
            }
        }
        // Module not exists or out of date
        let source_content = readFileSync(
            path, {encoding: 'utf8', flag: 'r'});
        const mod = await this.parseAsModuleInternal(abs_path, source_content);

        if (need_rebuild_graph) {
            assert(outdated_mod != null);
            assert(mod != null);

            outdated_mod.getAllDepBy().forEach((m: Module) => {
                this.linkModule(m, mod);
            });
        }
        return mod;
    }

    private async parseAsModuleInternal(
        path: string, source: string): Promise<Module | null> {

        let tree = await this.parse(source);

        assert(tree != null);

        let mod = await Module.build(path, tree);
        this.notifyAllObservers(mod);

        let s_import: Searcher = new Searcher(
            tree.rootNode, 'import');

        const imports: Node[] =
            s_import.searching_all(
                tree.rootNode.walk(),
                (n:Node) => {
                    return n.type == 'import' ||
                           n.parent?.type == 'translation_unit';
                }).filter((n: Node) => n.isNamed);

        for (const n of imports) {
            await this.parseExternalSymbols(mod, n);
        }

        return mod;
    }

    public async parseAsModule(path: string): Promise<Module | null> {
        let m: Module | null = null;
        m = await this.parseAsModuleFromFileInternal(path);

        if (m != null) {
            Analyzer.analyze(m);
        }

        return m;
    }

    // Build Relation such that l_mod is depend on r_mod
    private linkModule(l_mod: Module, r_mod: Module) {
        if (l_mod.isDepOn(r_mod, false)) {
            const dep_mod = l_mod.getDep(r_mod.ident);
            if (dep_mod?.equal(r_mod)) {
                return true;
            } else {
                l_mod.removeDep(r_mod.ident);
                l_mod.dep(r_mod);
            }
        }
        if (r_mod.isDepBy(l_mod, false)) {
            const dep_by_mod = r_mod.getDepBy(l_mod.ident);
            if (dep_by_mod?.equal(l_mod)) {
                return true;
            } else {
                r_mod.removeDepBy(l_mod.ident);
                r_mod.depBy(l_mod);
            }
        }
        if (!l_mod.isDepOn(r_mod, false) && !r_mod.isDepBy(l_mod, false)) {
            l_mod.dep(r_mod);
            r_mod.depBy(l_mod);
        }
    }

    private async parseExternalSymbols(mod: Module, node: Node) {
        let s_mod: Searcher = new Searcher(node, 'module_path');
        let s_symbols: Searcher = new Searcher(
            node, 'import_list', 'import_all');

        let mod_path_node: Node | null = s_mod.searching_next(node.walk());
        assert(mod_path_node != null);

        // Build Dependent module
        const module_path = relativeModPath(mod, mod_path_node);
        let dep_mod: Module | null = null;
        if (mod_group.pathExists(module_path)) {
            dep_mod = mod_group.search_by_path(module_path) as Module;
        } else {
            dep_mod = await this.parseAsModuleFromFileInternal(
                relativeModPath(mod, mod_path_node));
        }
        assert(dep_mod != null);
        this.linkModule(mod, dep_mod)

        /* Parsing external symbols */
        let import_symbols_node: Node | null =
            s_symbols.searching_next(node.walk());
        assert(import_symbols_node != null);
        if (import_symbols_node.text == '*') {
            this.importAllSymbols(mod, dep_mod);
        } else {
            let symbol_searcher: Searcher =
                new Searcher(import_symbols_node, 'ident_pattern_token');
            let symbol_nodes: Node[] =
                symbol_searcher.searching_all(import_symbols_node.walk());

            let symbols = symbol_nodes.map((n:Node) => n.text);
            mod.setExternalSymbols(dep_mod.ident, symbols);
        }
    }

    private importAllSymbols(mod: Module, dep_mod: Module) {
        /* Retrive all identifiers that defined in dep_mod */
        let s_ident: Searcher = new Searcher(
            dep_mod.tree.rootNode, 'ident');
        let ident_nodes: Node[] =
            s_ident.searching_all(dep_mod.tree.rootNode.walk());

        let idents = ident_nodes.map((n:Node) => n.text);
        mod.setExternalSymbols(dep_mod.ident, idents);
    }
}

///////////////////////////////////////////////////////////////////////////////
//                        Tree-Sitter Helper Functions                       //
///////////////////////////////////////////////////////////////////////////////
export function isLeave(node: Node) {
    return node.childCount == 0;
}

export function isInterriorNode(node: Node) {
    return !isLeave(node);
}


///////////////////////////////////////////////////////////////////////////////
//                           Iteration Of ParseTree                          //
///////////////////////////////////////////////////////////////////////////////
export function preorderIterate(
    cursor: TreeCursor,
    rootNode: Node,
    convergent_cond? : (n:Node) => boolean): Node | null {

    const convergent =
        convergent_cond != null &&
        convergent_cond(cursor.currentNode);

    if (convergent || !cursor.gotoFirstChild()) {
        if (!cursor.gotoNextSibling()) {
            if (cursor.currentNode.equals(rootNode)) {
                return null;
            }
            while (true) {
                if (!cursor.gotoParent()) {
                    return null;
                }
                if (cursor.currentNode.equals(rootNode)) {
                    return null;
                }
                if (cursor.gotoNextSibling()) {
                    break;
                }
            }
        }
    }
    return cursor.currentNode;
}

export function iterateUntil(cursor: TreeCursor,
                             cond: (n: Node) => boolean,
                             root: Node | null = null,
                             convergent_cond? : (n:Node) => boolean): Node | null {
    let rootNode = cursor.currentNode;
    if (root != null) {
        rootNode = root;
    }

    do {
        let current = cursor.currentNode;
        if (cond(current)) {
            return current;
        }
    } while(preorderIterate(cursor, rootNode, convergent_cond) != null);

    return null;
}

export function gotoNextNodeWithTypes(
    cursor: TreeCursor, node_types: WGSLNodeType[],
    rootNode: Node,
    convergent_cond? : (n:Node) => boolean): Node | null {

    preorderIterate(cursor, rootNode, convergent_cond);
    return iterateUntil(
        cursor,
        (n: Node) => node_types.find(
            (nt: WGSLNodeType) => nt == n.type) != undefined,
        rootNode,
        convergent_cond);
}

export class Searcher {
    private rootNode_: Node;
    private expected_node_type: WGSLNodeType[] = [];

    constructor(rootNode: Node, ...expected_types: WGSLNodeType[]) {
        this.rootNode_ = rootNode;
        this.expected_node_type = this.expected_node_type.concat(expected_types);
    }

    public setExpectedNodeTypes(expecteds: WGSLNodeType[]) {
        this.expected_node_type = expecteds;
    }

    public plus(s: Searcher): Searcher {
        let sum = new Searcher(this.rootNode_);
        sum.expected_node_type = sum.expected_node_type
            .concat(this.expected_node_type)
            .concat(s.expected_node_type);
        return sum;
    }
    public *searching_yield(cursor: TreeCursor): Generator<Node | null> {
        let r = this.searching_next(cursor);
        if (r == null) {
            return null;
        } else {
            yield r;
        }
    }
    public searching_next(cursor: TreeCursor,
                          convergent_cond? : (n:Node) => boolean): Node | null {
        return gotoNextNodeWithTypes(
            cursor,
            this.expected_node_type,
            this.rootNode_,
            convergent_cond);
    }
    public searching_all(cursor: TreeCursor,
                         convergent_cond? : (n:Node) => boolean): Node[] {
        let results: Node[] = [];

        while (true) {
            let match_node = gotoNextNodeWithTypes(
                cursor,
                this.expected_node_type,
                this.rootNode_,
                convergent_cond);
            if (match_node == null) {
                return results;
            } else {
                results.push(match_node);
            }
        }
    }
}
