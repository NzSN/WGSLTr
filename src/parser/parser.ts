import { strict as assert } from 'assert';
import { WASM_MODULE_PATHS } from '../constants';
import { Parser, Language, Tree, TreeCursor, Node } from 'web-tree-sitter';
import { Module } from '../module';

type WGSLNode = Node;

export class WGSLParser {
    private parser_: Parser | null = null;
    private typescript_lang: Language | null = null;

    async parse(source: string): Promise<Tree | null> {
        if (this.parser_ == null) {
            await Parser.init();
            this.parser_ = new Parser();

            this.typescript_lang =
                await Language.load(WASM_MODULE_PATHS.TreeSitterWGSL);
            this.parser_.setLanguage(this.typescript_lang);
        }
        return this.parser_.parse(source);
    }

    async parseAsModule(path: string, source: string): Promise<Module | null> {
        let tree = await this.parse(source);
        assert(tree != null);

        return new Module(path, tree);
    }
}

///////////////////////////////////////////////////////////////////////////////
//                           Iteration Of ParseTree                          //
///////////////////////////////////////////////////////////////////////////////
export function preorderIterate(cursor: TreeCursor, rootNode: Node): Node | null{
    if (!cursor.gotoFirstChild()) {
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
                             root: Node | null = null): Node | null {
    let rootNode = cursor.currentNode;
    if (root != null) {
        rootNode = root;
    }

    do {
        let current = cursor.currentNode;
        if (cond(current)) {
            return current;
        }
    } while(preorderIterate(cursor, rootNode) != null);

    return null;
}

export function gotoNextNodeWithTypes(
    cursor: TreeCursor, node_types: WGSLNode[],
    rootNode: Node | null = null): Node | null {

    preorderIterate(cursor, cursor.currentNode);
    return iterateUntil(cursor,
                        (n: Node) =>
        node_types.find((nt: WGSLNode) => nt.type == n.type) != undefined,
                       rootNode);
}

export class Searcher {
    private rootNode_: Node;
    private expected_node_type: WGSLNode[] = [];

    constructor(rootNode: Node, ...expected_types: WGSLNode[]) {
        this.rootNode_ = rootNode;
        this.expected_node_type = this.expected_node_type.concat(expected_types);
    }

    public setExpectedNodeTypes(expecteds: WGSLNode[]) {
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
    public searching_next(cursor: TreeCursor): Node | null {
        return gotoNextNodeWithTypes(
            cursor, this.expected_node_type, this.rootNode_);
    }
    public searching_all(cursor: TreeCursor): Node[] {
        let results: Node[] = [];

        while (true) {
            let match_node = gotoNextNodeWithTypes(
                cursor, this.expected_node_type, this.rootNode_);
            if (match_node == null) {
                return results;
            } else {
                results.push(match_node);
            }
        }
    }
}
