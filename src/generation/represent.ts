import { strict as assert } from 'assert';
import { Node, TreeCursor } from 'web-tree-sitter';
import { Module } from "../module";
import { Searcher, isLeave, preorderIterate } from "../parser/parser";

export class Token {
    private _literal: string;

    constructor(s: string) {
        this._literal = s;
    }

    public get literal(): string {
        return this._literal
    }

    public concat(t: Token): Token {
        this._literal = this._literal + " " + t._literal;
        return this;
    }
}

interface TokenOperator {
    ident: string;
    eval(t: Token): Token;
}

export class Presentation {
    public readonly module: Module;
    private _tokenOps: TokenOperator[] = [];

    constructor(m: Module) {
        this.module = m;
    }

    public addProcessor(t: TokenOperator) {
        this._tokenOps.push(t);
    }

    public present(): Token[] {
        let tokens: Token[] = [];

        const rootNode = this.module.rootNode;
        let current: Node | null = rootNode;
        let cursor: TreeCursor = current.walk();

        while (current != null) {
            if (current.type == 'import') {
                let s: Searcher = new Searcher(current, 'module_path');
                let module_path_node = s.searching_next(current.walk());
                assert(module_path_node != null &&
                       Module.all.has(module_path_node.text));

                let dep_module = Module.all.get(module_path_node.text);
                tokens = tokens.concat(new Presentation(dep_module as Module).present())
            } else {
                if (isLeave(current)) {
                    tokens.push(new Token(current.text));
                }
            }
            current = preorderIterate(cursor, rootNode);
        }
        return tokens;
    }
}
