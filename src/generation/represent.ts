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
        this._literal = this._literal + t._literal;
        return this;
    }
}

interface TokenOperator<T> {
    ident: string;
    eval(t: Token, extra: T): Token;
}

export class Presentation<T> {
    public readonly module: Module;
    private _tokenOps: TokenOperator<T>[] = [];

    constructor(m: Module) {
        this.module = m;
    }

    public addProcessor(t: TokenOperator<T>) {
        this._tokenOps.push(t);
    }

    public present(extra: T): Token[] {
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
                tokens = tokens.concat(new Presentation(dep_module as Module).present(extra));
            } else {
                if (isLeave(current)) {
                    let token = new Token(current.text);
                    this._tokenOps.forEach((op: TokenOperator<T>) => {
                        token = op.eval(token, extra);
                    })
                    tokens.push(token);
                }
            }
            current = preorderIterate(cursor, rootNode);
        }
        return tokens;
    }
}
