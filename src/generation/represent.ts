import { strict as assert } from 'assert';
import { Node, TreeCursor } from 'web-tree-sitter';
import { Module } from "../module";
import { Searcher, isLeave, preorderIterate } from "../parser/parser";
import { importModPathStr } from '../parser/utility';

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

export interface TokenOperator<T> {
    ident: string;
    eval(t: Token, extra: T): Token;
}


enum FilterState {
    Ready,
    MEET,
    CONFIRMED,
    ROLLBACK,
}
interface TokenFilter {
    filter(t: Token): boolean;
}

class ImportStmtFilter implements TokenFilter {
    private _state: FilterState = FilterState.Ready;
    private _pending_tokens: Token[] = [];

    public get pendingTokens(): Token[] {
        return this._pending_tokens;
    }

    public get state() {
        return this._state;
    }

    public reset() {
        this._state = FilterState.Ready;
        this._pending_tokens = [];
    }

    public filter(t: Token): boolean {
        if (this._state == FilterState.CONFIRMED) {
            if (t.literal == ';') {
                this._state = FilterState.Ready;
                return false;
            } else {
                return false;
            }
        }

        switch (t.literal) {
            case 'import':
                if (this._state == FilterState.Ready) {
                    this._state = FilterState.MEET;

                    this._pending_tokens.push(t);
                    return false;
                } else {
                    assert("Filter not ready");
                }
                break;
            case '{':
            case '*':
                if (this._state == FilterState.MEET) {
                    this._state = FilterState.CONFIRMED;
                    this._pending_tokens = [];

                    return false;
                } else {
                    return true;
                }
            default:
                if (this._state == FilterState.MEET) {
                    this._state = FilterState.ROLLBACK;
                    return true;
                }
        }
        return true;
    }
}

export class Presentation<T> {
    public readonly module: Module;
    private _tokenOps: TokenOperator<T>[] = [];
    private _cwd: string = "";
    private _import_filter: ImportStmtFilter = new ImportStmtFilter();

    constructor(m: Module) {
        this.module = m;
        this._cwd = this.module.path;
    }

    public addProcessor(t: TokenOperator<T>) {
        this._tokenOps.push(t);
    }

    private tokenProc(tokens: Token[], token: Token, extra: T) {
        this._tokenOps.forEach((op: TokenOperator<T>) => {
            token = op.eval(token, extra);
        });
        tokens.push(token);
    }

    public present(extra: T): Token[] {
        let tokens: Token[] = [];

        const rootNode = this.module.rootNode;
        let current: Node | null = rootNode;
        let cursor: TreeCursor = current.walk();

        while (current != null) {
            if (current.type == 'import' &&
                current.isNamed) {

                let s: Searcher = new Searcher(current, 'module_path');
                let module_path_node = s.searching_next(current.walk());
                assert(module_path_node != null)

                let module_path = importModPathStr(this._cwd, module_path_node.text);
                assert(Module.all.has(module_path));

                let dep_module = Module.all.get(module_path);
                tokens = tokens.concat(new Presentation(dep_module as Module).present(extra));
            } else {
                /* Filter out import statements */
                if (isLeave(current)) {
                    let token: Token = new Token(current.text);

                    if (!this._import_filter.filter(token)) {
                        current = preorderIterate(cursor, rootNode);
                        continue;
                    } else {
                        if (this._import_filter.state == FilterState.ROLLBACK) {
                            this._import_filter.pendingTokens.forEach((t: Token) => {
                                this.tokenProc(tokens, t, extra);
                            });
                        }
                    }
                    this.tokenProc(tokens, token, extra);
                }
            }
            current = preorderIterate(cursor, rootNode);
        }
        return tokens;
    }
}
