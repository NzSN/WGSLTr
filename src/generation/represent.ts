import { strict as assert } from 'assert';
import { Node, TreeCursor } from 'web-tree-sitter';
import { Module } from "../module";
import { Searcher, isLeave, preorderIterate } from "../parser/parser";
import { importModPathStr } from '../parser/utility';
import { Token, TokenOPEnv, TokenOperator, ModuleQualifier, Obfuscator } from './token_processors';
import { Semantic } from '../analyzer/semantic';

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

export class CircularExcept extends Error {
    constructor() {
        super("Circular dependences is not allowed");
    }
}

export class Presentation {
    public readonly module: Module;
    private _tokenOps: TokenOperator<TokenOPEnv>[] = [];
    private _cwd: string = "";
    private _import_filter: ImportStmtFilter = new ImportStmtFilter();
    private _op_env: TokenOPEnv;

    constructor(m: Module) {
        this.module = m;

        if (this.module.circular_point.length > 0) {
            throw new CircularExcept();
        }

        this._cwd = this.module.path;
        this._op_env = new TokenOPEnv();
        this._op_env.module = this.module;
    }

    public addProcessor(t: TokenOperator<TokenOPEnv>) {
        this._tokenOps.push(t);
    }

    private tokenProc(tokens: Token[], token: Token) {
        this._tokenOps.forEach((op: TokenOperator<TokenOPEnv>) => {
            token = op.eval(token, this._op_env);
        });
        tokens.push(token);
    }

    public present(): Token[] {
        let tokens: Token[] = [];

        const rootNode = this.module.rootNode;
        let current: Node | null = rootNode;
        let cursor: TreeCursor = current.walk();

        while (current != null) {
            Semantic.verify(this.module, current);

            if (current.type == 'import' &&
                current.isNamed) {

                let s: Searcher = new Searcher(current, 'module_path');
                let module_path_node = s.searching_next(current.walk());
                assert(module_path_node != null);

                let module_path = importModPathStr(this._cwd, module_path_node.text);
                assert(Module.all.has(module_path));

                let dep_module = Module.all.get(module_path);
                tokens = tokens.concat(new Presentation(dep_module as Module).present());
            } else {
                if (isLeave(current)) {
                    let token: Token = new Token(current);
                    /* Filter out import statements */
                    if (!this._import_filter.filter(token)) {
                        current = preorderIterate(cursor, rootNode);
                        continue;
                    } else {
                        if (this._import_filter.state == FilterState.ROLLBACK) {
                            this._import_filter.pendingTokens.forEach((t: Token) => {
                                this.tokenProc(tokens, t);
                            });
                        }
                    }
                    this.tokenProc(tokens, token);
                }
            }
            current = preorderIterate(cursor, rootNode);
        }
        return tokens;
    }
}
