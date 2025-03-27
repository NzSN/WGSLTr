import { strict as assert } from 'assert';
import { Node } from "web-tree-sitter";
import { Module } from "../module";
import { ObfIdent } from './obfuscate';
import { isBuiltinSymbol } from '../lang';

export class Token {
    public type: string;
    public literal: string;

    constructor(node: Node) {
        this.literal = node.text;
        this.type = node.type;
    }

    public concat(t: Token): Token {
        this.literal = this.literal + t.literal;
        return this;
    }
}

export interface TokenOperator<T> {
    ident: string;
    eval(t: Token, extra: T): Token;

    comp(op: TokenOperator<T>): TokenOperator<T>;
}


export class TokenOPEnv {
    public module: Module | null = null;
}

enum TokenOpST {
    NORMAL,
    // Skip next identifier
    SKIP,
}
export class TokenOperatorBase {
    private _state: TokenOpST = TokenOpST.NORMAL;

    protected ignorable(t: Token, extra: TokenOPEnv): boolean {
        if (t.type == "override") {
            this._state = TokenOpST.SKIP;
        }
        if (t.type == 'ident_pattern_token' &&
            this._state == TokenOpST.SKIP) {
            this._state = TokenOpST.NORMAL;
            return true;
        }

        return false;
    }
}

export class ComposableTokenOperator<T> extends TokenOperatorBase
                                        implements TokenOperator<T> {
    public ident: string = "composedTokenOperator";
    private operators: TokenOperator<T>[] = [];

    public eval(t: Token, extra: T): Token {
        let r = t;

        this.operators.forEach((operator) => {
            r = operator.eval(r, extra);
        });

        return r;
    }

    public comp(op: TokenOperator<T>): ComposableTokenOperator<T> {
        let composed_operator = new ComposableTokenOperator<T>();
        composed_operator.operators.push(op);
        return composed_operator;
    }
}

export class ModuleQualifier extends ComposableTokenOperator<TokenOPEnv>
                             implements TokenOperator<TokenOPEnv> {
    public readonly ident = "ModuleQualifier";

    private decorateWithModID(mod: Module, symbol: string): string {
        return "__" + mod.ident + "_" + symbol;
    }

    private resolve(mod: Module, ext_symbol: string): string {
        const dep_mod_id = mod.symbolFrom(ext_symbol);
        assert(dep_mod_id != null);
        const dep_mod = Module.all_by_id.get(dep_mod_id);
        assert(dep_mod != undefined);
        return this.decorateWithModID(dep_mod, ext_symbol);
    }

    public eval(t: Token, extra: TokenOPEnv): Token {
        if (this.ignorable(t, extra)) {
            return t;
        }
        if (t.type == 'ident_pattern_token') {
            assert(extra.module != null);
            if (extra.module.isExternalSymbol(t.literal)) {
                t.literal = this.resolve(extra.module, t.literal);
            } else {
                t.literal = this.decorateWithModID(extra.module, t.literal);
            }
        }
        return t;
    }
}

export class Obfuscator extends ComposableTokenOperator<TokenOPEnv>
                        implements TokenOperator<TokenOPEnv> {

    public readonly ident = "Obfuscator";
    private records: Map<string, string> = new Map();
    private obf_id: ObfIdent = new ObfIdent("a");

    private doObfuscate(t: Token, extra: TokenOPEnv): Token {
        let id = this.obf_id.value;

        while (isBuiltinSymbol(id)) {
            this.obf_id.next();
            id = this.obf_id.value;
        }

        this.obf_id.next();
        this.records.set(t.literal, id);

        t.literal = id;

        return t;
    }

    public eval(t: Token, extra: TokenOPEnv): Token {
        if (this.ignorable(t, extra)) return t;

        /* Obfucate */
        if (t.type == 'ident_pattern_token') {
            const exist_obf_id = this.records.get(t.literal);
            if (exist_obf_id != undefined) {
                t.literal = exist_obf_id;
            } else {
                return this.doObfuscate(t, extra);
            }
        }
        return t;
    }
}
