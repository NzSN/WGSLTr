import { Module } from "../module";

export class Token {
    private _literal: string;

    constructor(s: string) {
        this._literal = s;
    }

    public get literal(): string {
        return this._literal
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
        return [];
    }
}
