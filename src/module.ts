import { Tree } from "web-tree-sitter";

type ModID = string;
export class Module {
    private _ident: string;
    private _tree: Tree;
    private _deps: Module[]   = [];
    private _depBys: Module[] = [];

    constructor(ident: ModID, tree: Tree) {
        this._ident = ident;
        this._tree = tree;
    }

    public dep(m: Module) {
        this._deps.push(m);
    }

    public depBy(m: Module) {
        this._depBys.push(m);
    }
}
