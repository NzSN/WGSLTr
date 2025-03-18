import { Tree, Node } from "web-tree-sitter";

type ModPath = string;
export class Module {
    public static all: Map<ModPath, Module> = new Map();

    private _path: string;
    private _tree: Tree;
    private _deps: Module[]   = [];
    private _depBys: Module[] = [];

    constructor(path: ModPath, tree: Tree) {
        this._path = path;
        this._tree = tree;

        Module.all.set(path, this);
    }

    public get rootNode(): Node {
        return this._tree.rootNode;
    }

    public isDepOn(may_dep: Module) {
        return this._deps.find((m: Module) => {
            return may_dep.path == m.path;
        }) != undefined;
    }

    public isDepBy(may_dep_by: Module) {
        return may_dep_by.isDepOn(this);
    }

    public dep(m: Module) {
        this._deps.push(m);
    }

    public depBy(m: Module) {
        this._depBys.push(m);
    }

    public get path() {
        return this._path;
    }
}
