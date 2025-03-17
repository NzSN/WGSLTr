import { Tree, Node } from "web-tree-sitter";

type ModPath = string;
export class Module {
    public static all: Map<ModPath, Module>;

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

    public dep(m: Module) {
        this._deps.push(m);
    }

    public depBy(m: Module) {
        this._depBys.push(m);
    }
}
