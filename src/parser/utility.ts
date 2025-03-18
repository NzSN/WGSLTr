import { strict as assert } from 'assert';
import * as PathUtil from 'path';
import { Node } from 'web-tree-sitter';
import { Module } from '../module';

type Path = string;

export function isWGSLModuleByExtension(path: Path): boolean {
    const extName = PathUtil.extname(path);
    return extName == '' || extName == '.wgsl'
}
export function wgslPathComplete(path: Path): Path {
    if (isWGSLModuleByExtension(path)) {
        const extName = PathUtil.extname(path);
        if (extName == '') {
            return path + ".wgsl";
        } else {
            return path;
        }
    } else {
        return path;
    }
}

export function relativeModPath(current_mod: Module, module_path_node: Node) {
    assert(module_path_node.type == "module_path");
    let import_path = module_path_node.text
        .replace(/\'/g, "")
        .replace(/\"/, "")
        .replace(/;/, "");
    let path = PathUtil.join(
        PathUtil.dirname(current_mod.path),
        import_path);
    return "./" + wgslPathComplete(path)
}
