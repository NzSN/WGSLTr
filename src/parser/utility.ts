import { strict as assert } from 'assert';
import * as PathUtil from 'path';
import { Node } from 'web-tree-sitter';
import { Module } from '../module';
import { platform } from 'process';

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

export function importModPathStr(current_mod: string, import_mod_path: string) {
    let import_path = import_mod_path
        .replace(/\'/g, "")
        .replace(/\"/, "")
        .replace(/;/, "");
    let path = PathUtil.join(
        PathUtil.dirname(current_mod),
        import_path);

    if (platform == 'win32') {
        path = path.replace(/\\/g, "/");
    }

    return "./" + wgslPathComplete(path)

}

export function relativeModPath(
    current_mod: Module, module_path_node: Node) {

    assert(module_path_node.type == "module_path");
    return importModPathStr(current_mod.path, module_path_node.text);
}
