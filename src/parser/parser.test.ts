import fc from 'fast-check';
import { Tree, Node } from 'web-tree-sitter';
import { Searcher, WGSLParser } from './parser';
import { Module } from '../module';
import { mod_group } from '../module_group';

import Path from 'path';
import fs from 'fs';

describe("Parser Unittests", () => {

    let parser: WGSLParser = new WGSLParser();
    parser.attach(mod_group);

    afterEach(() => {
        mod_group.reset();
    })

    afterAll(() => {
        mod_group.reset();
    })

    test("Basic Parsing", async () => {
        const source = (n: number) => {
            return `fn main() { var a: vec2<f32> = 1 < ${n}; }` };

        await fc.assert(fc.asyncProperty(fc.nat(), async (n: number) => {
            let tree: Tree | null = await parser.parse(source(n));

            let s: Searcher = new Searcher((tree as Tree).rootNode, "less_than");
            let bb = s.searching_all((tree as Tree).rootNode.walk());

            return tree != null && tree?.rootNode.text == source(n);
        }));

        await fc.assert(
            fc.asyncProperty(
                fc.nat(), fc.nat(),
                async (ln: number, rn: number) => {
                    fc.pre(ln != rn);
                    let tree: Tree | null = await parser.parse(source(ln));
                    return tree != null && tree.rootNode.text != source(rn);
                }));
    })

    test("Recursively Parsing", async () => {
        const path = "./Test/wgsl_samples/A.wgsl";
        const path_of_B = "./Test/wgsl_samples/B.wgsl";
        let M = await parser.parseAsModule(path);

        expect(mod_group.size == 2).toBeTruthy();

        const module_A = mod_group.search_by_path(Path.resolve(path));
        expect(module_A != undefined).toBeTruthy();
        const module_B = mod_group.search_by_path(Path.resolve(path_of_B));
        expect(module_B != undefined).toBeTruthy();
        expect(module_A?.isDepOn(module_B as Module)).toBeTruthy();
        expect(module_B?.isDepBy(module_A as Module)).toBeTruthy();
    })

    test("Circular import", async () => {
        const path_A = "./Test/wgsl_samples/circular/A.wgsl";
        const path_B = "./Test/wgsl_samples/circular/B.wgsl";

        const mod_A = await parser.parseAsModule(path_A);
        const mod_B = mod_group.search_by_path(Path.resolve(path_B));

        expect(mod_A != null).toBeTruthy();
        expect(mod_B != null).toBeTruthy();

        expect(mod_A?.circular_point.length == 1).toBeTruthy();
        expect(mod_A?.circular_point[0].path == Path.resolve(path_B)).toBeTruthy();
    })

    test("Dual Circular import", async () => {
        const path_A = "./Test/wgsl_samples/circular/Case1/A.wgsl";
        const path_B = "./Test/wgsl_samples/circular/Case1/B.wgsl";
        const path_C = "./Test/wgsl_samples/circular/Case1/C.wgsl";

        const mod_A = await parser.parseAsModule(path_A);
        const mod_B = mod_group.search_by_path(Path.resolve(path_B));
        const mod_C = mod_group.search_by_path(Path.resolve(path_C));

        expect(mod_A != null).toBeTruthy();
        expect(mod_B != null).toBeTruthy();
        expect(mod_C != null).toBeTruthy();

        expect(mod_A?.circular_point.length == 2).toBeTruthy();
        expect(mod_A?.circular_point[0].path == Path.resolve(path_B)).toBeTruthy();
        expect(mod_A?.circular_point[1].path == Path.resolve(path_C)).toBeTruthy();
    })

    test("Global Unique Module", async () => {
        const path_B = "./Test/wgsl_samples/global_unique_module/Dir0/B.wgsl";
        const path_C = "./Test/wgsl_samples/global_unique_module/C.wgsl";

        await parser.parseAsModule(path_B);
        await parser.parseAsModule(path_C);

        expect(mod_group.size == 3).toBeTruthy();
    })

    test("Single Module Outdated", async () => {
        const path = "./Test/wgsl_samples/module_outdate/D.wgsl";

        let fd = fs.openSync(path, "w+");
        fs.writeSync(fd, "const pipi = 0;");
        fs.close(fd);

        const mod = await parser.parseAsModule(path);

        fd = fs.openSync(path, "w+");
        fs.writeSync(fd, "const pipi = 1;");
        fs.close(fd);

        const mod_new = await parser.parseAsModule(path);

        expect(!mod?.equal(mod_new as Module)).toBeTruthy();
        expect(mod?.rootNode.text != mod_new?.rootNode.text).toBeTruthy();
        expect(mod_group.size == 1).toBeTruthy();
        expect(mod_group.search_by_id((mod as Module).ident) != null).toBeTruthy();
        expect(!(mod_group.search_by_id((mod as Module).ident) as Module)
                    .equal(mod as Module)).toBeTruthy();
        expect((mod_group.search_by_id((mod as Module).ident) as Module)
                   .equal(mod_new as Module)).toBeTruthy();
    })

    test("Module Outdated in graph", async () => {

    })
})
