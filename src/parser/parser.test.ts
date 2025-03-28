import fc from 'fast-check';
import { strict as assert } from 'assert';
import { Tree, Node } from 'web-tree-sitter';
import { Searcher, WGSLParser } from './parser';
import { Module } from '../module';
import { Analyzer } from '../analyzer/analyzer';

describe("Parser Unittests", () => {

    test("Basic Parsing", async () => {
        let parser = new WGSLParser();
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
        let parser = new WGSLParser();
        const path = "./Test/wgsl_samples/A.wgsl";
        const path_of_B = "./Test/wgsl_samples/B.wgsl";
        let M = await parser.parseAsModule(path);

        expect(Module.all.size == 2).toBeTruthy();

        const module_A = Module.all.get(path);
        expect(module_A != undefined);
        const module_B = Module.all.get(path_of_B);
        expect(module_B != undefined);
        expect(module_A?.isDepOn(module_B as Module)).toBeTruthy();
        expect(module_B?.isDepBy(module_A as Module)).toBeTruthy();
    })

    test("Circular import", async () => {
        let parser = new WGSLParser();
        const path_A = "./Test/wgsl_samples/circular/A.wgsl";
        const path_B = "./Test/wgsl_samples/circular/B.wgsl";

        const mod_A = await parser.parseAsModule(path_A);
        const mod_B = Module.all.get(path_B);

        expect(mod_A != null).toBeTruthy();
        expect(mod_B != null).toBeTruthy();

        expect(mod_A?.circular_point.length == 1).toBeTruthy();
        expect(mod_A?.circular_point[0].path == path_B).toBeTruthy();
    })

    test("Dual Circular import", async () => {
        let parser = new WGSLParser();
        const path_A = "./Test/wgsl_samples/circular/Case1/A.wgsl";
        const path_B = "./Test/wgsl_samples/circular/Case1/B.wgsl";
        const path_C = "./Test/wgsl_samples/circular/Case1/C.wgsl";

        const mod_A = await parser.parseAsModule(path_A);
        const mod_B = Module.all.get(path_B);
        const mod_C = Module.all.get(path_C);

        expect(mod_A != null).toBeTruthy();
        expect(mod_B != null).toBeTruthy();
        expect(mod_C != null).toBeTruthy();

        expect(mod_A?.circular_point.length == 2).toBeTruthy();
        expect(mod_A?.circular_point[0].path == path_B).toBeTruthy();
        expect(mod_A?.circular_point[1].path == path_C).toBeTruthy();
    })
})
