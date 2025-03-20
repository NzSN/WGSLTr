import fc from 'fast-check';
import { Tree } from 'web-tree-sitter';
import { Searcher, WGSLParser } from './parser';
import { Module } from '../module';

describe("Parser Unittests", () => {

    test("Parsing", async () => {
        let parser = new WGSLParser();
        const source = (n: number) => {
            return `fn main() { var a: vec2<f32> = 1 < ${n}; }` };

        await fc.assert(fc.asyncProperty(fc.nat(), async (n: number) => {
            let tree: Tree | null = await parser.parse(source(n));

            let s: Searcher = new Searcher((tree as Tree).rootNode, "less_than");
            let bb = s.searching_all((tree as Tree).rootNode.walk());
            console.log(bb[0].text);

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
        let M = await parser.parseAsModuleFromFile(path);

        expect(Module.all.size == 2).toBeTruthy();

        const module_A = Module.all.get(path);
        expect(module_A != undefined);
        const module_B = Module.all.get(path_of_B);
        expect(module_B != undefined);
        expect(module_A?.isDepOn(module_B as Module)).toBeTruthy();
        expect(module_B?.isDepBy(module_A as Module)).toBeTruthy();
    })
})
