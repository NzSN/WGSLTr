import fc from 'fast-check';
import { Tree } from 'web-tree-sitter';
import { WGSLParser } from './parser';

describe("Parser Unittests", () => {

    test("Parsing", async () => {
        let parser = new WGSLParser();
        const source = (n: number) => {
            return `fn main() { var a: f32 = ${n}; }` };

        await fc.assert(fc.asyncProperty(fc.nat(), async (n: number) => {
            let tree: Tree | null = await parser.parse(source(n));
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

    })
})
