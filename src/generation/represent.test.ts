import fc from 'fast-check';
import { WGSLParser } from '../parser/parser';
import { CircularExcept, Presentation } from './represent';
import { Module } from '../module';
import { ModuleQualifier, Obfuscator } from './token_processors';

describe("Representation Unittests", () => {
    test("Basic Present", async () => {
        let source = (n:number) => {
            return `fn abs() { abs(${n}); };` };
        let parser: WGSLParser = new WGSLParser();

        await fc.assert(fc.asyncProperty(fc.nat(), async (n:number) => {
            let mod: Module | null = await parser.parseAsModule("M", source(n));

            if (mod == null) return false;
            let p: Presentation = new Presentation(mod);

            let present = p.present(new ModuleQualifier()).reduce(
                (acc,cur) => { return acc + cur.literal; },
                "");

            return present == source(n).replace(/\s/g, '');
        }));
    })

    test("Recursively Present", async () => {
        let parser: WGSLParser = new WGSLParser();

        let mod: Module | null =
            await parser.parseAsModule(
                "./Test/wgsl_samples/A.wgsl");
        expect(mod != null).toBeTruthy();
        let p: Presentation = new Presentation(mod as Module);
        let present = p.present().reduce(
            (acc,cur) => acc + " " + cur.literal, "");
    })

    test("Circular Present", async () => {
        let parser: WGSLParser = new WGSLParser();

        let mod: Module | null =
            await parser.parseAsModule(
                "./Test/wgsl_samples/circular/A.wgsl");
        expect(mod != null).toBeTruthy();
        try {
            let p: Presentation = new Presentation(mod as Module);
        } catch (e) {
            if (e instanceof CircularExcept) {
                return;
            }
        }
        fail();
    })
})
