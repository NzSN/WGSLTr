import fc from 'fast-check';
import { WGSLParser } from '../parser/parser';
import { Presentation } from './represent';
import { Module } from '../module';

describe("Representation Unittests", () => {

    test("Present", async () => {
        let source = (n:number) => {
            return `fn main() {${n}};` };

        let parser: WGSLParser = new WGSLParser();

        await fc.assert(fc.asyncProperty(fc.nat(), async (n:number) => {
            let mod: Module | null = await parser.parseAsModule("M", source(n));

            if (mod == null) return false;
            let p: Presentation = new Presentation(mod);

            let present = p.present().reduce(
                (acc,cur) => { return acc + cur.literal; },
                "");

            return present == source(n).replace(/\s/g, '');
        }));
    })

    test("Recursive Present", async () => {
        let parser: WGSLParser = new WGSLParser();

        let mod: Module | null =
            await parser.parseAsModuleFromFile(
                "./Test/wgsl_samples/abc.wgsl");
        expect(mod != null).toBeTruthy();

        let p: Presentation = new Presentation(mod as Module);
        let present = p.present().reduce(
            (acc,cur) => acc + " " + cur.literal, "");
        console.log(present);
    })
})
