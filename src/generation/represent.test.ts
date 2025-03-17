import fc from 'fast-check';
import { WGSLParser } from '../parser/parser';
import { Presentation } from './represent';
import { Module } from '../module';

describe("Representation Unittests", () => {

    test("Introduction", async () => {
        let source = (n:number) => {
            return "fn main() {${n}};" };

        let parser: WGSLParser = new WGSLParser();

        await fc.assert(fc.asyncProperty(fc.nat(), async (n:number) => {
            let mod: Module | null = await parser.parseAsModule("M", source(n));

            if (mod == null) return false;
            let p: Presentation = new Presentation(mod);
            return p.present().reduce(
                (acc,cur) => { return acc + cur.literal; },
                "") == source(n).replace('/\s/g', '');
        }));
    })

})
