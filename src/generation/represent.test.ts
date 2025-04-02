import fc from 'fast-check';
import { WGSLParser } from '../parser/parser';

import { CircularExcept, Presentation } from './represent';
import { Module } from '../module';
import { mod_group } from '../module_group';

describe("Representation Unittests", () => {
    let parser: WGSLParser = new WGSLParser();
    parser.attach(mod_group);

    afterEach(() => mod_group.reset());

    test("Recursively Present", async () => {
        let mod: Module | null =
            await parser.parseAsModule(
                "./Test/wgsl_samples/A.wgsl");
        expect(mod != null).toBeTruthy();
        let p: Presentation = new Presentation(mod as Module);
        let present = p.present().reduce(
            (acc,cur) => acc + " " + cur.literal, "");
    })

    test("Circular Present", async () => {
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

    test("Represent Cache", async () => {

    })
})
