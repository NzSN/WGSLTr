import { Module } from "../module";

function brokenOverrides(mod: Module): boolean | never {



}

export function semanticVerify(mod: Module): boolean | never {
    brokenOverrides(mod);
}
