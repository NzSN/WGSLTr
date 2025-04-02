import { strict as assert } from 'assert';
import { Observer, Event } from "./base/observer";
import { ModPath, ModID, Module } from "./module";

export class ModuleGroup implements Observer<Module> {
    private modules: Map<ModPath, Module> = new Map();
    private modules_by_id: Map<ModID, Module> = new Map();

    public addModule(mod: Module) {
        this.modules.set(mod.path, mod);
        this.modules_by_id.set(mod.ident, mod);
    }

    public search_by_path(path: ModPath): Module | null {
        return this.modules.get(path) ?? null;
    }

    public search_by_id(ident: ModID): Module | null {
        return this.modules_by_id.get(ident) ?? null;
    }

    public identExists(ident: ModID): boolean {
        return this.search_by_id(ident) != null;
    }

    public pathExists(path: ModPath): boolean {
        return this.search_by_path(path) != null;
    }

    public get size() {
        assert(this.modules.size == this.modules_by_id.size);
        return this.modules.size;
    }

    public reset() {
        this.modules = new Map();
        this.modules_by_id = new Map();
    }
    public async destruct_by_id(ident: ModID) {
        let mod = this.modules_by_id.get(ident);
        if (mod == undefined) {
            return;
        }
        // Destructing the module
        if (!await mod.isOutOfDate()) {
            mod.destruct();
        }

        this.modules_by_id.delete(ident);
        this.modules.delete(mod.path);
    }

    public async destruct_by_path(path: ModPath) {
        let mod = this.modules.get(path);
        if (mod == undefined) {
            return;
        }
        // Destructing the module
        if (!await mod.isOutOfDate()) {
            mod.destruct();
        }
        this.modules_by_id.delete(mod.ident);
        this.modules.delete(path);
    }

    public update(event: Event, m: Module): void {
        switch (event) {
            case Event.PARSER_MODULE_CREATED:
                this.addModule(m);
                break;
            case Event.PARSER_MODULE_OUTDATED:
                this.destruct_by_id(m.ident);
                break;
        }
    }
}

export let mod_group: ModuleGroup = new ModuleGroup();
