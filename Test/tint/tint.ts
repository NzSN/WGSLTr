import * as process from 'process';
import { writeFileSync, existsSync } from 'fs';
import { execFileSync } from 'child_process';
import * as tmp from 'tmp';

export class Tint {
    public static verify(source: string): boolean {
        // Tmp file to to contain wgsl sources
        const tmpfile = tmp.fileSync({postfix:'.wgsl'});
        writeFileSync(tmpfile.name, source);

        let pathToBin: string | null = null;
        switch (process.platform) {
            case 'linux': {
                pathToBin = './Test/tint/platforms/linux/tint';
                break;
            }
            case 'win32': {}
            case 'darwin': {
                pathToBin = './Test/tint/platforms/mac/tint';
                break;
            }
        }

        if (!pathToBin || !existsSync(pathToBin)) {
            throw new Error("Failed to find Tint binary (" + process.platform + "): " + pathToBin);
        }

        let success = true;
        try {
            const child = execFileSync(pathToBin, [tmpfile.name]);
        } catch(e) {
            console.log(e);
            success = false;
        }

        tmp.setGracefulCleanup();

        return success;
    }
}
