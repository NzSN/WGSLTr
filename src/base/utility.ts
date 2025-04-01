export function delay(ms: number) {
    return new Promise( resolve => setTimeout(resolve, ms) );
}

export async function waitUntil(cond: () => boolean, timeout: number = 0) {
    let counter: number = 0;
    const stride_len = 10;
    while (true) {
        await delay(stride_len);
        if (cond() ||
            (timeout > 0 && counter > timeout)) {
            break;
        }
        counter += stride_len;
    }
}
