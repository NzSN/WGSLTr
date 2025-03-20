class ObfBit {
    static minimum: string = "a";
    static maximum: string = "z";

    static valid(s: string): boolean {
        for (let i = 0; i < s.length; ++i) {
            if (s[i] < this.minimum || s[i] > this.maximum) {
                return false;
            }
        }
        return true;
    }

    static next(value: string): string {
        if (value == this.maximum) {
            value = this.minimum;
        } else {
            value = String.fromCharCode(
                value.charCodeAt(0) + 1);
        }
        return value;
    }
};

// A type that satisified
// ObfNamed x {1} -> ObfNamed
export class ObfIdent {
    private _v: string;
    private _prefix: string;
    private _postfix: string;

    get value() {
        return this._prefix + this._v;
    }

    constructor(s: string, prefix?: string, postfix?: string) {
        if (ObfBit.valid(s)) {
            this._v = s;
            this._prefix = prefix ?? "";
            this._postfix = postfix ?? "";
        } else {
            throw TypeError(
                "String to construct ObfIdent contains invalid char");
        }
    }

    next(): ObfIdent  {
        let lsbIdx = this._v.length - 1;
        let lsb = this._v[lsbIdx];

        if (lsb = ObfBit.maximum) {
            let i = lsbIdx;
            for (let i = lsbIdx; i >= 0; --i) {
                if (this._v[i] == ObfBit.maximum) {
                    this._v = this._v.substring(0, i) +
                        ObfBit.minimum + this._v.substring(i+1);

                    if (i == 0) {
                        this._v = ObfBit.minimum + this._v;
                    }
                } else {
                    this._v = this._v.substring(0, i) +
                        ObfBit.next(this._v[i]) +
                        this._v.substring(i+1);
                    break;
                }
            }
        }

        return this;
    }
};
