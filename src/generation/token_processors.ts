import { Token, TokenOperator } from './represent';

class ModuleQualifier implements TokenOperator<void> {
    public readonly ident = "ModuleQualifier";

    public eval(t: Token): Token {
        return t;
    }
}
