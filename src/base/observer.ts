/* Implement observer pattern */

export enum Event {
    /* Parser Event */
    PARSER_MODULE_CREATED,
    PARSER_MODULE_OUTDATED,
}

export interface Observer<T> {
    update(event: Event, x: T): void;
}

export class Subject<T> {
    private observers: Observer<T>[] = [];

    public resetObservers() {
        this.observers = [];
    }

    public attach(observer: Observer<T>): void {
        this.observers.push(observer);
    }

    public notifyAllObservers(event: Event, x: T): void {
        for (let observer of this.observers) {
            observer.update(event, x);
        }
    }
}
