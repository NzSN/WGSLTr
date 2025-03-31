/* Implement observer pattern */
export interface Observer<T> {
    update(x: T): void;
}

export class Subject<T> {
    private observers: Observer<T>[] = [];

    public resetObservers() {
        this.observers = [];
    }

    public attach(observer: Observer<T>): void {
        this.observers.push(observer);
    }

    public notifyAllObservers(x: T): void {
        for (let observer of this.observers) {
            observer.update(x);
        }
    }
}
