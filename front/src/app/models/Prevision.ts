import CryptoTrend from './Crypto';
import { IGraph } from './IGraph';

export interface IPrevisionOpts {
    label: string;
    cryptoName: string;
    trainStartDate: string;
    trainEndDate: string;
    testStartDate: string;
    testEndDate: string;
    currency: string;
    from: number;
}

export default class Prevision {
    label: string;
    crypto: CryptoTrend;
    trainStartDate: string;
    trainEndDate: string;
    testStartDate: string;
    testEndDate: string;
    graph: IGraph;

    constructor(options: IPrevisionOpts) {
        this.label = options.label;
        this.trainStartDate = options.trainStartDate;
        this.trainEndDate = options.trainEndDate;
        this.testStartDate = options.testStartDate;
        this.testEndDate = options.testEndDate;
        this.crypto = new CryptoTrend(options.cryptoName, options.currency);
    }
}
