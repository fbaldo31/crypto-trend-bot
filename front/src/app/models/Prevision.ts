import CryptoTrend from './Crypto';

export interface IPrevisionOpts {
    label: string;
    cryptoName: string;
    splitDate: string;
    currency: string;
    from: number;
}

export default class Prevision {
    label: string;
    crypto: CryptoTrend;
    splitDate: Date;

    constructor(options: IPrevisionOpts) {
        this.label = options.label;
        this.splitDate = new Date(options.splitDate);
        this.crypto = new CryptoTrend(options.cryptoName, options.currency, options.from);
    }
}
