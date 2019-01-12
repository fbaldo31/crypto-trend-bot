export interface IRawData {
    time: number;
    close: number;
    high: number;
    low: number;
    open: number;
    volumefrom: number;
    volumeto: number;
}

export interface ICryptoTrendData {
    Timestamp: number;
    Open: number;
    High: number;
    Low: number;
    Close: number;
    'Volume_(BTC)': number;
    'Volume_(Currency)': number;
    'Weighted_Price': number;
}

export default class CryptoTrend {
    name: string;
    currency: string;
    from?: number;
    data?: ICryptoTrendData[];

    constructor(name: string, currency) {
        this.name = name;
        this.currency = currency;
    }
}
