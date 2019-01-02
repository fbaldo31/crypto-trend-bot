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
    from: number;
    data?: ICryptoTrendData[];

    constructor(name: string, currency, from: number, data?: IRawData[]) {
        this.name = name;
        this.currency = currency;
        this.from = from;
        if (data) {
            this.formatData(data);
        }
    }

    formatData(rawData: any[]) {
        rawData.forEach((item: IRawData) => {
            this.data.push({
                Timestamp: item.time,
                Open: item.open,
                High: item.high,
                Low: item.low,
                Close: item.close,
                'Volume_(BTC)': item.volumefrom,
                'Volume_(Currency)': item.volumeto,
                /** @todo What is that ? */
                'Weighted_Price': item.volumefrom
            });
        });
    }
}
