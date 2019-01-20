import { Injectable } from '@angular/core';
import { HttpClient, HttpResponse } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class CryptoService {

  readonly SERVER_URL = 'http://0.0.0.0:8000';
  public CRYPTO = 'BTC';
  public from = 365;
  public currency = 'EUR';
  readonly API_KEY = '<YOUR_API_KEY>';

  constructor(private http: HttpClient) { }

  public getCryptoDataFrom(crypto = 'BTC', from = 365) {
    this.CRYPTO = crypto;
    this.from = from;
    const url =
    `https://min-api.cryptocompare.com/data/histoday?fsym=${this.CRYPTO}&tsym=${this.currency}&limit=${this.from}&api_key=${this.API_KEY}`;
    return this.http.get(url);
  }

  public async sendServerRequest(path: string, params: string): Promise<any> {
    return await new Promise((resolve, reject) => {
      try {
        resolve(this.http.get(this.SERVER_URL + '/' + path + '?' + params).toPromise());
      } catch (error) {
        reject(error);
      }
    });
  }
}
