import { Component, OnInit } from '@angular/core';
import { HttpResponse } from '@angular/common/http';
import { CryptoService } from '../services/crypto.service';
import Prevision, { IPrevisionOpts } from '../models/Prevision';
import { IGraph } from '../models/IGraph';

@Component({
  selector: 'app-bot',
  templateUrl: './bot.component.html',
  styleUrls: ['./bot.component.scss']
})
export class BotComponent implements OnInit {

  public crypto: string;
  public from: number;
  public currency: string;
  public splitDate: Date;
  public addFormOpen = false;
  public previsions: Prevision[];
  public previsionSelected: Prevision;
  public graphs: IGraph[] = [];
  public errMsg = '';
  public successMsg = '';
  public isLoading = false;

  constructor(private btc: CryptoService) {
    this.previsions = [];
  }

  ngOnInit() {
    let previsions;
    try {
      previsions = JSON.parse(localStorage.getItem('previsions'));
    } catch (error) {
      console.log(error);
    }
    if (previsions && previsions.length) {
      this.previsions = previsions;
    }
  }

  onPrevisionAdded(event: any) {
    let prevision: Prevision;
    try {
      prevision = new Prevision(<IPrevisionOpts>event);
    } catch (error) {
      console.error(error);
    }
    if (prevision) {
      const double = this.previsions.find((item: Prevision) => item.label === prevision.label);
      if (double) {
        this.errMsg = 'A prevision with the same label already exists';
        return;
      }
      this.previsions.push(prevision);
      this.savePrevisions();
      this.successMsg = 'Your prevision has been created.';
    }
  }

  removePrevision(index: number) {
    this.previsions.splice(index, 1);
    this.savePrevisions();
  }

  savePrevisions() {
    localStorage.setItem('previsions', JSON.stringify(this.previsions));
  }

  startPrevision(index: number) {
    this.isLoading = true;
    const tStart = this.previsions[index].trainStartDate;
    const tEnd = this.previsions[index].trainEndDate;
    const tsStart = this.previsions[index].testStartDate;
    const tsEnd = this.previsions[index].testEndDate;

    this.btc.sendServerRequest('prevision', `1start=${tStart}&1end=${tEnd}&2start=${tsStart}&2end=${tsEnd}`)
      .then((res: any) => {
        return this.handleResponse(res);
      })
      .then((data: IGraph) => {
        this.previsions[index].graph = data;
        this.savePrevisions();
      })
      .catch(this.handleError);
  }

  async handleResponse(res: any) {
    if (res.status) {
      // No data: Error
      throw new Error('Something went wrong, please try again');
    }
    let graphData: IGraph;
    try {
      graphData = JSON.parse(res);
    } catch (err) {
      this.errMsg = err.message;
      this.isLoading = false;
      console.error(err);
      return;
    }
    this.isLoading = false;
    return await graphData;
  }

  handleError(err: Error) {
    this.errMsg = err.message || 'Error Uknown';
    console.error(err);
    this.isLoading = false;
  }

  /**
   * DATA ONLINE PART
   */

  // getData() {  // this.crypto, this.from
  //   return new Promise((resolve, reject) => {
  //     this.btc.getCryptoDataFrom().subscribe(async (res: any) => {
  //       if (res.Response && res.Response === 'Error') {
  //         this.errMsg = res.Message;
  //         reject();
  //       }
  //       console.log('Result', res);
  //       resolve(res.Data);
  //     });
  //   });
  // }

  // startPrevision(index: number) {
  //   console.log('Start', index);
  //   this.previsionSelected = this.previsions[index];
  //   this.crypto = this.previsions[index].crypto.name;
  //   this.from = this.previsions[index].crypto.from;
  //   this.getData().then((rawData: IRawData[]) => {
  //     return this.formatData(rawData);
  //   })
  //   .then((cryptoData: ICryptoTrendData[]) => this.previsions[index].crypto.data = cryptoData)
  //   .catch(console.error);
  // }

  // async formatData(rawData: any[]) {
  //   const data: ICryptoTrendData[] = [];
  //   rawData.forEach((item: IRawData) => {
  //       data.push({
  //           Timestamp: item.time,
  //           Open: item.open,
  //           High: item.high,
  //           Low: item.low,
  //           Close: item.close,
  //           'Volume_(BTC)': item.volumefrom,
  //           'Volume_(Currency)': item.volumeto,
  //           /** @todo What is that ? */
  //           'Weighted_Price': item.volumefrom
  //       });
  //   });
  //   return await Promise.all(data);
  // }

}
