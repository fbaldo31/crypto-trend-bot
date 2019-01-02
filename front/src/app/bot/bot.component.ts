import { Component, OnInit } from '@angular/core';
import { CryptoService } from '../services/crypto.service';
import Prevision, { IPrevisionOpts } from '../models/Prevision';

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
  public errMsg = '';

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
    console.log(event);
    let prevision: Prevision;
    try {
      prevision = new Prevision(<IPrevisionOpts>event);
    } catch (error) {
      console.error(error);
    }
    if (prevision) {
      if (this.previsions.find((item: Prevision) => item.label === prevision.label)) {
        this.errMsg = 'A prevision with the same label already exists';
        return;
      }
      this.previsions.push(prevision);
      this.savePrevisions();
    }
  }

  removePrevision(index: number) {
    this.previsions.splice(index);
    this.savePrevisions();
  }

  getData() {  // this.crypto, this.from
    return new Promise((resolve, reject) => {
      this.btc.getCryptoDataFrom().subscribe(async (res: any) => {
        if (res.Response && res.Response === 'Error') {
          this.errMsg = res.Message;
          reject();
        }
        console.log('Result', res);
        this.previsionSelected.crypto.formatData(res);
        resolve(this.previsionSelected);
      });
    });
  }

  savePrevisions() {
    localStorage.setItem('previsions', JSON.stringify(this.previsions));
  }

  startPrevision(index: number) {
    console.log('Start', index);
    this.previsionSelected = this.previsions[index];
    this.crypto = this.previsions[index].crypto.name;
    this.from = this.previsions[index].crypto.from;
    this.getData().then((data) => console.log(data)).catch(console.error);
  }

}
