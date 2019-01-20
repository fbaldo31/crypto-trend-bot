import { Component, OnInit } from '@angular/core';

import { IGraph } from '../models/IGraph';
import { ApiService } from "../services/api.service";

@Component({
  selector: 'app-intro',
  templateUrl: './intro.component.html',
  styleUrls: ['./intro.component.scss']
})
export class IntroComponent implements OnInit {

  graphs: IGraph[];
  successMsg = '';
  isLoading = false;
  step = 0;
  errMsg = '';
  stepText = [
    'First we analyse the influence by such factors as trend or seasonality.',
    'The model has been trainned and we can can compare the Train Loss and the Test loss.'
  ];

  constructor(private api: ApiService) { }

  ngOnInit() {
    this.graphs = [];

  }

  nextStep() {
    this.errMsg = '';
    this.isLoading = true;
    if (this.step === 4) {
      this.errMsg = 'Intro has finished';
      return;
    }
    this.api.startStep(this.step + 1)
    .then((data: string) => {
      this.step++;
      this.graphs.push(JSON.parse(data) as IGraph);
      this.isLoading = false;
    })
    .catch((err: Error) => this.errMsg = 'Server error, please check your internet connexion and retry.');
  }

  // addImage(step: number): void {
  //   this.isLoading = true;
  //   this.fig1 = this.render.createElement('img');
  //   this.fig1.src = this.API_URL + '/step' + step;

  //   this.fig1.onload = (e: Event) => {
  //     this.isLoading = false;
  //     this.step++;
  //   };
  //   this.render.appendChild(this.render.selectRootElement('#step' + step + '-card'), this.fig1);
  // }
}
