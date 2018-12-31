import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpResponse } from '@angular/common/http';

import { IGraph } from '../models/IGraph';

@Component({
  selector: 'app-intro',
  templateUrl: './intro.component.html',
  styleUrls: ['./intro.component.scss']
})
export class IntroComponent implements OnInit {

  readonly API_URL = 'http://localhost:8000';
  graphs: IGraph[];
  lastYearData;
  isLoading = false;
  step = 0;
  errMsg = '';

  constructor(private http: HttpClient) { }

  ngOnInit() {
    this.graphs = [];

  }

  nextStep() {
    this.isLoading = true;
    this.errMsg = '';
    this.http.get(this.API_URL + '/step' + (this.step + 1))
      .subscribe((res: any) => {
        this.handleResponse(res)
          .then((data: IGraph) => this.graphs.push(data))
          .catch((err: Error) => this.errMsg = err.message);
      });
  }

  // nextStep() {
  //   this.addImage(this.step + 1);
  // }

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

  async handleResponse(res: any) {
    let graphData: IGraph;
    try {
      graphData = JSON.parse(res);
    } catch (err) {
      this.errMsg = err.message;
      this.isLoading = false;
      console.error(err);
      return;
    }
    this.step++;
    this.isLoading = false;
    return await graphData;
  }

  handleError(err: Error) {
    this.errMsg = err.message || 'Error Uknown';
    console.error(err);
  }

}
