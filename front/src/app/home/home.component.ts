import { Component, OnInit, Renderer2, ViewChild, ElementRef } from '@angular/core';
import { HttpClient, HttpResponse } from '@angular/common/http';

import { Observable } from 'rxjs';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

  readonly API_URL = 'http://localhost:8000';
  fig1: HTMLImageElement;
  isLoading = false;
  step1: ElementRef;
  step = 0;
  errMsg = '';

  constructor(private http: HttpClient, private render: Renderer2, private el: ElementRef) { }

  ngOnInit() {
  }

  nextStep() {
    this.addImage(this.step + 1);
  }

  addImage(step: number): void {
    this.isLoading = true;
    this.fig1 = this.render.createElement('img');
    this.fig1.src = this.API_URL + '/step' + step;

    this.fig1.onload = (e: Event) => {
      this.isLoading = false;
      this.step++;
    };
    this.render.appendChild(this.render.selectRootElement('#step' + step + '-card'), this.fig1);
  }

  handleResponse(res: HttpResponse<any>) {
    if (res.ok === true) {
      this.step++;
      console.log('Success', res.body);
      return res.body;
    } else {
      this.handleError(new Error(res.statusText));
    }
  }

  handleError(err: Error) {
    this.errMsg = err.message || 'Error Uknown';
    console.error(err);
  }

}
