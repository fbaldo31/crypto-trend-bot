import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-graph',
  templateUrl: './graph.component.html',
  styleUrls: ['./graph.component.scss']
})
export class GraphComponent {

  /**
   [
      { x: [1, 2, 3], y: [2, 6, 3], type: 'scatter', mode: 'lines+points', marker: {color: 'red'} },
      { x: [1, 2, 3], y: [2, 5, 3], type: 'bar' }
    ]
  */
  @Input() data: any;
  /** @example { width: 320, height: 240, title: 'A Fancy Plot' } */
  @Input() layout: any;
  public graph: any;
  hasData = false;

  constructor() {
    while (!this.graph) {
      this.hasData = false;
    }
    this.hasData = true;
  }

  init() {
    this.graph = { data: this.data, layout: this.layout };
    this.hasData = true;
  }
}
