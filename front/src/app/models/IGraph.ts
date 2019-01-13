import { ICryptoTrendData } from './Crypto';
/**
 * @example
 * let graph = {
 *   data: [
 *       { x: [1, 2, 3], y: [2, 6, 3], type: 'scatter', mode: 'lines+points', marker: {color: 'red'} },
 *      { x: [1, 2, 3], y: [2, 5, 3], type: 'bar' },
 *  ],
 *  layout: {width: 320, height: 240, title: 'A Fancy Plot'}
 *  }
 */
export interface IGraph {
    data: ICryptoTrendData[];
    layout: any;
    active?: boolean;
}
