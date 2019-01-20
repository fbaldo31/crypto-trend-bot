import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  readonly API_URL = 'http://0.0.0.0:8000';

  constructor(private http: HttpClient) { }

  async startStep(step: number) {
    return await new Promise((resolve, reject) => {
      try {
        resolve(this.http.get(this.API_URL + '/step' + step).toPromise());
      } catch (error) {
        reject(error);
      }
    });
  }
}
