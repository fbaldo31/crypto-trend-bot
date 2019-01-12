import { Component, Output, OnInit, EventEmitter } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';

@Component({
  selector: 'app-add-prevision',
  templateUrl: './add-prevision.component.html',
  styleUrls: ['./add-prevision.component.scss']
})
export class AddPrevisionComponent implements OnInit {

  public alerts: { type: string, message: string }[] = [];

  @Output() previsionAdded = new EventEmitter<boolean>();
  public addPrevisionForm: FormGroup;

  constructor() { }

  ngOnInit() {
    this.addPrevisionForm = new FormGroup({
      'label': new FormControl('', Validators.required),
      'cryptoName': new FormControl('', Validators.required),
      'currency': new FormControl('', Validators.required),
      'trainStartDate': new FormControl('2016-01-01', Validators.required),
      'trainEndDate': new FormControl('2018-12-31', Validators.required),
      'testStartDate': new FormControl('2018-12-01', Validators.required),
      'testEndDate': new FormControl('2018-12-31', Validators.required)
    });
  }

  onSubmitPrevisionForm() {
    // console.warn(this.addPrevisionForm.value);
    if (this.addPrevisionForm.valid) {
      const data = this.addPrevisionForm.value;
        this.previsionAdded.emit(data);
    } else {
      for (const error in this.addPrevisionForm.errors) {
        if (this.addPrevisionForm.errors.hasOwnProperty(error)) {
          this.alerts.push({ type: 'Error', message: error });
        }
      }
    }
  }

  formatDate(data: any): Date {
    const date = new Date();
    date.setFullYear(data.splitDate.year);
    date.setMonth(data.splitDate.mont);
    date.setDate(data.splitDate.day);
    console.log(date);
    return date;
  }

  closeAlert(i: number): void {
    this.alerts.splice(i, 1);
  }
}
