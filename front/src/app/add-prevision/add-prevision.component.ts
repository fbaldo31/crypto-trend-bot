import { Component, Output, OnInit, EventEmitter } from '@angular/core';
import { FormControl, FormGroup } from '@angular/forms';

@Component({
  selector: 'app-add-prevision',
  templateUrl: './add-prevision.component.html',
  styleUrls: ['./add-prevision.component.scss']
})
export class AddPrevisionComponent implements OnInit {

  @Output() previsionAdded = new EventEmitter<boolean>();
  public addPrevisionForm = new FormGroup({
    previsionLabel: new FormControl(''),
    cryptoName: new FormControl(''),
    currency: new FormControl(''),
    from: new FormControl(365),
    splitDate: new FormControl('')
  });

  constructor() { }

  ngOnInit() {
  }

  onSubmitPrevisionForm() {
    console.warn(this.addPrevisionForm.value);
    if (this.addPrevisionForm.valid) {
      const data = this.addPrevisionForm.value;
      this.previsionAdded.emit(data);
    }
  }
}
