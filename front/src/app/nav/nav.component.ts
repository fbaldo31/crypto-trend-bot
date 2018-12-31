import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { IMenuLink } from '../models/IMenuLink';

@Component({
  selector: 'app-nav',
  templateUrl: './nav.component.html',
  styleUrls: ['./nav.component.scss']
})
export class NavComponent implements OnInit {

  public isCollapsed = false;
  public links: Set<IMenuLink> = new Set([
    { path: 'intro', label: 'Get started !', active: false },
    { path: 'bot', label: 'Go further', active: false, disabled: true }
  ]);

  constructor(private router: Router) { }

  ngOnInit() {
    this.setActiveLink();
  }

  private setActiveLink() {
    this.links.forEach((link: IMenuLink) => {
      if (link.path === this.router.url) {
        link.active = true;
      }
    });
  }

}
