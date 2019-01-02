import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { PlotlyModule } from 'angular-plotly.js';
import { AngularFontAwesomeModule } from 'angular-font-awesome';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HomeComponent } from './home/home.component';
import { IntroComponent } from './intro/intro.component';
import { NavComponent } from './nav/nav.component';
import { BotComponent } from './bot/bot.component';

import { CryptoService } from './services/crypto.service';
import { AddPrevisionComponent } from './add-prevision/add-prevision.component';


@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    IntroComponent,
    NavComponent,
    BotComponent,
    AddPrevisionComponent,

  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    NgbModule,
    HttpClientModule,
    FormsModule, ReactiveFormsModule,
    PlotlyModule,
    AngularFontAwesomeModule,
  ],
  providers: [ CryptoService ],
  bootstrap: [AppComponent]
})
export class AppModule { }
