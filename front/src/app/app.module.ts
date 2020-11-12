import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

import { NbAccordionModule, NbButtonModule, NbCardModule, NbThemeModule, NbLayoutModule, NbActionsModule, NbSpinnerModule } from '@nebular/theme';
import { NbEvaIconsModule } from '@nebular/eva-icons';

import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import * as PlotlyJS from 'plotly.js/dist/plotly.js';
import { PlotlyModule } from 'angular-plotly.js';
PlotlyModule.plotlyjs = PlotlyJS;

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HomeComponent } from './home/home.component';
import { IntroComponent } from './intro/intro.component';
import { NavComponent } from './nav/nav.component';
import { BotComponent } from './bot/bot.component';

import { CryptoService } from './services/crypto.service';
import { ApiService } from './services/api.service';
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
    BrowserAnimationsModule,
    NbActionsModule, NbAccordionModule,
    NbButtonModule, NbCardModule,
    NbThemeModule.forRoot({ name: 'default' }),
    NbLayoutModule,
    NbEvaIconsModule,
    NbSpinnerModule,
  ],
  providers: [ CryptoService, ApiService ],
  bootstrap: [AppComponent]
})
export class AppModule { }
