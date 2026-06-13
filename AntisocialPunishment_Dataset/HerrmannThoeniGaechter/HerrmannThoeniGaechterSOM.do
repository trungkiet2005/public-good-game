*STATA code for SUPPLEMENTARY ONLINE MATERIAL for Herrmann, Thoeni & Gaechter, Antisocial Punishment Across Societies, Science 319, 7 March 2008, 1362-1367, DOI: 10.1126/science.1153808
* Supplementary Online Material: http://www.sciencemag.org/cgi/content/full/319/5868/1362/DC1

*Use HerrmannThoeniGaechterDATA.dta

*************************************************************************************
*2. Subject pools and their societal/cultural background
*************************************************************************************

*____________________________________________________________________________________
* 2.1. Cultural and societal background of our subject pools


*TABLE S1
preserve
   collapse country trust civic kgdppercap ruleoflaw pdi idv mas uai surv_self trad_sec democracy, by(cityorder)
   qui do label.do
   label values country country
 forvalues i=1(1)4{
 gen _`i'=.
 }
 drop if cityorder==57
  order cityorder country civic trust _1  kgdppercap _2 ruleoflaw democracy _3 pdi idv mas uai _4 surv_self trad_sec
 outsheet using output\s01, replace noquote
restore


*FIGURE S1
local nod nodraw
*Panel A
local panel A
preserve
  use dta\wvs, clear
  gen observed=1 if inlist(country, 36, 112, 156, 208, 276, 300, 410, 512, 643, 682, 756, 792, 826, 804, 840)
  replace observed=0 if observed==.
  qui do label
  label values country country
  capture drop pos
  gen pos=3
  replace pos=9 if inlist(country,756,682,300,792)
  replace pos=2 if inlist(country, 826)
  replace pos=4 if inlist(country, 840, 410)
  local lcivic : variable label civic
  local ltrust : variable label trust
  drop if inlist(country,9996,9997,9998,9999)
  local var "trust civic"
  foreach i of local var{
   sum `i'
   local `i'm3=`r(mean)'-3*`r(sd)'
   local `i'm2=`r(mean)'-2*`r(sd)'
   local `i'm1=`r(mean)'-1*`r(sd)'
   local `i'0=`r(mean)'-0*`r(sd)'
   local `i'1=`r(mean)'+1*`r(sd)'
   local `i'2=`r(mean)'+2*`r(sd)'
   local `i'3=`r(mean)'+3*`r(sd)'
   local `i'4=`r(mean)'+4*`r(sd)'
  }
  gen a=0 in 1
  gen b=0 in 1
  replace a=100 in 2
  replace b=100 in 2
  sum trust
  local trustmin=`r(min)'
  local trustmax=`r(max)'
  sum civic
  local civicmin=`r(min)'
  local civicmax=`r(max)'
  /* old version (with normalization)
  #delimit ;
  tw (scatter trust civic if !observed, msymbol(O) mcolor(gs10)
      ymlabel(`trustm2' "-2" `trustm1' "-1" `trust0' "0" `trust1' "1" `trust2' "2"
      `trust3' "3" `trust4' "4", grid  labcolor(gs8) glcolor(gs8) glwidth(thin) notick)
      xmlabel(`civicm3' "-3" `civicm2' "-2" `civicm1' "-1" `civic0' "0" `civic1' "1" `civic2' "2", notick labcolor(gs8) glcolor(gs8) glwidth(thin) grid))
     (scatter trust civic if observed, msymbol(O) msize(*1) mlabel(country) mcolor(black) mlabvposition(pos) mlabsize(vsmall))
     (function y=`trust0', range(6.8 10) color(black) lwidth(medthin))
     (function y=`civic0', range(0 1) color(black) lwidth(medthin) horizontal)
     , legend(off) xtitle("Civic norms") ytitle("Trust")
     saving(output\fig9, replace)
     ;
  #delimit cr
  */
  #delimit ;
  tw (scatter trust civic if !observed, msymbol(O) mcolor(gs10)
      ymlabel(, grid  labcolor(gs8) glcolor(gs8) glwidth(thin) notick)
      xmlabel(, notick labcolor(gs8) glcolor(gs8) glwidth(thin) grid))
     (scatter trust civic if observed, msymbol(O) msize(*1) mlabel(country) mcolor(black) mlabvposition(pos) mlabsize(vsmall))
     (scatter a b, xaxis(2) yaxis(2) msymbol(none))
     (function y=`trust0', range(`civicmin' `civicmax') color(black) lwidth(thin))
     (function y=`civic0', range(`trustmin' `trustmax')  color(black) lwidth(thin) horizontal)
     , legend(off) xtitle("`lcivic'") ytitle("`ltrust'")
     xlabel(7(1)10) ylabel(0(.1).7)
     xlabel("", axis(2)) xtitle("", axis(2))
     ylabel("", axis(2)) ytitle("", axis(2))
     text(100 0 "`panel'", size(large) xaxis(2) yaxis(2) placement(se))
     `nod' name(p`panel', replace)
     ;
  #delimit cr
restore
***************
*Panel E
local panel E
preserve
  use dta\inglehart.dta, clear
  gen observed=1 if inlist(country, 36, 112, 156, 208, 276, 300, 410, 512, 643, 682, 756, 792, 826, 804, 840)
  replace observed=0 if observed==.
  qui do label
  label values country country
  capture drop pos
  gen pos=3
  replace pos=9 if inlist(country,156,208)
  replace pos=6 if inlist(country,804,826)
  replace pos=12 if inlist(country,276,36)
  sum trad_sec if country==9999
  local meantrad_sec=`r(mean)'
  sum trad_sec if country==9998
  local maxtrad_sec=`r(mean)'
  sum trad_sec if country==9997
  local mintrad_sec=`r(mean)'
  sum surv_self if country==9999
  local meansurv_self=`r(mean)'
  sum surv_self if country==9998
  local maxsurv_self=`r(mean)'
  sum surv_self if country==9997
  local minsurv_self=`r(mean)'
  local ltrad_sec : variable label trad_sec
  local lsurv_self : variable label surv_self
  drop if inlist(country,9996,9997,9998,9999)
  gen a=0 in 1
  gen b=0 in 1
  replace a=100 in 2
  replace b=100 in 2
  #delimit ;
  tw (scatter trad_sec surv_self if !observed, msymbol(O) mcolor(gs10)
      ymlabel(, grid  labcolor(gs8) glcolor(gs8) glwidth(thin) notick)
      xmlabel(, notick labcolor(gs8) glcolor(gs8) glwidth(thin) grid))
     (scatter trad_sec surv_self if observed, msymbol(O) msize(*1) mlabel(country) mcolor(black) mlabvposition(pos) mlabsize(vsmall))
     (scatter a b, xaxis(2) yaxis(2) msymbol(none))
     (function y=`meantrad_sec', range(`minsurv_self' `maxsurv_self') color(black) lwidth(thin))
     (function y=`meansurv_self', range(`mintrad_sec' `maxtrad_sec') color(black) lwidth(thin) horizontal)
     , legend(off) xtitle("`lsurv_self'") ytitle("`ltrad_sec'")
     xlabel("", axis(2)) xtitle("", axis(2))
     ylabel("", axis(2)) ytitle("", axis(2))
     text(100 0 "`panel'", size(large) xaxis(2) yaxis(2) placement(se))
     `nod' name(p`panel', replace)
     ;
  #delimit cr
restore
***************
*Panel C & D
local panel1 C
local panel2 D
preserve
  use dta\hofstede, clear
  gen observed=1 if inlist(country, 36, 112, 156, 208, 276, 300, 410, 512, 643, 682, 756, 792, 826, 804, 840)
  replace observed=0 if observed==.
  qui do label
  label values country country
  sum pdi if country==9999
  local meanpdi=`r(mean)'
  sum pdi if country==9998
  local maxpdi=`r(mean)'
  sum pdi if country==9997
  local minpdi=`r(mean)'
  sum idv if country==9999
  local meanidv=`r(mean)'
  sum idv if country==9998
  local maxidv=`r(mean)'
  sum idv if country==9997
  local minidv=`r(mean)'
  sum mas if country==9999
  local meanmas=`r(mean)'
  sum mas if country==9998
  local maxmas=`r(mean)'
  sum mas if country==9997
  local minmas=`r(mean)'
  sum uai if country==9999
  local meanuai=`r(mean)'
  sum uai if country==9998
  local maxuai=`r(mean)'
  sum uai if country==9997
  local minuai=`r(mean)'
  local lpdi : variable label pdi
  local lidv : variable label idv
  local lmas : variable label mas
  local luai : variable label uai
  drop if inlist(country,9996,9997,9998,9999)
  gen a=0 in 1
  gen b=0 in 1
  replace a=100 in 2
  replace b=100 in 2
  capture drop pos
  gen pos=3
  replace pos=6 if inlist(country,300, 410)
  replace pos=9 if inlist(country,826)
  replace pos=12 if inlist(country,792,682,643)
  #delimit ;
  tw (scatter pdi idv if !observed, msymbol(O) mcolor(gs10)
      ymlabel(, grid  labcolor(gs8) glcolor(gs8) glwidth(thin) notick)
      xmlabel(, notick labcolor(gs8) glcolor(gs8) glwidth(thin) grid))
     (scatter pdi idv if observed, msymbol(O) msize(*1) mlabel(country) mcolor(black) mlabvposition(pos) mlabsize(vsmall))
     (scatter a b, xaxis(2) yaxis(2) msymbol(none))
     (function y=`meanpdi', range(`minidv' `maxidv') color(black) lwidth(thin))
     (function y=`meanidv', range(`minpdi' `maxpdi') color(black) lwidth(thin) horizontal)
     , legend(off) xtitle("`lidv'") ytitle("`lpdi'")
     xlabel("", axis(2)) xtitle("", axis(2))
     ylabel("", axis(2)) ytitle("", axis(2))
     text(100 0 "`panel1'", size(large) xaxis(2) yaxis(2) placement(se))
     `nod' name(p`panel1', replace)
     ;
  #delimit cr
  capture drop pos
  gen pos=3
  replace pos=9 if inlist(country,156,756,410,300,840)
  #delimit ;
  tw (scatter mas uai if !observed, msymbol(O) mcolor(gs10)
      ymlabel(, grid  labcolor(gs8) glcolor(gs8) glwidth(thin) notick)
      xmlabel(, notick labcolor(gs8) glcolor(gs8) glwidth(thin) grid))
     (scatter mas uai if observed, msymbol(O) msize(*1) mlabel(country) mcolor(black) mlabvposition(pos) mlabsize(vsmall))
     (scatter a b, xaxis(2) yaxis(2) msymbol(none))
     (function y=`meanmas', range(`minuai' `maxuai') color(black) lwidth(thin))
     (function y=`meanuai', range(`minmas' `maxmas') color(black) lwidth(thin) horizontal)
     , legend(off) xtitle("`luai'") ytitle("`lmas'")
     xlabel("", axis(2)) xtitle("", axis(2))
     ylabel("", axis(2)) ytitle("", axis(2))
     text(100 0 "`panel2'", size(large) xaxis(2) yaxis(2) placement(se))
     `nod' name(p`panel2', replace)
     ;
  #delimit cr
restore
***************
*Panel B
local panel B
preserve
  use dta\weo.dta, clear
  rename iso code
  sort code country
  joinby code country using dta\wgi.dta, unmatched(both)
  gen observed=1 if inlist(country, 36, 112, 156, 208, 276, 300, 410, 512, 643, 682, 756, 792, 826, 804, 840)
  replace observed=0 if observed==.
  qui do label
  label values country country
  capture drop pos
  gen pos=3
  replace pos=9 if inlist(country,36,112,276,826, 208)
  replace pos=5 if inlist(country, 804)
  replace pos=1 if inlist(country,643)
  replace pos=12 if inlist(country,756)
  sum gdppercap if country==9999
  local meangdp=`r(mean)'
  sum gdppercap if country==9998
  local maxgdp=`r(mean)'
  sum gdppercap if country==9997
  local mingdp=`r(mean)'
  sum ruleoflaw if country==9999
  local meanrol=`r(mean)'
  sum ruleoflaw if country==9998
  local maxrol=`r(mean)'
  sum ruleoflaw if country==9997
  local minrol=`r(mean)'
  local lgdp : variable label gdppercap
  local lrol : variable label ruleoflaw
*spearman correlations reported in text
local vars vo_ac po_st go_eff reg_qu co_corr
foreach i of local vars{
spearman ruleoflaw `i'
}
local vars vo_ac po_st go_eff reg_qu co_corr
foreach i of local vars{
spearman ruleoflaw `i' if observed
}
spearman  gdppercap ruleoflaw if country<9000
spearman  gdppercap ruleoflaw if observed
  drop if inlist(country,9996,9997,9998,9999)
  drop if  countryname=="Luxembourg"  //Outlier eliminated!!!
  gen a=0 in 1
  gen b=0 in 1
  replace a=100 in 2
  replace b=100 in 2
  #delimit ;
  tw (scatter gdppercap ruleoflaw if !observed, msymbol(O) mcolor(gs10)
      ymlabel(, grid  labcolor(gs8) glcolor(gs8) glwidth(thin) notick)
      xmlabel(, notick labcolor(gs8) glcolor(gs8) glwidth(thin) grid))
     (scatter gdppercap ruleoflaw if observed, msymbol(O) msize(*1) mlabel(country) mcolor(black) mlabvposition(pos) mlabsize(vsmall))
     (scatter a b, xaxis(2) yaxis(2) msymbol(none))
     (function y=`meangdp', range(`minrol' 2.3) color(black) lwidth(thin))
     (function y=`meanrol', range(`mingdp' 44000) color(black) lwidth(thin) horizontal)
     , legend(off) xtitle("`lrol'") ytitle("`lgdp'")
     xlabel("", axis(2)) xtitle("", axis(2))
     ylabel("", axis(2)) ytitle("", axis(2))
     text(100 0 "`panel'", size(large) xaxis(2) yaxis(2) placement(se))
     `nod' name(p`panel', replace)
     ;
  #delimit cr
restore
*Figure S1
graph combine pA pB pC pD pE, saving(output\figS01, replace) cols(2) ysize(7.5) xsize(6) iscale(*.8)

*/


*____________________________________________________________________________________
* 2.2 Subject pool details
*TABLE S2
preserve
   keep if data
   collapse (mean) p np session city country age female numknown cityorder urbanbackground singlechild middleclass membership, by(subjectid)
   egen nses=count(subjectid), by(session)
   gen percknown=numknown/nses
   sum age
   local mage=`r(mean)'
   sum female
   local mfem=`r(mean)'
   sum percknown
   local mperc=`r(mean)'
   local nsub=_N
   sum urbanbackground
   local murb=`r(mean)'
   sum singlechild
   local msc=`r(mean)'
   sum middleclass
   local mmidc=`r(mean)'
   sum membership
   local mmem=`r(mean)'
   gen n=1
   gen n_np=1 if np & p!=0 & p!=1
   gen n_pn=1 if !np & p!=0 & p!=1
   gen n_n=1 if p==0
   gen n_p=1 if p==1
   collapse (mean) country age female percknown urbanbackground singlechild middleclass membership cityorder  (sum) n-n_p , by(city)
   lab values city city
   lab values country country
   sort city
   gen str20 university=""
 *adding data from other sources
   replace university = "The Institute for Empirical Research in Economics (IERE), University of Zurich" if city== 1
   replace university = "Samara State University" if city== 2
   replace university = "Belarusian National Technical University" if city== 3
   replace university = "University Louis Pasteur" if city== 4
   replace university = "Harvard University" if city== 5
   replace university = "Sultan Qaboos University" if city== 6
   replace university = "Research Institute for Empirical Economics and Economic Policy, University of St. Gallen" if city== 7
   replace university = "University of Copenhagen" if city== 8
   replace university = "University of Nottingham" if city== 9
   replace university = "Dnipropetrovs'k Regional Institute of Public Administration"  if city==10
   replace university = "Imam Muhammad bin Saud University"  if city==11
   replace university = "Bogazici University"  if city==12
   replace university = "Southwest Jiaotong University "  if city==13
   replace university = "Chung-Ang University"  if city==14
   replace university = "Bonn Laboratory for Experimental Economics, University of Bonn"  if city==15
   replace university = "Panteion University"  if city==16
   replace university = "University of Melbourne"  if city==17
   replace university = "University of Zurich" if city==1
   replace university = "University of St. Gallen" if city==7
   replace university = "University of Bonn" if city==15
   gen str exchangerate=""
   gen str lumpsum=""
   replace exchangerate="CHF .07" if city== 1
   replace exchangerate="RUB .2" if city== 2
   replace exchangerate="BYR 17" if city== 3
   replace exchangerate="EUR .075" if city== 4
   replace exchangerate="USD .03" if city== 5
   replace exchangerate="OMR .006" if city== 6
   replace exchangerate="CHF .07" if city== 7
   replace exchangerate="DKK .3" if city== 8
   replace exchangerate="GBP .015" if city== 9
   replace exchangerate="UAH .03"  if city==10
   replace exchangerate="SAR .15"  if city==11
   replace exchangerate="TRY .04"  if city==12
   replace exchangerate="CNY .07"  if city==13
   replace exchangerate="KRW 20"  if city==14
   replace exchangerate="EUR .025"  if city==15
   replace exchangerate="EUR .02"  if city==16
   replace exchangerate="AUD .09"  if city==17
   replace lumpsum="CHF 10" if city== 1
   replace lumpsum="RUB 10" if city== 2
   replace lumpsum="BYR 1000" if city== 3
   replace lumpsum="EUR " if city== 4
   replace lumpsum="USD 10" if city== 5
   replace lumpsum="OMR 1" if city== 6
   replace lumpsum="" if city== 7
   replace lumpsum="DKK 60" if city== 8
   replace lumpsum="GBP 3" if city== 9
   replace lumpsum="UAH 15"  if city==10
   replace lumpsum="SAR 30"  if city==11
   replace lumpsum="TRY 5"  if city==12
   replace lumpsum="CNY 15"  if city==13
   replace lumpsum="KRW 5000"  if city==14
   replace lumpsum="EUR 4"  if city==15
   replace lumpsum="EUR 5"  if city==16
   sort cityorder
   drop cityorder
   local nobs=_N+1
   set obs `nobs'
   replace city=9990 in `nobs'
   replace age=`mage' in `nobs'
   replace female=`mfem' in `nobs'
   replace percknown=`mperc' in `nobs'
   replace urbanbackground=`murb' in `nobs'
   replace singlechild=`msc' in `nobs'
   replace middleclass=`mmidc' in `nobs'
   replace membership=`mmem' in `nobs'
   replace n=`nsub' in `nobs'
   sum n_np
   replace n_np=`r(sum)' in `nobs'
   sum n_pn
   replace n_pn=`r(sum)' in `nobs'
   sum n_n
   replace n_n=`r(sum)' in `nobs'
   sum n_p
   replace n_p=`r(sum)' in `nobs'
   gen n_reverse=n_pn + n_p
   tostring n_np, replace
   tostring n_pn, replace
   tostring n_n, replace
   tostring n_p, replace
   tostring n_reverse, replace
   tostring n, replace
   gen a="/"
   gen n_treat=n_np+a+n_pn+a+n_n+a+n_p
   drop n_np n_pn n_n n_p a
   gen n_2=n+" ("+n_reverse+")"
   replace age=round(age, 0.1)
   gen female1=round(1000*female)/10
   drop female
   replace percknown=round(100*percknown, 0.1)
   replace middleclass=round(100*middleclass,.1)
   replace singlechild=round(100*singlechild,.1)
   replace membership=round(100*membership,.1)
   replace urbanbackground=round(100*urbanbackground,.1)
   drop lumpsum n n_reverse n_treat
   order  city university country exchangerate n_2 percknown age female1 urbanbackground middleclass singlechild membership
   outsheet using output\s02, replace noquote
restore



*************************************************************************************
*3. Methodological issues
*************************************************************************************
preserve
collapse (count) subjectid if period==1 & p==1 & data==1, by(session)
replace subject=subject/3
sum subject
restore


*sequence effects
preserve
 collapse c np city , by(p mgroupid)
*St Gallen
   ttest c if p==0 & city==7, by(np)
   ranksum c if p==0 & city==7, by(np)
   ttest c if p==1 & city==7, by(np)
   ranksum c if p==1 & city==7, by(np)
*Minsk
   ttest c if p==0 & city==3, by(np)
   ranksum c if p==0 & city==3, by(np)
   ttest c if p==1 & city==3, by(np)
   ranksum c if p==1 & city==3, by(np)
*Samara
   ttest c if p==0 & city==2, by(np)
   ranksum c if p==0 & city==2, by(np)
   ttest c if p==1 & city==2, by(np)
   ranksum c if p==1 & city==2, by(np)

*Zurich
   ttest c if p==1 & city==1, by(np)
   ranksum c if p==1 & city==1, by(np)


restore

*************************************************************************************
*4. Supporting analyses
*************************************************************************************

*____________________________________________________________________________________
* 4.1 Punishment behavior

*FIGURE S2
#delimit ;
graph bar dpun if data, over(devclass, label(labsize(vsmall) )) by(cityorder)
ytitle("Relative frequency of punishment") saving(output\figS02, replace)
;
#delimit cr


*TABLE S3
set more off
*negative deviations
  levelsof cityorder if data, local(places)
  foreach i of local places{
  display "-----------------> negative deviations, City `i'
  intreg punll punul otherscontribution senderscontribution avcother2 recpunL1 period finalperiod if negdev & cityorder==`i', cluster(mgroupid)
   estimates store es1_`i'
   }
    #delimit ;
    estout  es1_51 es1_52 es1_53 /*es1_54*/ es1_55 es1_56 es1_57 es1_58 es1_59 es1_60 es1_61 es1_62 es1_63 es1_64 es1_65 es1_66 es1_67
        using output\s03A.out, replace
        cells(b(star fmt(%9.3f))  se(par ))
        stats(sigma p ll  N, fmt(%9.3f %9.3f %9.3g %9.0g) )
        starlevels(* 0.1 ** 0.05 *** 0.01)
        varwidth(16) modelwidth(12)
        margin  style(tab)
        legend label varlabels(_cons Constant)
        posthead("") prefoot("") postfoot("")
        ;
    #delimit cr
*nonnegative deviations
  gen revengeASP=.
    levelsof cityorder if data, local(places)
    foreach i of local places{
    display "-----------------> non-negative deviations, City `i'
    intreg punll punul otherscontribution senderscontribution avcother2 recpunL1  period finalperiod if !negdev & cityorder==`i', cluster(mgroupid)
     estimates store es1_`i'
     replace revengeASP=_coef[recpunL1] if cityorder==`i'  & p==1
     }
      #delimit ;
      estout es1_51 es1_52 es1_53 /*es1_54*/ es1_55 es1_56 es1_57 es1_58 es1_59 es1_60 es1_61 es1_62 es1_63 es1_64 es1_65 es1_66 es1_67
          using output\s03B.out, replace
          cells(b(star fmt(%9.3f))  se(par ))
           stats(sigma p ll  N, fmt(%9.3f %9.3f %9.3g %9.0g) )
          starlevels(* 0.1 ** 0.05 *** 0.01)
          varwidth(16) modelwidth(12)
          margin  style(tab)
          legend label varlabels(_cons Constant)
          posthead("") prefoot("") postfoot("")
          ;
      #delimit cr



*TABLE S4
set more off
   intreg punll punul otherscontribution senderscontribution avcother2 period finalperiod dcity51-dcity67 female ageu21 singlechild urbanbackground middleclass membership numknown  if negdev, nocons cluster(mgroupid)
   estimates store s4a
   test  dcity51= dcity52= dcity53= dcity55= /*dcity56=*/ dcity57= dcity58= dcity59= dcity60= dcity61= dcity62= dcity63= dcity64= dcity65= dcity66 = dcity67
   intreg punll punul otherscontribution senderscontribution avcother2 recpunL1 period finalperiod dcity51-dcity67 female ageu21 singlechild urbanbackground middleclass membership numknown  if negdev, nocons cluster(mgroupid)
   estimates store s4a2
   test  dcity51= dcity52= dcity53= dcity55= /*dcity56=*/ dcity57= dcity58= dcity59= dcity60= dcity61= dcity62= dcity63= dcity64= dcity65= dcity66 = dcity67
   intreg punll punul otherscontribution senderscontribution avcother2 period finalperiod dcity51-dcity67 female ageu21 singlechild urbanbackground middleclass membership numknown  if !negdev, nocons cluster(mgroupid)
   estimates store s4b
   test  dcity51= dcity52= dcity53= dcity55= /*dcity56=*/ dcity57= dcity58= dcity59= dcity60= dcity61= dcity62= dcity63= dcity64= dcity65= dcity66 = dcity67
   intreg punll punul otherscontribution senderscontribution avcother2 recpunL1 period finalperiod dcity51-dcity67 female ageu21 singlechild urbanbackground middleclass membership numknown  if !negdev, nocons cluster(mgroupid)
   estimates store s4b2
   test  dcity51= dcity52= dcity53= dcity55= /*dcity56=*/ dcity57= dcity58= dcity59= dcity60= dcity61= dcity62= dcity63= dcity64= dcity65= dcity66 = dcity67
       #delimit ;
       estout  s4a s4a2 s4b s4b2
           using output\s04.out, replace
           cells(b(star fmt(%9.3f))  se(par ))
           stats(sigma p ll  N, fmt(%9.3f %9.3f %9.3g %9.0g) )
           indicate("Socio-economic controls" = female ageu21 singlechild urbanbackground middleclass membership numknown  , labels(yes no))
           starlevels(* 0.1 ** 0.05 *** 0.01)
           varwidth(16) modelwidth(12)
           margin  style(tab)
           legend label varlabels(_cons Constant)
           posthead("") prefoot("") postfoot("")
           ;
       #delimit cr



*____________________________________________________________________________________
* 4.2 Cooperation in the P-experiment

*TABLE S5
  levelsof cityorder if data, local(places)
   foreach i of local places{
     display "--------------------> `i'"
   intreg cll cul period finalperiod if p & cityorder==`i', cluster(mgroupid)
   estimates store es1_`i'
   }
 #delimit ;
  quietly  estout es1_51 es1_52 es1_53 /* es1_54 */ es1_55 es1_56 es1_57 es1_58 es1_59 es1_60 es1_61 es1_62 es1_63 es1_64 es1_65 es1_66 es1_67
        using output\s05.out, replace
        cells(b(star fmt(%9.3f))  se(par ))
        stats(sigma p ll  N, fmt(%9.3f %9.3f %9.3g %9.0g) )
        starlevels(* 0.1 ** 0.05 *** 0.01)
        varwidth(16) modelwidth(12)
        margin  style(tab)
        legend label varlabels(_cons Constant)
        posthead("") prefoot("") postfoot("")
        ;
    #delimit cr




*____________________________________________________________________________________
*4.3 (Relative) payoffs in the N- and the P-experiment (efficiency)

*TABLE S6
table cityorder p, c(mean profit)  sc



*FIGURE S3
preserve
   local obs = _N
   local obsnew = `obs' + 10
   set obs `obsnew'
   replace period= _n - `obs' if period==.
   replace p=0 if p==.
   local obs = _N
   local obsnew = `obs' + 10
   set obs `obsnew'
   replace period= _n - `obs' if period==.
   replace p=1 if p==.
   replace city=99 if city==.
   egen avp=mean(profit), by(period p)
   replace profit=avp if city==99
   egen m_profit=mean(profit), by(p city)
   collapse (mean) profit m_profit, by(city p period)
   sort city period p
   gen ratio=profit/profit[_n-1]
   gen m_ratio=m_profit/m_profit[_n-1]
   keep if p
   drop p profit
   local xlabeldot=10.8              //x-axis of label dots (max one digit after comma)
   local xlabel=11                   //x-axis of label  (max one digit after comma)
drop if city==99
   inspect city
   local nocities=`r(N_unique)'
   local obs = _N
   local obsnew =`obs' + `nocities'
   set obs `obsnew'
   replace period=(10 + `xlabeldot')/2 if period==.
   local obs = _N
   local obsnew =`obs' + `nocities'
   set obs `obsnew'
   replace period=`xlabeldot' if period==.
   local obs = _N
   local obsnew =`obs' + `nocities'
   set obs `obsnew'
   replace period=`xlabel' if period==.
   sort period city
   replace city=city[_n - `nocities']  if city==.
   sort city period
   replace m_ratio = m_ratio[_n-1] if m_ratio==.
   sort period m_ratio
   gen rank= mod(_n-1,`nocities')+1  //ranking of the cities according to mean ratio
   sum ratio
 *  local ymin = `r(min)' // range of labels along y-axis
  local ymin = .22 // range of labels along y-axis
     display `ymin'
   local ymax = `r(max)'
   gen ylabel = (rank - 1)*(`ymax' - `ymin')/(`nocities'-1)+`ymin'
   replace ratio = ylabel if round(period,.1)== round(`xlabeldot',.1)
   sort city period
   label define city 99 Mean, modify
 *add mean ratio to label
   tostring m_ratio, gen(m_ratio1) force usedisplayformat
   replace m_ratio1=substr(m_ratio1,1,4) if m_ratio>=1
   replace m_ratio1=substr(m_ratio1,1,3) if m_ratio<1
   decode city, gen(city1)
   gen city2=city1 + " ("+m_ratio1+")"
   encode city2, gen(city3)
     *legend symbol fo mean
     local newobs=_N+1
     set obs `newobs'
     replace city=-1 if city==.
     local newobs=_N+1
     set obs `newobs'
     replace city=-2 if city==.
     replace period=`xlabeldot'-.2 if city==-1
     replace period=`xlabeldot'+.2 if city==-2
     replace ylabel=ylabel[_n-1] if city<0
     replace ratio=ylabel if city<0
     replace city=99 if city<0
drop if city==99 //***************************eliminate av
   local sym1 "msymbol(O) msize(*1.5) mlcolor(black)"
   local sym2 "msymbol(s) msize(*1.8) mlcolor(black)"
   local sym3 "msymbol(d) msize(*1.5) mlcolor(black)"
   local sym4 "msymbol(t) msize(*1.8) mlcolor(black)"
     #delimit ;
      tw
    (scatter ratio period, connect(l) clwidth(medthin) cmissing(n) msymbol(i))
    (scatter ratio period if city==5, `sym1'  mfcolor("0 128 0"))
    (scatter ratio period if city==8,  `sym2' mfcolor("0 255 0"))
    (scatter ratio period if city==7,  `sym3' mfcolor("128 255 128"))
    (scatter ratio period if city==1, `sym4' mfcolor("128 0 0"))
    (scatter ratio period if city==9, `sym1' mfcolor("255 0 0"))
    (scatter ratio period if city==14, `sym2' mfcolor("255 128 128"))
    (scatter ratio period if city==15, `sym3' mfcolor("0 0 128"))
    (scatter ratio period if city==17, `sym4' mfcolor("0 0 255"))
    (scatter ratio period if city==13, `sym1' mfcolor("128 128 255"))
    (scatter ratio period if city==3, `sym2' mfcolor("128 128 0"))
    (scatter ratio period if city==2, `sym3' mfcolor("128 64 0"))
    (scatter ratio period if city==10, `sym4' mfcolor("255 128 0"))
    (scatter ratio period if city==6, `sym1' mfcolor("255 255 0"))
    (scatter ratio period if city==12, `sym2' mfcolor("128 0 128"))
    (scatter ratio period if city==11, `sym3' mfcolor("128 0 64"))
    (scatter ratio period if city==16, `sym4' mfcolor("128 128 128"))
/*  (scatter ratio period if city==99, connect(l) clwidth(medthick) cmissing(n) msymbol(i))*/
    (scatter ylabel period if round(period,.1)==round(`xlabel',.1), msymbol(i) mlabel(city3)  mlabsize(vsmall))
     (function y=1, range(1 10) lpattern(solid) clwidth(medthin))
     , legend(off) xtitle("Period") ytitle("Earnings in P / Earnings in N") xscale(range(1 13)) xlabel(1(1)10) ylabel(0.2(.2)1.4)
     saving(output\figS03, replace);
     #delimit cr
restore


*Earnings ratio and period: OLS
preserve
  collapse city profit if bothtreat & data, by(group p period)
  sort city groupid period p
  gen ratio=profit/profit[_n-1] if p
  by city: reg ratio period, r cluster(groupid)
restore

*____________________________________________________________________________________
*4.4 Reactions to received punishment

*TABLE S7
qui do reaction.do
     #delimit ;
     estout es1_51 es1_52 es1_53 /* es1_54 */ es1_55 es1_56 es1_57 es1_58 es1_59 es1_60 es1_61 es1_62 es1_63 es1_64 es1_65 es1_66 es1_67
         using output\s07a.out, replace
         cells(b(star fmt(%9.3f))  se(par ))
         stats(r2  N, fmt(%9.3f %9.0g))
         starlevels(* 0.1 ** 0.05 *** 0.01)
         varwidth(16) modelwidth(12)
         margin  style(tab)
         legend label varlabels(_cons Constant)
         posthead("") prefoot("") postfoot("")
         ;
     estout es2_51 es2_52 es2_53 /* es2_54 */ es2_55 es2_56 es2_57 es2_58 es2_59 es2_60 es2_61 es2_62 es2_63 es2_64 es2_65 es2_66 es2_67
         using output\s07b.out, replace
         cells(b(star fmt(%9.3f))  se(par ))
         stats(r2  N, fmt(%9.3f %9.0g))
         starlevels(* 0.1 ** 0.05 *** 0.01)
         varwidth(16) modelwidth(12)
         margin  style(tab)
         legend label varlabels(_cons Constant)
         posthead("") prefoot("") postfoot("")
         ;
     #delimit cr

*____________________________________________________________________________________
*4.5 Cooperation in the N-experiment and the change in contributions between the N- and P-experiment

*FIGURE S4: NP figures with confidence bounds (a la nature paper)
preserve
   drop if !data
   collapse (mean) c cityorder if data , by(p mgroupid period)
   label values cityorder cityorder
   gen clb = .
   gen cub = .
   levelsof cityorder, local(places)
   levelsof period, local(per)
   foreach i of numlist 0 1 {
    foreach j of local places{
   display "`i' -- `j'"
   qui foreach k of local per{
     mean c if p==`i' & cityorder==`j' & period==`k'
     replace clb=_coef[c]-1.96*_se[c] if p==`i' & cityorder==`j' & period==`k'
     replace cub=_coef[c]+1.96*_se[c] if p==`i' & cityorder==`j' & period==`k'
     }
    }
   }
  collapse (mean) c clb cub, by(cityorder p period)
  lab var cityorder "Subject pool
  gen period2=period
  replace period2=period+10 if p
   inspect cityorder
   local obs = _N
   local obsnew =`obs' + `r(N_unique)'
   set obs `obsnew'
   sort p period cityorder
   replace cityorder=cityorder[_n-`r(N_unique)'] if cityorder==.
   replace period2=10.5 if period2==.  //insert break to prevent connecting
   sort cityorder period2
   gen lab=p if inlist(period2,1,11)
   label values lab p
   gen ylab=.5 if lab!=.
  #delimit ;
  tw   (scatter c clb cub period2,
  msymbol(O i i) mcolor(black black black) connect(l l l) clwidth(medthin thin thin) lpattern(solid shortdash shortdash) cmissing(n n n) legend(off)  )
  (scatter ylab period2, msymbol(i) mlabel(lab) mlabsize(vsmall))
  ,by(cityorder, legend(off))
  xtitle("Period")
  xlabel(1 "1" 2"2" 3 "3" 4 "4" 5 "5" 6 "6" 7 "7" 8 "8" 9 "9" 10 "10" 11 "1" 12"2" 13 "3" 14 "4" 15 "5" 16 "6" 17 "7" 18 "8" 19 "9" 20 "10", labsize(vsmall))
  ytitle("Contribution")
  ylabel( ,labsize(vsmall))
  saving(output\figS04, replace);
  #delimit cr
restore


*TABLE S8
set more off
  levelsof cityorder if data, local(places)
   foreach i of local places{
     display "--------------------> `i'"
     intreg cll cul period finalperiod if !p & cityorder==`i', cluster(mgroupid)
   estimates store es1_`i'
   }
    #delimit ;
  quietly  estout es1_51 es1_52 es1_53 /* es1_54 */ es1_55 es1_56 es1_57 es1_58 es1_59 es1_60 es1_61 es1_62 es1_63 es1_64 es1_65 es1_66 es1_67
        using output\s08.out, replace
        cells(b(star fmt(%9.3f))  se(par ))
        stats(sigma p ll  N, fmt(%9.3f %9.3f %9.3g %9.0g) )
        starlevels(* 0.1 ** 0.05 *** 0.01)
        varwidth(16) modelwidth(12)
        margin  style(tab)
        legend label varlabels(_cons Constant)
        posthead("") prefoot("") postfoot("")
        ;
    #delimit cr

***********
*The change in contributions between the N- and the P-experiment
*Table S9 - part 1
preserve
 drop if c==.
 drop if period!=1
 replace mgroupid=session if city==2  //session as indep obs in Samara
 keep p c mgroupid cityorder subjectid
 reshape wide c, i(subjectid) j(p)
 *summary stat
  local n=_N+1
  set obs `n'
  replace mgroupid=9999 in `n'
  replace cityorder=99 in `n'
  egen t0=mean(c0)
  egen t1=mean(c1)
  replace c0=t0 if cityorder==99
  replace c1=t1 if cityorder==99
  collapse c0 c1 cityorder, by(mgroupid)
  label values cityorder cityorder
  gen pval=.
  levelsof cityorder, local(places)
  quietly  foreach i of local places{
  signrank c0=c1 if cityorder==`i'
  replace pval=2*normprob(-abs(`r(z)')) if cityorder==`i'
  }
  signrank c0=c1
  replace pval=2*normprob(-abs(`r(z)')) if cityorder==99
  collapse c0 c1 pval, by(cityorder)
  gen perc=(c1-c0)/c0*100
  order cityorder c0 c1 perc pval
  replace c0=round(c0,.1)
  replace c1=round(c1,.1)
  replace perc=round(perc,.1)
  replace pval=round(pval,.001)
  outsheet using output\s09p1, replace noquote
restore


*Table S9 - part 2
preserve
 drop if c==.
 replace mgroupid=session if city==2  //session as indep obs in Samara
 keep p c mgroupid cityorder subjectid period
 reshape wide c, i(subjectid period) j(p)
 *summary stat
  local n=_N+1
  set obs `n'
  replace mgroupid=9999 in `n'
  replace cityorder=99 in `n'
  egen t0=mean(c0)
  egen t1=mean(c1)
  replace c0=t0 if cityorder==99
  replace c1=t1 if cityorder==99
  collapse c0 c1 cityorder, by(mgroupid)
  label values cityorder cityorder
  gen pval=.
  levelsof cityorder, local(places)
  quietly  foreach i of local places{
  signrank c0=c1 if cityorder==`i'
  replace pval=2*normprob(-abs(`r(z)')) if cityorder==`i'
  }
  signrank c0=c1
  replace pval=2*normprob(-abs(`r(z)')) if cityorder==99
  collapse c0 c1 pval, by(cityorder)
  gen perc=(c1-c0)/c0*100
  order cityorder c0 c1 perc pval
  replace c0=round(c0,.1)
  replace c1=round(c1,.1)
  replace perc=round(perc,.1)
  replace pval=round(pval,.001)
  outsheet using output\s09p2, replace noquote
restore



*____________________________________________________________________________________
*2.6 Anti-social punishment and the economic and cultural back-ground of societies

*TABLE S10
intreg punll punul  trust otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if !negdev, cluster(mgroupid)
estimates store PNND_trust
intreg punll punul   kgdppercap  otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if !negdev, cluster(mgroupid)
estimates store PNND_kgdppercap
intreg punll punul  democracy otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if !negdev, cluster(mgroupid)
estimates store PNND_democracy
intreg punll punul  pdi otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if !negdev, cluster(mgroupid)
estimates store PNND_pdi
intreg punll punul  idv otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if !negdev, cluster(mgroupid)
estimates store PNND_idv
intreg punll punul  mas otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if !negdev, cluster(mgroupid)
estimates store PNND_mas
intreg punll punul  uai otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if !negdev, cluster(mgroupid)
estimates store PNND_uai
intreg punll punul   trad_sec otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if !negdev, cluster(mgroupid)
estimates store PNND_trad_sec
intreg punll punul   surv_self otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if !negdev, cluster(mgroupid)
estimates store PNND_surv_self
     #delimit ;
      estout
          PNND_trust PNND_kgdppercap PNND_democracy PNND_pdi PNND_idv PNND_mas PNND_uai PNND_trad_sec PNND_surv_self
          using output\s10.out, replace
          cells(b(star fmt(%9.3f))  se(par ))
          stats(sigma p ll  N, fmt(%9.3f %9.3f %9.3g %9.0g) )
          order(trust kgdppercap democracy pdi idv mas uai trad_sec surv_self)
          indicate("Controls" =  otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown , labels(yes no))
          starlevels(* 0.1 ** 0.05 *** 0.01)
          varwidth(16) modelwidth(12)
          margin  style(tab)
          legend label varlabels(_cons Constant)
          posthead("") prefoot("") postfoot("")
          ;
      #delimit cr


*Punishment of free riding
intreg punll punul  trust otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if negdev, cluster(mgroupid)
estimates store PNND_trust
intreg punll punul   kgdppercap  otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if negdev, cluster(mgroupid)
estimates store PNND_kgdppercap
intreg punll punul  democracy otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if negdev, cluster(mgroupid)
estimates store PNND_democracy
intreg punll punul  pdi otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if negdev, cluster(mgroupid)
estimates store PNND_pdi
intreg punll punul  idv otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if negdev, cluster(mgroupid)
estimates store PNND_idv
intreg punll punul  mas otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if negdev, cluster(mgroupid)
estimates store PNND_mas
intreg punll punul  uai otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if negdev, cluster(mgroupid)
estimates store PNND_uai
intreg punll punul   trad_sec otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if negdev, cluster(mgroupid)
estimates store PNND_trad_sec
intreg punll punul   surv_self otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if negdev, cluster(mgroupid)
estimates store PNND_surv_self
/*    #delimit ;
      estout
          PNND_trust PNND_kgdppercap PNND_democracy PNND_pdi PNND_idv PNND_mas PNND_uai PNND_trad_sec PNND_surv_self
          using output\s10fr.out, replace
          cells(b(star fmt(%9.3f))  se(par ))
          stats(sigma p ll  N, fmt(%9.3f %9.3f %9.3g %9.0g) )
          order(trust kgdppercap democracy pdi idv mas uai trad_sec surv_self)
          indicate("Controls" =  otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown , labels(yes no))
          starlevels(* 0.1 ** 0.05 *** 0.01)
          varwidth(16) modelwidth(12)
          margin  style(tab)
          legend label varlabels(_cons Constant)
          posthead("") prefoot("") postfoot("")
          ;
      #delimit cr
*/



log close
