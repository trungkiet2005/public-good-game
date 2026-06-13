*STATA code for ARTICLE Herrmann, Thoeni, Gaechter: Antisocial Punishment Across Societies. Science 319, 7 March 2008, 1362-1367, DOI: 10.1126/science.1153808

*use HerrmannThoeniGaechterDATA.dta

***********************************************************
*EXPERIMENTS
*num of subjects
preserve
 collapse p if data, by(subjectid)
 count
restore


***********************************************************
*RESULTS

*____________________________________________________
*FIGURE 1
local nod //nodraw
preserve
    local lpunnegdev : variable label punnegdev
    local lpunnonnegdev : variable label punnonnegdev
    egen freqpun=mean(dpun) if data, by(city negdev)
    collapse punishment freqpun negdev if data , by(devclass city)
    gen pun2=punishment
    replace pun2=-1*pun2 if devclass<0
    gen order=devclass
    replace order=-1 if devclass==-20
    replace order=-2 if devclass==-10
    decode city, gen(city1)
    gen label=city1
    encode label, gen(city2)
    egen mpun=sum(punishment) if devclass>=0, by(city)
    gsort negdev devclass -mpun
    replace mpun=. if devclass!=0
    gen ycoord=6.36*_n-3.9 if mpu!=.  //set y coordinate of bar labels
    egen barlabh=sum(punishment) , by(city negdev)
    replace barlabh=-barlabh-.2 if negdev  //set x-coordinate of bar labels
    replace barlabh=barlabh+.2 if !negdev
    levelsof city, local(places)
    display "`places'"
    qui foreach i of local places{
      forvalues j=0(1)1{
       sum  freqpun if city==`i' & negdev==`j'
       local l`i'`j'=round(`r(mean)', .01)
       sum  barlabh if city==`i' & negdev==`j'
       local x`i'`j'=`r(mean)'
       sum ycoord if city==`i'
       local y`i'`j'=`r(mean)'
    }
    }
local labelopt "placement(c) size(small)"
   #delimit ;
    graph hbar pun2, `nod'
    saving(output\fig1, replace)
    graphregion(margin(l = 15 t = 8))
    over (devclass, sort(order)) over (city2, label(labsize(small)) sort(mpun)) asyvars stack
    bar(1, blcolor(black) bfcolor(0 85 0))
    bar(2, blcolor(black) bfcolor(85 255 0))
    bar(3, blcolor(black) bfcolor(255 255 170))
    bar(4, blcolor(black) bfcolor(255 85 0))
    bar(5, blcolor(black) bfcolor(128 0 0))
    legend(title("Deviation from" "punisher's contrib.", size(small))
    keygap(*.25) size(small) rowgap(*.7) position(1) ring(0) cols(1) symxsize(*.4) symysize(*.5) region(lstyle(none) fcolor(none) ))
    aspectratio(1)
    ylabel(0 " ", grid glcolor(black))
    ymlabel(-5 "5" -4 "4" -3 "3" -2 "2" -1 "1" 0 "0" 1 "1" 2 "2" 3 "3" 4 "4", /*grid glwidth(vthin) glcolor(gs12)*/ labsize(small))
    ytitle("Mean punishment expenditures")
    text(-2.8 110 "`lpunnegdev'" "(negative deviations)", placement(c) size(small))
    text(2.2 110 "`lpunnonnegdev'" "(non-negative deviations)", placement(c) size(small))
    ;
   #delimit cr
restore

*____________________________________________________
*Punishment estimates
 intreg punll punul otherscontribution senderscontribution avcother2 period finalperiod dcity51-dcity67 female ageu21 singlechild urbanbackground middleclass membership numknown  if !negdev, nocons cluster(mgroupid) nolog
   test  dcity51= dcity52= dcity53= dcity55= /*dcity56=*/ dcity57= dcity58= dcity59= dcity60= dcity61= dcity62= dcity63= dcity64= dcity65= dcity66 = dcity67
 intreg punll punul otherscontribution senderscontribution avcother2 period finalperiod dcity51-dcity67 female ageu21 singlechild urbanbackground middleclass membership numknown   if negdev, nocons cluster(mgroupid) nolog
   test  dcity51= dcity52= dcity53= dcity55= /*dcity56=*/ dcity57= dcity58= dcity59= dcity60= dcity61= dcity62= dcity63= dcity64= dcity65= dcity66 = dcity67

*____________________________________________________
*Kwallis p
preserve
  collapse (mean) c city , by(p mgroupid)
  kwallis c if p==1, by(city)
restore

*cooperation in p
table cityorder if p, c(mean c)

*____________________________________________________
*FIGURE 2
local nod //nodraw
*Panel A
forvalues i=1(1)1{
 preserve
  egen meanc=mean(c) if p==`i', by(period)
  collapse (mean)c  meanc  if p==`i', by(period city)
   egen meanccit=mean(c), by(city)
   local xlabeldot=10.8    //x-axis of label dots (max one digit after comma)
   local xlabel=11       //x-axis of label  (max one digit after comma)
   qui inspect city
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
   replace meanccit=meanccit[_n-1] if c==.
   sort period meanccit
   gen rank= mod(_n-1,`nocities')+1  //ranking of the cities according to mean ratio
   local ymin = 1 // range of labels along y-axis
   local ymax = 19
   gen ylabel = (rank - 1)*(`ymax' - `ymin')/(`nocities'-1)+`ymin'
   replace c = ylabel if round(period,.1)== round(`xlabeldot',.1)
   sort city period
   label define city 99 Mean, modify
   decode city, generate(city1)  //relabel cityvariable (add average contribution)
   replace meanccit=round(meanccit,.1)
   tostring meanccit, generate (meanccitstr) force usedisplayformat
   replace city1="Dniprop." if city1=="Dnipropetrovs'k"
   gen city2=city1 + " (" + meanccitstr + ")"
   encode city2, generate (city3)
   drop  city2 city1 meanccitstr
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
     replace c=ylabel if city<0
     replace city=99 if city<0
drop if city==99 //***************************eliminate av
   local panel B
   if `i'==1{
   local panel A
   }
   local sym1 "msymbol(O) msize(*1.5) mlcolor(black)"
   local sym2 "msymbol(s) msize(*1.8) mlcolor(black)"
   local sym3 "msymbol(d) msize(*1.5) mlcolor(black)"
   local sym4 "msymbol(t) msize(*1.8) mlcolor(black)"
     #delimit ;
      tw
    (scatter c period, connect(l) clwidth(medthin) cmissing(n) msymbol(i))
    (scatter c period if city==5, `sym1'  mfcolor("0 128 0"))
    (scatter c period if city==8,  `sym2' mfcolor("0 255 0"))
    (scatter c period if city==7,  `sym3' mfcolor("128 255 128"))
    (scatter c period if city==1, `sym4' mfcolor("128 0 0"))
    (scatter c period if city==9, `sym1' mfcolor("255 0 0"))
    (scatter c period if city==14, `sym2' mfcolor("255 128 128"))
    (scatter c period if city==15, `sym3' mfcolor("0 0 128"))
    (scatter c period if city==17, `sym4' mfcolor("0 0 255"))
    (scatter c period if city==13, `sym1' mfcolor("128 128 255"))
    (scatter c period if city==3, `sym2' mfcolor("128 128 0"))
    (scatter c period if city==2, `sym3' mfcolor("128 64 0"))
    (scatter c period if city==10, `sym4' mfcolor("255 128 0"))
    (scatter c period if city==6, `sym1' mfcolor("255 255 0"))
    (scatter c period if city==12, `sym2' mfcolor("128 0 128"))
    (scatter c period if city==11, `sym3' mfcolor("128 0 64"))
    (scatter c period if city==16, `sym4' mfcolor("128 128 128"))
    /*(scatter c period if city==99, connect(l) clwidth(medthick) cmissing(n) msymbol(i) lpattern(solid))*/
    (scatter ylabel period if round(period,.1)==round(`xlabel',.1), msymbol(i) mlabel(city3)  mlabsize(small))
     , `nod'
     text(-1.8 5 "Period" , size( medium))
     legend(off) xtitle("") ytitle("Contribution" , size( medium))
     xscale(range(.8 14.3)) xlabel(1(1)10, labsize(medsmall)) ylabel(0(2)20, labsize(medsmall))
     /*text(21.5 -.5 "`panel'", size(vlarge))*/
     graphregion(margin(t = 8))
     plotregion(margin(zero))
     saving(output\fig2A, replace)
     ;
     #delimit cr
  restore
  }
*

*Panel B
preserve
 local lpunnonnegdev : variable label punnonnegdev
 collapse (mean) punnonnegdev c if p==1, by(city)
  gen a=0 in 1
  gen b=0 in 1
  replace a=100 in 2
  replace b=100 in 2
   capture drop pos
   gen pos=3
   replace pos=9 if inlist(city,6,15)
   replace pos=1 if inlist(city,5)
   replace pos=2 if inlist(city,9)
   replace pos=5 if inlist(city,13,10)
   local mue=char(181)
  spearman punnonnegdev c
  local rho=round(`r(rho)',.01)
  local p=round(`r(p)',.001)
  local sprk "n = `r(N)', rho = `rho', p = `p'"
  local sprk "n = `r(N)', rho = -.90, p = .000"
  #delimit ;
   tw (scatter c punnonnegdev,
   mlabel(city) mlabvposition(pos) msymbol(O) mcolor(black) msize(medsmall))
   (scatter a b, xaxis(2) yaxis(2) msymbol(none))
   (lfit c punnonnegdev , lcolor(black)  lwidth(vthin) ),
   xscale(range(0 1.25))
   xlabel(0(.2)1.2, labsize(medsmall))
   ylabel(0(2)20, labsize(medsmall))
   legend(off)
   graphregion(margin(t = 5))
   ytitle("Mean contribution (all periods)", size( medium))
   xtitle("")
   text(-2 .61 "Mean antisocial punishment", size( medium))
    /*text(21.5 -.13 "B", size(vlarge))*/
    /*text(111 -10 "B", size(vlarge) xaxis(2) yaxis(2) placement(se))*/
    text(95 98 "`sprk'", size(medsmall) xaxis(2) yaxis(2) placement(w))
     graphregion(margin(t = 8))
     plotregion(margin(zero))
     aspectratio(1)
     xlabel("", axis(2)) xtitle("", axis(2))
     ylabel("", axis(2)) ytitle("", axis(2))
   `nod'
   saving(output\fig2B, replace)
  ;
  #delimit cr
restore

*  graph combine "output\fig2A" "output\fig2B", ycommon xsize(6) ysize(3) saving(output\fig2, replace)


*Profits
table cityorder if p, c(mean profit)


*____________________________________________________
*Reaction to punishment for below average contributions

qui do reaction.do
preserve
  collapse (mean) reactionFR reactionFRmin punnonnegdev if data, by(city)
  spearman reactionFR punnonnegdev
  spearman reactionFRmin punnonnegdev  //documented in SOM
restore



*____________________________________________________
*TABLE 1
preserve
  drop if !p | !data
  foreach v of var * {
      local l`v' : variable label `v'
      if `"`l`v''"' == "" {
      local l`v' "`v'"
      }
     }
  collapse (mean) c c1 c2to10 punnegdev punnonnegdev mgroupid cityorder dcity* , by(groupid)
  drop dcity51
  foreach v of var * {
      label var `v' "`l`v''"
      }
  local opt "r cluster(mgroupid)"
  reg c2to10 c1 punnegdev punnonnegdev, `opt'
  estimates store e1
  reg c2to10 c1 punnegdev punnonnegdev dcity* , `opt'
  estimates store e2
      #delimit ;
      estout e1 e2
          using output\table1.out, replace
          cells(b(star fmt(%9.3f))  se(par ))
          stats(r2_a F p N, fmt(%9.2f %9.1f %9.3f %9.0g))
          indicate("Subject pool dummies" = dcity* , labels(yes no))
          starlevels(* 0.1 ** 0.05 *** 0.01)
          varwidth(25) modelwidth(12)
          margin  style(tab)
          legend label varlabels(_cons Constant)
          posthead("") prefoot("") postfoot("")
          ;
      #delimit cr
restore


*____________________________________________________
*Kwallis n
preserve
  collapse (mean) c city , by(p mgroupid)
  kwallis c if p==0, by(city)
restore

table cityorder if p==0, c(mean c)


*____________________________________________________
*FIGURE 3
local nod //nodraw
forvalues i=0(1)0{
 preserve
  egen meanc=mean(c) if p==`i', by(period)
  collapse (mean)c  meanc  if p==`i', by(period city)
   egen meanccit=mean(c), by(city)
   local xlabeldot=10.8    //x-axis of label dots (max one digit after comma)
   local xlabel=11       //x-axis of label  (max one digit after comma)
   qui inspect city
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
   replace meanccit=meanccit[_n-1] if c==.
   sort period meanccit
   gen rank= mod(_n-1,`nocities')+1  //ranking of the cities according to mean ratio
   local ymin = 1 // range of labels along y-axis
   local ymax = 19
   gen ylabel = (rank - 1)*(`ymax' - `ymin')/(`nocities'-1)+`ymin'
   replace c = ylabel if round(period,.1)== round(`xlabeldot',.1)
   sort city period
   label define city 99 Mean, modify
   decode city, generate(city1)  //relabel cityvariable (add average contribution)
   replace meanccit=round(meanccit,.1)
   tostring meanccit, generate (meanccitstr) force usedisplayformat
   replace city1="Dniprop." if city1=="Dnipropetrovs'k"
   gen city2=city1 + " (" + meanccitstr + ")"
   encode city2, generate (city3)
   drop  city2 city1 meanccitstr
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
     replace c=ylabel if city<0
     replace city=99 if city<0
drop if city==99 //***************************eliminate av
   local panel B
   if `i'==1{
   local panel A
   }
   local sym1 "msymbol(O) msize(*1.5) mlcolor(black)"
   local sym2 "msymbol(s) msize(*1.8) mlcolor(black)"
   local sym3 "msymbol(d) msize(*1.5) mlcolor(black)"
   local sym4 "msymbol(t) msize(*1.8) mlcolor(black)"
     #delimit ;
      tw
    (scatter c period, connect(l) clwidth(medthin) cmissing(n) msymbol(i))
    (scatter c period if city==5, `sym1'  mfcolor("0 128 0"))
    (scatter c period if city==8,  `sym2' mfcolor("0 255 0"))
    (scatter c period if city==7,  `sym3' mfcolor("128 255 128"))
    (scatter c period if city==1, `sym4' mfcolor("128 0 0"))
    (scatter c period if city==9, `sym1' mfcolor("255 0 0"))
    (scatter c period if city==14, `sym2' mfcolor("255 128 128"))
    (scatter c period if city==15, `sym3' mfcolor("0 0 128"))
    (scatter c period if city==17, `sym4' mfcolor("0 0 255"))
    (scatter c period if city==13, `sym1' mfcolor("128 128 255"))
    (scatter c period if city==3, `sym2' mfcolor("128 128 0"))
    (scatter c period if city==2, `sym3' mfcolor("128 64 0"))
    (scatter c period if city==10, `sym4' mfcolor("255 128 0"))
    (scatter c period if city==6, `sym1' mfcolor("255 255 0"))
    (scatter c period if city==12, `sym2' mfcolor("128 0 128"))
    (scatter c period if city==11, `sym3' mfcolor("128 0 64"))
    (scatter c period if city==16, `sym4' mfcolor("128 128 128"))
    (scatter ylabel period if round(period,.1)==round(`xlabel',.1), msymbol(i) mlabel(city3)  mlabsize(small))
     , `nod'
     legend(off) xtitle("") ytitle("Contribution")
     text(-1.6 5 "Period" , size( medium))
     xscale(range(.8 14.3)) xlabel(1(1)10, labsize(medsmall)) ylabel(0(2)20, labsize(medsmall))
     plotregion(margin(zero))
     aspectratio(.8)
     saving(output\fig3, replace)
     ;
     #delimit cr
  restore
  }






*____________________________________________________
*Increase of contribution from N to P
preserve
  gen cP=c if p==1
  gen cN=c if p==0
  gen c1P=c1 if p==1
  collapse (mean) punnonnegdev cP cN c1P if data, by(city)
  gen delta=(cP-cN)/cN
  spearman delta punnonnegdev
  spearman c1P punnonnegdev
restore




*____________________________________________________
*TABLE 2
*Table
intreg punll punul  civic otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if negdev, cluster(mgroupid) nolog
estimates store PND_civic
intreg punll punul  ruleoflaw otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if negdev, cluster(mgroupid) nolog
estimates store PND_ruleoflaw
intreg punll punul  ruleoflaw  civic otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if negdev, cluster(mgroupid) nolog
estimates store PND_all

intreg punll punul  civic otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if !negdev, cluster(mgroupid) nolog
estimates store PNND_civic
intreg punll punul  ruleoflaw otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if !negdev, cluster(mgroupid) nolog
estimates store PNND_ruleoflaw
intreg punll punul  ruleoflaw  civic otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if !negdev, cluster(mgroupid) nolog
estimates store PNND_all

     #delimit ;
      estout
          PND_civic PND_ruleoflaw PND_all PNND_civic PNND_ruleoflaw PNND_all
          using output\table2.out, replace
          cells(b(star fmt(%9.3f))  se(par ))
          stats(sigma p ll  N, fmt(%9.3f %9.3f %9.3g %9.0g) )
          order(civic ruleoflaw )
          indicate("Controls" =  otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown , labels(yes no))
          starlevels(* 0.1 ** 0.05 *** 0.01)
          varwidth(16) modelwidth(12)
          margin  style(tab)
          legend label varlabels(_cons Constant)
          posthead("") prefoot("") postfoot("")
          ;
      #delimit cr



***********************************************************
*DISCUSSION
intreg punll punul  idv otherscontribution senderscontribution  avcother2 period finalperiod female ageu21 singlechild urbanbackground middleclass membership numknown   if !negdev, cluster(mgroupid) nolog



log close
