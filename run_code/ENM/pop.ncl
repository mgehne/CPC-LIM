load "/home/ycheng/Scripts/loadFiles.ncl"
load "./PrnOscPat_driver.ncl"    ; creates the POPs from EOFs
load "./pltEofPop.ncl"           ; plot EOF/POP
load "./pltClim.ncl"             ; plot climatology


inf_eof = addfile("../colIrr/1/EOF_colIrr.nc","r")
inf_eof_ts = addfile("../colIrr/pc.all.nc","r")

eof = inf_eof->eof_colIrr(0:22,:,:)
varexp = inf_eof->varexp(0:22)

eof_ts = inf_eof_ts->pc(0:22,:)

eof!1= "lat"
eof!2= "lon"
eof@pcvar = varexp
printVarSummary(eof)
printVarSummary(eof_ts)

kPOP = 1
pop_results  = PrnOscPat_driver(eof, eof_ts, kPOP)   ; returns a list 
; print(pop_results)                                   ; containing variables
 
; ; =================================================================
; ; For convenience & clarity extract each of the two elements 
; ;     and place them into separate variables reather than acces
; ;     via 'list' syntax.
; ; =================================================================
           
pop_ts    = pop_results[0]     ; POP time series  (nEOF,time)
printVarSummary(pop_ts)
;print("pop_ts: min="+min(pop_ts)+"   max="+max(pop_ts))

pop_pat   = pop_results[1]     ; POP patterns     (nEOF,lat,lon)
printVarSummary(pop_pat)
pop_pat!1= "lat"
pop_pat!2= "lon"
printVarSummary(pop_pat)

delete(pop_results)            ; no longer needed

; ===========================================================
; Miscellaneous for file creation and plots
; ===========================================================
sfx    = "colIrr"
froot  = "" 

yyyymm = cd_calendar(eof_ts&time,-2)/100  
yrfrac = yyyymm_to_yyyyfrac(yyyymm, 0.0)   

yyyymm!0 = "time"
yrfrac!0 = "time"
yyyymm&time = eof_ts&time
yrfrac&time = eof_ts&time

; ===========================================================
; NETCDF: write POP data to file: Use simple method of writing netCDF
; ===========================================================

pathNC = sfx+".nc" 
system("/bin/rm -f " + pathNC)    ; remove file if exists
fout = addfile(pathNC,"c")
setfileoption(fout,"DefineMode",True)

; create global attributes
fileAtt		= True
fileAtt@creation_date = systemfunc("date")
fileattdef(fout,fileAtt)

fout->yyyymm      = yyyymm    
fout->yrfrac      = yrfrac    
fout->POP_TS      = pop_ts
fout->POP_PATTERN = pop_pat

fout->EOF_TS      = eof_ts
fout->EOF         = eof

;============================================================
; PLOTS
;============================================================
pltEOF = True
pltType = "png"
pltPOP = True
pltCLIM = True
if (pltEOF) then
pltEofPop(eof_ts, eof, yrfrac, froot, pltType, "EOF", "BlueYellowRed")
end if    ; pltEOF

if (pltPOP) then
pltEofPop(pop_ts, pop_pat, yrfrac, froot, pltType, "POP", "BlueYellowRed")
end if    ; pltPOP

; if (pltCLIM) then
;   pltClim(clim, froot, pltType, "CLIM", "BlAqGrYeOrReVi200")
; end if    ; pltCLIM


