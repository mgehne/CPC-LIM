#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:59:42 2021

@author: slillo
"""

import os
import calendar
from datetime import datetime as dt, timedelta
from dateutil.relativedelta import relativedelta


# =============================================================================
# WRITE DAY PAGE
# =============================================================================

def write_day_html(T_INIT,path):
    
    html_path = '/'+'/'.join(path.split('/')[3:])

    YYYYMM = f'{T_INIT:%Y%m}'
    YYYYMMDD = f'{T_INIT:%Y%m%d}'
    doy365 = int(f'{T_INIT:%j}')%365+1
    
    text = '''
    <!DOCTYPE html public "-//w3c//dtd html 4.0 transitional//en">
    <HTML>
    <HEAD>
      <META NAME="GENERATOR" CONTENT="Adobe PageMill 3.0 Mac">
      <META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=iso-8859-1">
      <META NAME="GENERATOR" CONTENT="Mozilla/4.51 [en] (X11; U; SunOS 5.5 sun4m) [Netscape]">
      <TITLE>LIM Week 3-4 Forecast</TITLE>
    </HEAD>
    <BODY BGCOLOR="#f5fffa">
    <STYLE>A {font-family:helvetica; color:#0000ff; text-decoration: none;} </STYLE>
    
    <P>&nbsp;</P>
    
    <H2><CENTER><FONT COLOR="#990000" SIZE="+2">LIM Week 3-4 Forecast</FONT></CENTER></H2>
    <H2><CENTER><FONT COLOR="BLACK" SIZE="+1">'''
    
    if os.path.isdir(f'{path}{T_INIT-timedelta(days=1):%Y%m}/{T_INIT-timedelta(days=1):%Y%m%d}'):
        text += f'''<a href="{html_path}{T_INIT-timedelta(days=1):%Y%m}/{T_INIT-timedelta(days=1):%Y%m%d}.html" style="color:black;"><<&nbsp&nbsp</a>'''
    else:
        text += '''<span style="padding-right:15px"></span>'''
    text += f'''Initialization: {T_INIT:%-d} <a href="{html_path}web_{YYYYMM}.html" style="color:black;">{T_INIT:%B}</a> {T_INIT:%Y}'''
    if os.path.isdir(f'{path}{T_INIT+timedelta(days=1):%Y%m}/{T_INIT+timedelta(days=1):%Y%m%d}'):
        text += f'''<a href="{html_path}{T_INIT+timedelta(days=1):%Y%m}/{T_INIT+timedelta(days=1):%Y%m%d}.html" style="color:black;">&nbsp&nbsp>></a>'''
    else:
        text += '''<span style="padding-right:15px"></span>'''
    
    text+='''</FONT></CENTER></H2>
    
    <center>
    <P><CENTER><B><FONT COLOR="#000000" SIZE="+1">Spatial Map</FONT></B>
    <center><table BORDER=3 CELLSPACING=2 CELLPADDING=0 HEIGHT="34" >
    <tr>
    <td WIDTH="12%"><center>Variable</center></td>
    <td WIDTH="30%"><center>Anomaly</center></td>
    <td WIDTH="30%"><center>Probability</center></td>
    </tr>
    '''
    
    vardict = {'T2m':'T2m','H500':'H500','SLP':'SLP','colIrr':'Tropical Heating'}
    for varlabel,varname in vardict.items():
        text += f'''
        <tr>
        <td WIDTH="12%"><center>{varname}</center></td>
        <td WIDTH="30%" valign="top"><center>
         <table border="0"><tr>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt000.png">Init</a>,&nbsp
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt014.png">Week 2</a>,&nbsp
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt021.png">Week 3</a>,&nbsp
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt028.png">Week 4</a>,&nbsp
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt021-028.png">Weeks 3-4</a>,&nbsp
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}.gif">Loop</a>
          </tr>
          '''
        if os.path.isfile(f'{path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt021-028_obs.png') and varlabel is not 'T2m':
            text += f'''
          <tr>
          <td>&nbsp
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt014_obs.png">obs</a>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt021_obs.png">obs</a>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt028_obs.png">obs</a>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt021-028_obs.png">obs</a>
          </tr>
            '''
        if os.path.isfile(f'{path}{YYYYMM}/{YYYYMMDD}/CPCverif_anom.{T_INIT+timedelta(days=28):%Y%m%d}.014.png') and varlabel is 'T2m':
            text += f'''
          <tr>
          <td>&nbsp
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/CPCverif_anom.{T_INIT+timedelta(days=14):%Y%m%d}.007.png">obs</a>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/CPCverif_anom.{T_INIT+timedelta(days=21):%Y%m%d}.007.png">obs</a>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/CPCverif_anom.{T_INIT+timedelta(days=28):%Y%m%d}.007.png">obs</a>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/CPCverif_anom.{T_INIT+timedelta(days=28):%Y%m%d}.014.png">obs</a>
          </tr>
            '''
            
        text += f'''
         </table>
        </center></td>
        <td WIDTH="30%" valign="top"><center>
         <table border="0"><tr>
          <td><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}-PROB_lt014.png">Week 2</a>,&nbsp
          <td><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}-PROB_lt021.png">Week 3</a>,&nbsp
          <td><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}-PROB_lt028.png">Week 4</a>,&nbsp
          <td><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}-PROB_lt021-028.png">Weeks 3-4</a>
         </tr>
         '''
        if os.path.isfile(f'{path}{YYYYMM}/{YYYYMMDD}/T2m_lt021-028_hitmiss_55.png') and varlabel is 'T2m':
            text += f'''
          <tr>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/CPCverif_cat.{T_INIT+timedelta(days=14):%Y%m%d}.007.png">obs</a>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/CPCverif_cat.{T_INIT+timedelta(days=21):%Y%m%d}.007.png">obs</a>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/CPCverif_cat.{T_INIT+timedelta(days=28):%Y%m%d}.007.png">obs</a>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/CPCverif_cat.{T_INIT+timedelta(days=28):%Y%m%d}.014.png">obs</a>
          </tr>
          <tr>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt014_hitmiss_55.png">hit/miss</a>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt021_hitmiss_55.png">hit/miss</a>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt028_hitmiss_55.png">hit/miss</a>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt021-028_hitmiss_55.png">hit/miss</a>
          </tr>
          <tr>
          <td>&nbsp
          <td>&nbsp
          <td>&nbsp
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_HSS_timeseries.png">HSS</a> | <a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_RPSS_timeseries.png">RPSS</a>
          </tr>
            '''
        text += '''
         </table>
        </center></td>
        </tr>
        '''
        
    weekday = T_INIT.weekday()
    dayoffset = (4-weekday)%7
    T_INIT_CPC = T_INIT + timedelta(days=dayoffset)
 
    text += f'''
    </table>
    
    <br>
    <br>
    
    <center>
    <P><CENTER><B><FONT COLOR="#000000" SIZE="+1">Upcoming CPC Weeks 3-4 Valid: {T_INIT_CPC+timedelta(days=15):%-d %b} - {T_INIT_CPC+timedelta(days=28):%-d %b %Y}</FONT></B>
    <center><table BORDER=3 CELLSPACING=2 CELLPADDING=0 HEIGHT="34" >
    <tr>
    <td WIDTH="12%"><center>Variable</center></td>
    <td WIDTH="30%"><center>Anomaly</center></td>
    <td WIDTH="30%"><center>Probability</center></td>
    </tr>
    '''
    
    vardict = {'T2m':'T2m','H500':'H500','SLP':'SLP','colIrr':'Tropical Heating'}
    for varlabel,varname in vardict.items():
        text += f'''
        <tr>
        <td WIDTH="12%"><center>{varname}</center></td>
        <td WIDTH="30%" valign="top"><center>
         <table border="0"><tr>
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt{21+dayoffset:03}.png">Week 3</a>,&nbsp
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt{28+dayoffset:03}.png">Week 4</a>,&nbsp
          <td><center><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}_lt{21+dayoffset:03}-{28+dayoffset:03}.png">Weeks 3-4</a>
          </tr>
          '''

        text += f'''
         </table>
        </center></td>
        <td WIDTH="30%" valign="top"><center>
         <table border="0"><tr>
          <td><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}-PROB_lt{21+dayoffset:03}.png">Week 3</a>,&nbsp
          <td><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}-PROB_lt{28+dayoffset:03}.png">Week 4</a>,&nbsp
          <td><a href="{html_path}{YYYYMM}/{YYYYMMDD}/{varlabel}-PROB_lt{21+dayoffset:03}-{28+dayoffset:03}.png">Weeks 3-4</a>
         </tr>
         '''

        text += '''
         </table>
        </center></td>
        </tr>
        ''' 
    
    text += f'''
    </table>
    
    <br>
    <br>
    
    <P><CENTER><B><FONT COLOR="#000000" SIZE="+1">Hovmöller Diagram</FONT></B>
    <center><table BORDER=3 CELLSPACING=2 CELLPADDING=0 HEIGHT="34" >
    <tr>
    <td WIDTH="30%"><center>H500</center></td>
    <td WIDTH="30%"><center>Tropical Heating</center></td>
    </tr>
    <tr>
    <td WIDTH="30%"><center>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/HOV_500_40N60N.png">40<sup>o</sup>N - 60<sup>o</sup>N</a><br>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/HOV_500_30N50N.png">30<sup>o</sup>N - 50<sup>o</sup>N</a><br>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/HOV_500_20N40N.png">20<sup>o</sup>N - 40<sup>o</sup>N</a>

    </center></td>
    <td WIDTH="30%"><center>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/HOV_trop_0N15N.png">0<sup>o</sup>N - 15<sup>o</sup>N</a><br>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/HOV_trop_7.5S7.5N.png">7.5<sup>o</sup>S - 7.5<sup>o</sup>N</a><br>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/HOV_trop_15S0N.png">15<sup>o</sup>S - 0<sup>o</sup>N</a>

    </center></td>
    </tr>
    </table>
    
    <br>
    <br>
    
    <P><CENTER><B><FONT COLOR="#000000" SIZE="+1">Time Series: Teleconnection Pattern</FONT></B>
    <center><table BORDER=3 CELLSPACING=2 CELLPADDING=0 HEIGHT="34" >
    <tr>
    <td WIDTH="30%"><center>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/teleconnection_EA.png">E. Atlantic</a>
     <br><a href="{html_path}loadingpatterns/ea_{doy365:03}.png">(map)</a>
    </center></td>
    <td WIDTH="30%"><center>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/teleconnection_NAO.png">NAO</a>
     <br><a href="{html_path}loadingpatterns/nao_{doy365:03}.png">(map)</a>
    </center></td>
    <td WIDTH="30%"><center>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/teleconnection_SCAND.png">Scandinavia</a>
     <br><a href="{html_path}loadingpatterns/scand_{doy365:03}.png">(map)</a>
    </tr>
    <tr>
    <td WIDTH="30%"><center>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/teleconnection_EAWR.png">E. Atlantic/W. Russia</a>
     <br><a href="{html_path}loadingpatterns/eawr_{doy365:03}.png">(map)</a>
    </center></td>
    <td WIDTH="30%"><center>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/teleconnection_PNA.png">PNA</a>
     <br><a href="{html_path}loadingpatterns/pna_{doy365:03}.png">(map)</a>
    </center></td>
    <td WIDTH="30%"><center>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/teleconnection_TNH.png">TNH</a>
     <br><a href="{html_path}loadingpatterns/tnh_{doy365:03}.png">(map)</a>
    </center></td>
    </tr>
    <tr>
    <td WIDTH="30%"><center>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/teleconnection_EPNP.png">E. Pacific/N. Pacific</a>
     <br><a href="{html_path}loadingpatterns/epnp_{doy365:03}.png">(map)</a>
    </center></td>
    <td WIDTH="30%"><center>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/teleconnection_POLEUR.png">Polar/Eurasian</a>
     <br><a href="{html_path}loadingpatterns/poleur_{doy365:03}.png">(map)</a>
    </center></td>
    <td WIDTH="30%"><center>
     <a href="{html_path}{YYYYMM}/{YYYYMMDD}/teleconnection_WP.png">W. Pacific</a>
     <br><a href="{html_path}loadingpatterns/wp_{doy365:03}.png">(map)</a>
    </center></td>
    </tr>
    </table>
    
    
    
    </center>
    
    </BODY>
    </HTML>
    '''

    filename = f"{path}{YYYYMM}/{YYYYMMDD}.html"
    file = open(filename,"w")
    file.write(text)
    file.close()    
    return filename

# =============================================================================
# WRITE MONTH PAGE
# =============================================================================

def write_month_html(T_INIT,path):

    html_path = '/'+'/'.join(path.split('/')[3:])    

    text = '''
    <head>
    
    <BODY BGCOLOR="#f5fffa">
    <STYLE>A {font-family:helvetica; color:#0000ff; text-decoration: none;} </STYLE>
    
    <div class=WordSection1>
    
    <p class=MsoNormal align=center style='text-align:center'><style='mso-bidi-font-weight:
    normal'><span style='font-family:helvetica;font-size:20.0pt;line-height:115%;color:#0070C0'>
    LIM Week 3-4 Forecast
    <o:p></o:p></span></p>
    <p class=MsoNormal align=center style='text-align:center'><style='mso-bidi-font-weight:
    normal'><span style='font-family:helvetica;font-size:18.0pt;line-height:115%;color:#0070C0'>
    '''
    
    if os.path.isdir(f'{path}{T_INIT-relativedelta(months=1):%Y%m}'):
        text += f'''<a href="{html_path}web_{T_INIT-relativedelta(months=1):%Y%m}.html" style="color:#0070C0;"><<&nbsp&nbsp</a>'''
    else:
        text += '''<span style="padding-right:15px"></span>'''
    text += f'''
    {T_INIT:%B %Y}
    '''
    if os.path.isdir(f'{path}{T_INIT+relativedelta(months=1):%Y%m}'):
        text += f'''<a href="{html_path}web_{T_INIT+relativedelta(months=1):%Y%m}.html" style="color:#0070C0;">&nbsp&nbsp>></a>'''
    else:
        text += '''<span style="padding-right:15px">'''
    
    
    text += '''
    </span><o:p></o:p></span></p>
    <center>
    <table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0
     style='border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
     mso-yfti-tbllook:1184;mso-padding-alt:0in 5.4pt 0in 5.4pt'>
     <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes'>
      <td width=91 valign=top style='width:.95in;border:solid windowtext 1.0pt;
      mso-border-alt:solid windowtext .5pt;padding:0in 5.4pt 0in 5.4pt'>
      <p class=MsoNormal align=center style='margin-bottom:0in;margin-bottom:.0001pt;
      text-align:center;line-height:normal'><style='mso-bidi-font-weight:normal'><span
      style='font-family:helvetica;font-size:18.0pt'>Sun<o:p></o:p></span></p>
      </td>
      <td width=91 valign=top style='width:.95in;border:solid windowtext 1.0pt;
      border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
      solid windowtext .5pt;padding:0in 5.4pt 0in 5.4pt'>
      <p class=MsoNormal align=center style='margin-bottom:0in;margin-bottom:.0001pt;
      text-align:center;line-height:normal'><style='mso-bidi-font-weight:normal'><span
      style='font-family:helvetica;font-size:18.0pt'>Mon<o:p></o:p></span></p>
      </td>
      <td width=91 valign=top style='width:.95in;border:solid windowtext 1.0pt;
      border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
      solid windowtext .5pt;padding:0in 5.4pt 0in 5.4pt'>
      <p class=MsoNormal align=center style='margin-bottom:0in;margin-bottom:.0001pt;
      text-align:center;line-height:normal'><style='mso-bidi-font-weight:normal'><span
      style='font-family:helvetica;font-size:18.0pt'>Tue<o:p></o:p></span></p>
      </td>
      <td width=91 valign=top style='width:.95in;border:solid windowtext 1.0pt;
      border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
      solid windowtext .5pt;padding:0in 5.4pt 0in 5.4pt'>
      <p class=MsoNormal align=center style='margin-bottom:0in;margin-bottom:.0001pt;
      text-align:center;line-height:normal'><style='mso-bidi-font-weight:normal'><span
      style='font-family:helvetica;font-size:18.0pt'>Wed<o:p></o:p></span></p>
      </td>
      <td width=91 valign=top style='width:.95in;border:solid windowtext 1.0pt;
      border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
      solid windowtext .5pt;padding:0in 5.4pt 0in 5.4pt'>
      <p class=MsoNormal align=center style='margin-bottom:0in;margin-bottom:.0001pt;
      text-align:center;line-height:normal'><style='mso-bidi-font-weight:normal'><span
      style='font-family:helvetica;font-size:18.0pt'>Thu<o:p></o:p></span></p>
      </td>
      <td width=91 valign=top style='width:.95in;border:solid windowtext 1.0pt;
      border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
      solid windowtext .5pt;padding:0in 5.4pt 0in 5.4pt'>
      <p class=MsoNormal align=center style='margin-bottom:0in;margin-bottom:.0001pt;
      text-align:center;line-height:normal'><style='mso-bidi-font-weight:normal'><span
      style='font-family:helvetica;font-size:18.0pt'>Fri<o:p></o:p></span></p>
      </td>
      <td width=91 valign=top style='width:.95in;border:solid windowtext 1.0pt;
      border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
      solid windowtext .5pt;padding:0in 5.4pt 0in 5.4pt'>
      <p class=MsoNormal align=center style='margin-bottom:0in;margin-bottom:.0001pt;
      text-align:center;line-height:normal'><style='mso-bidi-font-weight:normal'><span
      style='font-family:helvetica;font-size:18.0pt'>Sat<o:p></o:p></span></p>
      </td>
     </tr>
    '''
    
    YYYYMM = f'{T_INIT:%Y%m}'
    days_in_month = calendar.monthrange(T_INIT.year,T_INIT.month)[1]
    first_dow = (T_INIT.replace(day=1).weekday()+1)%7
    last_dow = (T_INIT.replace(day=days_in_month).weekday()+1)%7
    calendar_days = [0 for i in range(first_dow)] + [i+1 for i in range(days_in_month)] + [0 for i in range(6-last_dow)]
    
    for iDAY,DAY in enumerate(calendar_days):
        
        YYYYMMDD = f'{YYYYMM}{DAY:02}'
        
        try:
            YYYYMMDDdt = dt.strptime(YYYYMMDD,'%Y%m%d')
            hasverif = os.path.isfile(f'{path}{YYYYMM}/{YYYYMMDD}/CPCverif_cat.{YYYYMMDDdt+timedelta(days=28):%Y%m%d}.014.png') or \
                os.path.isfile(f'{path}{YYYYMM}/{YYYYMMDD}/T2m_lt021-028_hitmiss_55.png')
        except:
            hasverif = False
        
        if iDAY%7==0:
             text += f'''
 <tr style='mso-yfti-irow:{int(iDAY//7+1)}'>
  <td width=91 valign=top style='width:.95in;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0in 0pt 0in 0pt'>         
            '''
        else:
            text += '''
  <td width=91 valign=top style='width:.95in;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0pt 0pt 0pt 0pt'>
            '''
        if os.path.isdir(f'{path}{YYYYMM}/{YYYYMMDD}') and hasverif:
            text += f'''  <p class=MsoNormal align=center style='margin-bottom:0in;margin-bottom:.0001pt;
  text-align:center;line-height:normal'><style='mso-bidi-font-weight:normal'>
  <a href="{html_path}{YYYYMM}/{YYYYMMDD}.html" title="Forecast initialized on {YYYYMMDD} with verification" onMouseOver="this.style.backgroundColor='dodgerblue'" onMouseOut="this.style.backgroundColor='lightskyblue'"  style="display:block;background-color:lightskyblue;color:black;">
  <span style='font-family:helvetica;font-size:18.0pt;'>{DAY if DAY>0 else ''} 
  </span></a></p>
  </td>
            '''  
        elif os.path.isdir(f'{path}{YYYYMM}/{YYYYMMDD}'):
            text += f'''  <p class=MsoNormal align=center style='margin-bottom:0in;margin-bottom:.0001pt;
  text-align:center;line-height:normal'><style='mso-bidi-font-weight:normal'>
  <a href="{html_path}{YYYYMM}/{YYYYMMDD}.html" title="Forecast initialized on {YYYYMMDD}" onMouseOver="this.style.backgroundColor='indianred'" onMouseOut="this.style.backgroundColor='lightcoral'" style="display:block;background-color:lightcoral;color:black;">
  <span style='font-family:helvetica;font-size:18.0pt;'>{DAY if DAY>0 else ''} 
  </span></a></p>
  </td>
            '''  
        else:
            text += f'''  <p class=MsoNormal align=center style='margin-bottom:0in;margin-bottom:.0001pt;
  text-align:center;line-height:normal'><style='mso-bidi-font-weight:normal'><span
  style='font-family:helvetica;font-size:18.0pt'>{DAY if DAY>0 else ''} <o:p></o:p></span></p>
  </td>
            '''
        if iDAY%7==6:
            text += ' </tr>'
    
    text += '''
    
    </table>
    
    <br>
    
    </p><br>
    
    <p class=MsoNormal align=center style='text-align:center'><style='mso-bidi-font-weight:
    normal'><span style='font-family:helvetica;font-size:18.0pt;line-height:115%;color:#0070C0'>
    Forecast Archive
    <o:p></o:p></span></p>
    
    <p style="font-family:helvetica; font-size: 20px;">
    '''
    
    latestyear = dt.now().year
    for calyear in range(2017,latestyear+1):
        if calyear == T_INIT.year:
            text += f''' | <b>{calyear}</b> '''
        elif os.path.isfile(f'{path}web_{calyear}{T_INIT.month:02}.html'):
            text += f''' | <a href="{html_path}web_{calyear}{T_INIT.month:02}.html">{calyear}</a> '''
        else:
            text += f''' | {calyear} '''
        if calyear == latestyear: text += '|<br><br>'

    calyear = T_INIT.year
    for calmonth in range(1,13):
        monthname = calendar.month_name[calmonth]

        if YYYYMM == f'{calyear}{calmonth:02}':
            text += f''' | <b>{monthname}</b> '''
        elif os.path.isfile(f'{path}web_{calyear}{calmonth:02}.html'):
            text += f''' | <a href="{html_path}web_{calyear}{calmonth:02}.html">{monthname}</a> '''
        else:
            text += f''' | {monthname} '''
        if calmonth in [7,12]: text += '|<br>'
    
    text += f'''
</p>    


<br><br>
<p>
<i><a href="{html_path}versionnotes.html">Running LIM version 1.2</a></i>
</p>

</center>
    
</div>
    
</body>
    
</html>
    
    '''

    filename = f"{path}web_{YYYYMM}.html"
    file = open(filename,"w")
    file.write(text)
    file.close()    
    return filename


