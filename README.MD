# IV Curve Normalization

### Introduction

In order to compare PV performance consistently across panels, we talk about power output in terms of STC (1000 W/m<sup>2</sup>, 25 C). My hope is that this document will help future solar array leads and members normalize their data. To help with this, I made a script to do the heavy lifting.



### Collecting Data

While PV testers are the standard way to test panel performance, their cost makes them pretty inaccessible to the club (at least at the moment). Luckily for us, we can still collect reasonably accurate data using only a multimeter and a variety of power resistors. I may make a more in depth doc w testing procedures, but you will basically just take voltage and current measurements for the panels over a range of different impedances. Plotting these raw points will yield an IV curve?



### Normalize the Data!

If testing in Madison, you can make use of the atmospheric sciences tower, which conveniently has a solar irradiance meter on top. Irradiance data is published through NOAA SOLRAD here: https://gml.noaa.gov/aftp/data/radiation/solrad/msn/

It’s possible to look directly in the NOAA .dat files to see irradiances for every minute of the day, but a script I vibecoded makes the whole process super easy. The file should be in the same sharepoint folder you saw this (IVNormalizer.py). Just record your raw IV data in a spreadsheet with column names “V” and “I”, and export the file as a csv. Next, go into the python script and fill in the temperature coefficients for whichever cell you're trying to test. The values I used when I first made this were for Maxeon Gen 7 cells. When you run the python script, it’ll ask you for the input csv (raw IV values), the filename for the output csv, the date/time of testing, and temperature. Once it runs, it’ll output a csv including your normalized datapoints as well as interpolated values for Pmp, Vmp, Imp, and some other useful info.

Good luck!


Written by Milan Mishra (if you need to reach me: milan\[dot]mishra\[at]wisc\[dot]edu)

