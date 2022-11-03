#!/bin/sh
# This script downloads Python pickles required for running the NOAA/PSL Experimental LIM Forecast model (v1.21)

echo "Downloading LIM Python files..."


HOST='ftp2.psl.noaa.gov'
USER='anonymous'
PASSWD=''

echo "Downloading data_clim files..."

mkdir data_clim
cd data_clim

ftp -n $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
binary
cd /Projects/LIM/Realtime/Realtime/webData/data_clim
prompt
mget *
quit
END_SCRIPT

mkdir cpcdata
cd cpcdata
ftp -n $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
binary
cd /Projects/LIM/Realtime/Realtime/webData/data_clim/cpcdata
prompt
mget *
quit
END_SCRIPT
cd ..

mkdir tmp
cd tmp
ftp -n $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
binary
cd /Projects/LIM/Realtime/Realtime/webData/data_clim/tmp
prompt
mget *
quit
END_SCRIPT
cd ..

cd ..
echo "data_clim downloaded"


echo "Downloading data_realtime files..."

mkdir data_realtime
cd data_realtime

ftp -n $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
binary
cd /Projects/LIM/Realtime/Realtime/webData/data_realtime
prompt
mget *
quit
END_SCRIPT

cd ..
echo "data_realtime downloaded"


echo "Downloading rawdata files..."

mkdir rawdata
cd rawdata
mkdir colIrr
cd colIrr

ftp -n $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
binary
cd /Projects/LIM/Realtime/Realtime/webData/rawdata/colIrr
prompt
mget *
quit
END_SCRIPT
cd ..

mkdir H100
cd H100

ftp -n $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
binary
cd /Projects/LIM/Realtime/Realtime/webData/rawdata/H100
prompt
mget *
quit
END_SCRIPT
cd ..

mkdir H500
cd H500

ftp -n $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
binary
cd /Projects/LIM/Realtime/Realtime/webData/rawdata/H500
prompt
mget *
quit
END_SCRIPT
cd ..

mkdir SLP
cd SLP

ftp -n $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
binary
cd /Projects/LIM/Realtime/Realtime/webData/rawdata/SLP
prompt
mget *
quit
END_SCRIPT
cd ..

mkdir T2m
cd T2m

ftp -n $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
binary
cd /Projects/LIM/Realtime/Realtime/webData/rawdata/T2m
prompt
mget *
quit
END_SCRIPT
cd ..

echo "rawdata downloaded"


echo "ALL DOWNLOADS COMPLETE"

exit 0

echo""
