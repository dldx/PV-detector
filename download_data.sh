#!/bin/sh

# Download all the image datasets from figshare.
# Credits Bradbury et al. https://www.nature.com/articles/sdata2016106

mv Example_Data Data
mv Data/config.py .
cd Data

wget https://ndownloader.figshare.com/files/6025419 -O SolarArrayPolygons.geojson
wget https://ndownloader.figshare.com/articles/3385828/versions/1 -O Fresno.zip
unzip Fresno.zip -d Fresno
rm Fresno.zip
wget https://ndownloader.figshare.com/articles/3385807/versions/1 -O Oxnard.zip
unzip Oxnard.zip -d Oxnard
rm Oxnard.zip
wget https://ndownloader.figshare.com/articles/3385789/versions/1 -O Modesto.zip
unzip Modesto.zip -d Modesto
rm Modesto.zip
wget https://ndownloader.figshare.com/articles/3385804/versions/1 -O Stockton.zip
unzip Stockton.zip -d Stockton
rm Stockton.zip
