#!/bin/sh

# Download all the image datasets from figshare.
# Credits Bradbury et al. https://www.nature.com/articles/sdata2016106

mkdir Data
cd Data

wget https://ndownloader.figshare.com/articles/3385828/versions/1 -O Fresno.zip
wget https://ndownloader.figshare.com/articles/3385807/versions/1 -O Oxnard.zip
wget https://ndownloader.figshare.com/articles/3385789/versions/1 -O Modesto.zip
wget https://ndownloader.figshare.com/articles/3385804/versions/1 -O Stockton.zip
wget https://ndownloader.figshare.com/files/6025419 -O SolarArrayPolygons.geojson

unzip Fresno.zip -d Fresno
unzip Modesto.zip -d Modesto
unzip Oxnard.zip -d Oxnard
unzip Stockton.zip -d Stockton

rm Stockton.zip Modesto.zip Oxnard.zip Fresno.zip