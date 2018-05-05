#!/bin/bash

MY_SCRIPT_DIR=
MGIZA_BIN=

SRC_FILE=
TRG_FILE=

# preprocess for mgiza
$MGIZA_BIN/plain2snt $SRC_FILE $TRG_FILE
$MGIZA_BIN/plain2snt $TRG_FILE $SRC_FILE
$MGIZA_BIN/mkcls -c256 -p$SRC_FILE -V$SRC_FILE.vcb.classes opt
$MGIZA_BIN/mkcls -c256 -p$TRG_FILE -V$TRG_FILE.vcb.classes opt
$MGIZA_BIN/snt2cooc "$SRC_FILE"_"TRG_FILE".cooc $SRC_FILE.vcb $TRG_FILE.vcb "$SRC_FILE"_"$TRG_FILE".snt
$MGIZA_BIN/snt2cooc "TRG_FILE"_"$SRC_FILE".cooc $SRC_FILE.vcb $TRG_FILE.vcb "$TRG_FILE"_"$SRC_FILE".snt

# f2e direction mgiza
rm ???-??-??.*
$MGIZA_BIN/mgiza giza.config.a -ncpu 1 > log.a.txt
mkdir giza.f2e
mv ???-??-??.* giza.f2e
mv giza.f2e/*.Ahmm.part000 giza.f2e/f2e.alignment

# e2f direction mgiza
rm ???-??-??.*
$MGIZA_BIN/mgiza giza.config.b -ncpu 1 > log.b.txt
mkdir giza.e2f
mv ???-??-??.* giza.e2f
mv giza.e2f/*.Ahmm.part000 giza.e2f/e2f.alignment

python3 giza_align_extract.py $SRC_FILE $TRG_FILE giza.f2e/f2e.alignment giza.e2f/e2f.alignment --output giza_alignment.final
