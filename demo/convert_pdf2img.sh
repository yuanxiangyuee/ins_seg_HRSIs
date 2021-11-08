#!/bin/bash

pdfdir='/raid/yuanxiangyue/project/PANet/tools/Outputs/results/panet+asff+carafe/model_step104999/vis/'

while getopts 'd:' flag; do
    case "$flag" in
        d) pdfdir=$OPTARG ;;
    esac
done

for pdf in $(ls ${pdfdir}/*.pdf); do
    fname="${pdf%.*}"
    convert -density 300x300 -quality 95 $pdf ${fname}.jpg
done
