#!/bin/sh

for f in $DREAM_ENCODE_DATADIR/DNase/*.bam; do
    echo $f
    outfile=$f.bai
    if [ ! -f $outfile ]; then
	samtools index $f
    fi
done
