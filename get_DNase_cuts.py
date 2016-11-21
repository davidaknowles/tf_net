# Scans the genome for DNase cuts and caches them in both a plain text file and npz. 

import pyDNase
import pysam
import numpy as np
import os

import glob

cell_type_id=int(os.environ["SLURM_ARRAY_TASK_ID"])

DATADIR=os.environ["DREAM_ENCODE_DATADIR"]

cell_types=open("cell_types.txt","rb").readlines()

cell_type=cell_types[ cell_type_id ].strip()

filebase=DATADIR + "/DNase/DNASE." + cell_type

bams=glob.glob(filebase+".*.bam")

if cell_type=="K562": # there are so many otherwise
    bams=[ filebase + '.biorep2.techrep%i.bam' % i for i in (3,5) ]
ps=pysam.AlignmentFile(bams[0],"rb")
chrs=zip(ps.references,ps.lengths)

bam_handlers=[]

for f in bams:
    try:
        bh=pyDNase.BAMHandler( f, caching=False)
        bam_handlers.append(bh)
    except:
        print("Problem with " + f)

total_cuts=0L

data={}
where={}
chunk=1000000

import gzip
outfiles={ strand:gzip.open(filebase+strand+".txt.gz","wb") for strand in ("+","-") }

for chrom,chrom_len in chrs:
    print(chrom)
    start=0
    while start < chrom_len:
        print(start)
        end=min(start+chunk,chrom_len)
        cuts={ "+":np.zeros(end-start), "-":np.zeros(end-start) }
        for reads in bam_handlers: # combine bio/tech replicates
            temp=reads["%s,%i,%i,+" % (chrom,start,end)]
            for k in cuts: # for each strand
                cuts[k] += temp[k]
        for strand,strand_cuts in cuts.iteritems():
            for i in np.arange(end-start):
                if strand_cuts[i]>0:
                    total_cuts += strand_cuts[i]
                    outfiles[strand].write("%s %d %d\n" % (chrom, start+i, strand_cuts[i]))
        start+=chunk

for strand in outfiles:
    outfiles[strand].close()

# Convert the output text format to more efficient .npz
import read_cuts
(data,where,total)=read_cuts.read_both(cell_type)

d={}
for strand in data:
    for chrom in data[strand]:
        d["data%s_%s" % (strand,chrom)]=data[strand][chrom]
        d["where%s_%s" % (strand,chrom)]=where[strand][chrom]
    d["total%s" % strand]=total[strand]

np.savez(DATADIR+cell_type+".npz", **d)

