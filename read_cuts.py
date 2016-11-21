import pysam
import numpy as np
import gzip
import timeit

DATADIR=os.environ["DREAM_ENCODE_DATADIR"]

ps=pysam.AlignmentFile(DATADIR+"DNase/DNASE.A549.biorep1.techrep1.bam","rb")
chrs=ps.references
chr_lens=dict(zip(ps.references,ps.lengths))

def read_cuts(input_fn):
    data={}
    where={}
    max_val=0
    max_pos=0
    total=0L
    prev_chrom=""
    start_time=timeit.default_timer()
    with gzip.open(input_fn,"rb") as input_file:
        for l in input_file:
            ss=l.strip().split()
            chrom=ss[0]
            if not chrom==prev_chrom and not prev_chrom=="":
                print("%s %i %i %f" % (prev_chrom,max_pos,chr_lens[prev_chrom],timeit.default_timer()-start_time))
                max_pos=0
            prev_chrom=chrom
            (pos,count)=map(int,ss[1:])
            if not chrom in data:
                data[chrom]=[]
                where[chrom]=[]
            max_val=max(max_val,count)
            max_pos=max(max_pos,pos)
            total += count
            data[chrom].append(count)
            where[chrom].append(pos)

    for chrom in where:
        data[chrom]=np.array(data[chrom], dtype=np.uint16)
        where[chrom]=np.array(where[chrom], dtype=np.uint32)

    return(data,where,total,max_val)

def read_npz(cell_type):
    d=np.load(DATADIR+cell_type+".npz")
    data={}
    where={}
    total={}
    for strand in ("+","-"):
        data[strand]={}
        where[strand]={}
        total[strand]=d["total%s" % strand]
        for chrom in (set( chr_lens.keys() )-set( ("chrY", ) )):
            data[strand][chrom]=d["data%s_%s" % (strand,chrom)]
            where[strand][chrom]=d["where%s_%s" % (strand,chrom)]
    return(data,where,total)
        
def get_chunk(start,end,data,where,strand,chrom,trans=np.sqrt):
    result=np.zeros( end-start, dtype=np.float32)

    if not chrom in where[strand]: return(result)
    wh=where[strand][chrom]
    da=data[strand][chrom]

    for i in range( np.searchsorted(wh, start), np.searchsorted(wh, end) ):
        result[ wh[i]-start ]=trans(da[ i ]) 
    return(result)

def read_both(cell_type):
    data={}
    where={}
    total={}
    for strand in ("+","-"):
        (data[strand],where[strand],total[strand],max_val)=read_cuts(DATADIR+"DNASE.%s%s.txt.gz" % (cell_type,strand))
    return(data,where,total)

def read_both_strands(cell_type):
    (data,where,total)=read_npz(cell_type)
    return(lambda chrom,strand,start,end: (get_chunk(start,end,data,where,strand,chrom,trans=lambda g:g)))

def read_both_strands_corrected(cell_type, default_num_reads=5e7):
    (data,where,total)=read_npz(cell_type)
    depth_correction=default_num_reads / np.sum(total.values())
    return(lambda chrom,strand,start,end: (get_chunk(start,end,data,where,strand,chrom) * depth_correction) )
