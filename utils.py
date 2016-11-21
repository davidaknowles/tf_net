import theano
import theano.tensor as T
import gzip
import numpy as np
import re

from collections import OrderedDict

from sortedcontainers import SortedDict

def randn( shape, sd, dtype=theano.config.floatX ):
    return( np.random.randn( np.prod(shape) ).astype( dtype ).reshape( shape ) * sd )

def valid_seq(seq,length):
    return(seq != None and seq!="" and len(seq)==length and not "N" in seq)

# Shared floating-point Theano variable from a numpy variable
def sharedf(x, name=None, borrow=False):
    return theano.shared(np.asarray(x, dtype=theano.config.floatX), name=name, borrow=borrow)

def AdaMax(w, objective, alpha=.01, beta1=.1, beta2=.001, verbose=False):
    if verbose: print 'AdaMax', 'alpha:',alpha,'beta1:',beta1,'beta2:',beta2
    g = T.grad(objective, w, disconnected_inputs='warn')
    
    new = OrderedDict()
    
    for i in range(len(w)):
        #gi = T.switch(T.isnan(gi),T.zeros_like(gi),gi) #remove NaN's
        mom1 = sharedf(w[i].get_value() * 0.)
        _max = sharedf(w[i].get_value() * 0.)
        new[mom1] = (1.0-beta1) * mom1 + beta1 * g[i]
        new[_max] = T.maximum((1.0-beta2)*_max, abs(g[i]) + 1e-8)
        new[w[i]] = w[i] - alpha *  new[mom1] / new[_max]                
    return new


# Find indices of str_to_find in str_to_search
def find_all(str_to_search,str_to_find):
    return [m.start() for m in re.finditer(str_to_find, str_to_search)]

# Get intersection of two lists
def intersect(a,b):
    return list(set(a).intersection(b))

def moveaxis(a, source, destination):

    order = [n for n in range(a.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    result = a.transpose(order)
    return result


def get_fasta_chrom(fasta_file, chroms):
    current_chrom=None
    dic={}
    line_counter=0L
    with gzip.open(fasta_file) as f:
        for l in f:
            line_counter+=1
            if l[0]==">":
                if not current_chrom is None:
                    dic[current_chrom]="".join(dic[current_chrom]).upper()
                fchrom=l[1:].strip()
                if fchrom in chroms:
                    current_chrom=fchrom
                    dic[current_chrom]=[]
                    print("Loading "+current_chrom)
                else:
                    print("Ignoring "+fchrom)
                    current_chrom=None
                    if len( set(chroms) - set(dic.keys()))==0:
                        return(dic)
            else:
                if not current_chrom is None:
                    dic[current_chrom].append( l.strip() )

    if not current_chrom is None:
        dic[current_chrom]="".join(dic[current_chrom]).upper()
        
    return(dic)

from string import maketrans
REVERSER=maketrans("AGCT","TCGA")
    
def reverse_complement(seq):
    return seq.translate(REVERSER)[::-1]

def fetch_sequence(dic, fasta_id, start, end, strand):
    if not fasta_id in dic:
        return None
    seq =  dic[fasta_id][int(start):int(end)]

    return seq if strand=="+" else reverse_complement(seq)

