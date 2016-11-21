basedir=paste0(Sys.getenv("DREAM_ENCODE_DATADIR"),"gene_expression/")
files=list.files(basedir,"*.gz")

a=read.table(paste0(basedir,files[1]), header = T, stringsAsFactors = F)
b=read.table(paste0(basedir,files[2]), header = T, stringsAsFactors = F)

require(foreach)
ge=foreach(f=files, .combine = cbind) %do% {
  read.table(paste0(basedir,f), header = T, stringsAsFactors = F)$FPKM
}

trans=asinh(ge)

require(irlba)
pca=irlba(trans,2)
plot(pca$v[,1],pca$v[,2],col="white")

fn_split=do.call(rbind,strsplit(files,".",fixed=T))

cell_types=fn_split[,2]

text(pca$v[,1],pca$v[,2],cell_types)

pca=irlba(trans,8)
v=pca$v
rownames(v)=paste(fn_split[,2],fn_split[,3],sep=".")
unique_cell_types=unique(cell_types)
sum_data=foreach(ct=unique_cell_types,.combine=rbind) %do% {
  .5*(v[paste0(ct,".biorep1"),]+v[paste0(ct,".biorep2"),] )
}
rownames(sum_data)=unique_cell_types

write.table(t(sum_data), "ge_pca.txt", row.names = F, col.names = T, quote=F, sep="\t")
