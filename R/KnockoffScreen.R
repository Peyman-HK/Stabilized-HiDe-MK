
KS.prelim<-function(Y, X=NULL, id=NULL, out_type="C"){
  ##Preliminary
  Y<-as.matrix(Y);n<-nrow(Y)
  
  if(length(X)!=0){X0<-svd(as.matrix(X))$u}else{X0<-NULL}
  X0<-cbind(rep(1,n),X0)
  
  if(out_type=="C"){nullglm<-glm(Y~0+X0,family=gaussian)}
  if(out_type=="D"){nullglm<-glm(Y~0+X0,family=binomial)}
  
  if (length(id)==0){id<-1:n}
  
  mu<-nullglm$fitted.values;Y.res<-Y-mu

  #prepare invserse matrix for covariates
  if(out_type=='D'){v<-mu*(1-mu)}else{v<-rep(as.numeric(var(Y.res)),length(Y))}
  inv.X0<-solve(t(X0)%*%(v*X0))
  inv.vX0<-solve(t(X0)%*%(v*X0))
  
  #prepare the preliminary features
  result.prelim<-list(Y=Y,id=id,n=n,X0=X0,nullglm=nullglm,out_type=out_type,inv.X0=inv.X0,inv.vX0=inv.vX0)
  return(result.prelim)
}

KS.chr<-function(result.prelim,seq.filename,window.bed,region.pos,tested.pos=NULL,excluded.pos=NULL,M=5,thres.single=0.01,thres.ultrarare=25,thres.missing=0.10,midout.dir=NULL,temp.dir=NULL,knockoffs.dir=NULL,jobtitle=NULL,Gsub.id=NULL,maxN.neighbor=Inf,maxBP.neighbor=Inf,impute.method='fixed',bigmemory=T,leveraging=T,LD.filter=NULL,corr_max=0.75){
  
  time.start<-proc.time()[3]
  #region.step=0.1*10^5
  options(scipen = 999) #remove scientific notations
  chr<-window.bed[1,1]
  pos.min<-min(window.bed[,2:3])
  pos.max<-max(window.bed[,2:3])
  max.size<-max(window.bed[,3]-window.bed[,2])

  null.model<-F;
  result.summary.single<-c();result.summary.window<-c();variant.info<-c()
  G.now<-c();G.before<-c();G.after<-c()
  start.index<-1
  while (start.index<length(region.pos)){
    start<-region.pos[start.index]
    end<-region.pos[start.index+1]-1

    print(paste0(percent((start-pos.min)/(pos.max-pos.min)),' finished. Time used: ', round(proc.time()[3]-time.start,digits=1),'s. Scan ',start,'-',end))
    start.index<-start.index+1
    
    print('extracting genotype data')
    
    temp.pos<-floor(unique(c(seq(start,end,by=min(25000,ceiling((end-start)/5))),end)))
    G<-c()
    for(i in 1:(length(temp.pos)-1)){
      temp.start<-temp.pos[i];temp.end<-temp.pos[i+1]
      
      if(length(grep('.bgen',seq.filename,ignore.case = T))!=0){
        if(chr<=9){bgen.chr<-paste0('0',chr)}else{bgen.chr<-chr} #change chr notation for .bgen file
        range<-paste0(bgen.chr,":",temp.start,"-",temp.end)
        temp.X<-Matrix(t(readBGENToMatrixByRange(seq.filename, range)[[1]]),sparse=T)
      }
      if(length(grep('.vcf',seq.filename,ignore.case = T))!=0){
        range<-paste0(chr,":",temp.start,"-",temp.end)
        temp.X<-Matrix(t(readVCFToMatrixByRange(seq.filename, range,annoType='')[[1]]),sparse=T)
      }

      if(i==1){
        #match phenotype id and genotype id
        if(length(Gsub.id)==0){match.index<-match(result.prelim$id,rownames(temp.X))}else{
          match.index<-match(result.prelim$id,Gsub.id)
        }
        if(mean(is.na(match.index))>0){
          msg<-sprintf("Some individuals are not matched with genotype. The rate is%f", mean(is.na(match.index)))
          warning(msg,call.=F)
        }
      }
    
      #match tested pos and genotype pos
      if(ncol(temp.X)==0){next}
      pos<-as.numeric(gsub("^.*\\:","",colnames(temp.X)))
      if(length(tested.pos)==0){selected.pos<-pos}else{selected.pos<-intersect(tested.pos,pos)}
      if(length(excluded.pos)!=0){selected.pos<-setdiff(selected.pos,excluded.pos)}
      matchG.index<-match(sort(selected.pos),pos)
      temp.X<-temp.X[match.index,matchG.index,drop=F]
      
      #missing rate filtering
      if(ncol(temp.X)==0){next}
      temp.MISS<-colMeans(temp.X<0 | temp.X>2)
      temp.X<-temp.X[,temp.MISS<=thres.missing,drop=F]
      
      G <- cbind(G,temp.X)
      rm(temp.X);gc()
    }


    print('processing genotype data')
    if(length(G)==0){
      msg<-'Number of variants in the specified range is 0'
      warning(msg,call.=F)
      next
    }else{
      if(ncol(G)==1){
        msg<-'Number of variants in the specified range is 1'
        warning(msg,call.=F)
        next
      }
    }
    G<-G[,match(unique(colnames(G)),colnames(G)),drop=F]
    
    # missing genotype imputation
    G[G<0 | G>2]<-NA
    MISS.freq<-colMeans(is.na(G))
    if(sum(G<0,na.rm=T)+sum(G>2,na.rm=T)+sum(is.na(G))>0){
      N_MISS<-sum(is.na(G))
      if(N_MISS>0){
        msg<-sprintf("The missing genotype rate is %f. Imputation is applied.", N_MISS/nrow(G)/ncol(G))
        warning(msg,call.=F)
        G<-Impute(G,impute.method)
      }
    }
    #sparse matrix operation
    MAF<-colMeans(G)/2;MAC<-colSums(G)
    MAF[MAF>0.5]<-1-MAF[MAF>0.5]
    MAC[MAF>0.5]<-nrow(G)*2-MAC[MAF>0.5]
    s<-colMeans(G^2)-colMeans(G)^2
    SNP.index<-which(MAF>0 & MAC>=thres.ultrarare & s!=0 & !is.na(MAF))# & MAC>10
  
    #check.index<-which(MAF>0 & MAC>=thres.ultrarare & s!=0 & !is.na(MAF)  & MISS.freq<0.1)
    if(length(SNP.index)<=1 ){
      msg<-'Number of variants with missing rate <=10% in the specified range is <=1'
      warning(msg,call.=F)
      next
    }
    G<-G[,SNP.index,drop=F]
    pos<-as.numeric(gsub("^.*\\:","",colnames(G)))

    ##single variant test for all variants
    p.single<-as.matrix(Get.p(G,result.prelim))
    
    #output info for tested variants
    temp.variant.info<-cbind(colnames(G),p.single,MAF[SNP.index],MAC[SNP.index])
    colnames(temp.variant.info)<-c('pos','pvalue','MAF','MAC')
    
    if(length(LD.filter)!=0){
      #clustering and filtering
      sparse.fit<-sparse.cor(G)
      cor.X<-sparse.fit$cor;cov.X<-sparse.fit$cov
      
      Sigma.distance = as.dist(1 - abs(cor.X))
      if(ncol(G)>1){
        fit = hclust(Sigma.distance, method="complete")
        corr_max = LD.filter
        clusters = cutree(fit, h=1-corr_max)
      }else{clusters<-1}
      
      temp.index<-sample(length(p.single))
      temp.index<-temp.index[match(unique(clusters),clusters[temp.index])]
      if(length(temp.index)<=1 ){
        msg<-'Number of variants after LD filtering in the specified range is <=1'
        warning(msg,call.=F)
        next
      }
      G<-G[,temp.index,drop=F]
      p.single<-p.single[temp.index,,drop=F]
      
      temp.variant.info<-cbind(temp.variant.info,paste0(start,'-',end,'-',clusters))
      colnames(temp.variant.info)[ncol(temp.variant.info)]<-'cluster'
    }
    variant.info<-rbind(variant.info,temp.variant.info)
    #get positions and reorder G
    pos<-as.numeric(gsub("^.*\\:","",colnames(G)))
    G<-G[,order(pos),drop=F]
    p.single<-p.single[order(pos),,drop=F]
    
    MAF<-colMeans(G)/2
    G<-as.matrix(G)
    G[,MAF>0.5 & !is.na(MAF)]<-2-G[,MAF>0.5 & !is.na(MAF)]
    MAF<-colMeans(G)/2;MAC<-colSums(G)
    G<-Matrix(G,sparse=T)
 
    #get positions
    pos<-as.numeric(gsub("^.*\\:","",colnames(G)))

    print('generating knockoffs')
    gc()
    #generate knockoffs
    if(leveraging==T){
      G_k<-create.MK.AL(G,pos,M=M,corr_max=corr_max,maxN.neighbor=maxN.neighbor,maxBP.neighbor=maxBP.neighbor,thres.ultrarare=thres.ultrarare,bigmemory=bigmemory)
    }else{
      G_k<-create.MK.AL(G,pos,M=M,corr_max=corr_max,maxN.neighbor=maxN.neighbor,maxBP.neighbor=maxBP.neighbor,thres.ultrarare=thres.ultrarare,bigmemory=bigmemory,n.AL=nrow(G))
    }

    #save knockoffs
    if(length(knockoffs.dir)!=0){
      print('saving knockoffs')
      temp.filename<-paste0(knockoffs.dir,chr,':',start,'-',end,'.gds')
      gfile <- createfn.gds(temp.filename)
      temp.X<-round(as.matrix(G),digits=1)
      add.gdsn(gfile, "original", val=temp.X, compress='LZMA_RA', storage="packedreal16")
      add.gdsn(gfile, "sample.id", val=rownames(G), compress='LZMA_RA', storage="fstring")
      add.gdsn(gfile, "variant.id", val=colnames(G), compress='LZMA_RA', storage="fstring")
      for(i in 1:M){
        print(i)
        temp.X<-as.matrix(G_k[[i]][,])
        add.gdsn(gfile, paste0("knockoffs_",i), val=temp.X, compress='LZMA_RA', storage="packedreal16")
      }
      closefn.gds(gfile)
    }
    
    print('knockoff analysis')
    ##single variant test for all variants
    p.single_k<-c()
    for(k in 1:M){
      temp.p<-c()
      for(i in 1:ceiling(ncol(G)/1000)){
        temp.X<-G_k[[k]][,(1+(i-1)*1000):min(ncol(G),i*1000),drop=F]
        temp.p<-rbind(temp.p,Get.p(temp.X,result.prelim=result.prelim))
      }
      p.single_k<-cbind(p.single_k,temp.p)
      gc()
    }
    
    print('common variants')
    ##common variants
    common.index<-which(MAF>=thres.single)
    if(length(common.index)>=1){
      p.common<-p.single[common.index,,drop=F]
      p.common_k<-p.single_k[common.index,,drop=F]
      
      W<-(-log10(p.common)-apply(-log10(p.common_k),1,median))*(-log10(p.common)>=apply(-log10(p.common_k),1,max))
      W[is.na(W)]<-0
      MK.stat<-MK.statistic(-log10(p.common),-log10(p.common_k),method='median')
      temp.summary.single<-cbind(chr,pos[common.index],pos[common.index],
                                 pos[common.index],pos[common.index],
                                 MK.stat,W,p.common,
                                 p.common_k,MAF[common.index])
      
      colnames(temp.summary.single)<-c('chr','start','end','actual_start','actual_end',
                                       'kappa','tau',
                                       'W_KS','P_KS',paste0('P_KS_k',1:M),'MAF')
      if(length(midout.dir)!=0){
        write.table(temp.summary.single,paste0(midout.dir,jobtitle,'_single_',chr,':',start,'-',end,'.txt'),sep='\t',row.names=F,col.names=T,quote=F)
      }
      result.summary.single<-rbind(result.summary.single,temp.summary.single)
    }
    
    print('rare variants')
    ##rare variants
    rare.index<-which(MAF<thres.single & MAC>=thres.ultrarare)
    if(length(rare.index)>=1){
      MAF<-MAF[rare.index];MAC<-MAC[rare.index]
      pos<-as.numeric(gsub("^.*\\:","",colnames(G)[rare.index]))
      p.rare<-p.single[rare.index]
      p.rare_k<-p.single_k[rare.index,,drop=F]
      
      #set beta weights and windows
      weight.beta<-dbeta(MAF,1,25)
      weight.matrix<-as.matrix(weight.beta)
      colnames(weight.matrix)<-c(paste0('MAF<',thres.single,'&MAC>',thres.ultrarare,'&Beta'))
      window.temp<-window.bed[window.bed[,2]<max(pos) & window.bed[,3]>min(pos),]
      if(length(nrow(window.temp))==0){window.temp<-matrix(window.temp,1,ncol(window.bed))}
      window.matrix0<-matrix(apply(window.temp,1,function(x)as.numeric(pos>=x[2] & pos<x[3])),length(pos),nrow(window.temp))
      window.string<-apply(window.matrix0,2,function(x)paste(as.character(x),collapse = ""))
      
      window.MAC<-apply(MAC*window.matrix0,2,sum)
      window.index<-intersect(match(unique(window.string),window.string),which(apply(window.matrix0,2,sum)>1 & window.MAC>=10))
      if(length(window.index)==0){
        start<-end
        next
      }
      window.matrix0<-as.matrix(window.matrix0[,window.index])
      
      window.matrix<-Matrix(window.matrix0)
      window.summary<-cbind(window.temp[window.index,2],window.temp[window.index,3],t(apply(window.matrix,2,function(x)c(min(pos[which(x==1)]),max(pos[which(x==1)])))))
      
      p.KS<-c();p.KS_k<-c();p.individual<-c()
      for(i in 1:ceiling(ncol(window.matrix)/100)){
        temp.window.matrix<-window.matrix[,(1+(i-1)*100):min(ncol(window.matrix),i*100),drop=F]
        temp.index<-which(rowSums(temp.window.matrix)!=0)
        temp.weight.matrix<-weight.matrix[temp.index,,drop=F]
        temp.window.matrix<-temp.window.matrix[temp.index,,drop=F]
        
        temp.G<-G[,rare.index[temp.index],drop=F]
        temp.G_k<-list()
        for(k in 1:M){
          temp.G_k[[k]]<-Matrix(G_k[[k]][,rare.index[temp.index],drop=F],sparse=T)
        }
        KS.fit<-KS.test(temp.G,temp.G_k,p.rare[temp.index],p.rare_k[temp.index,,drop=F],result.prelim,window.matrix=temp.window.matrix,weight.matrix=temp.weight.matrix)
        p.KS<-rbind(p.KS,KS.fit$p.KS)
        p.KS_k<-rbind(p.KS_k,KS.fit$p.KS_k)
        p.individual<-rbind(p.individual,KS.fit$p.individual)
        gc()
      }
      
      #######
      p.A<-p.KS;p.A_k<-p.KS_k
      #Knockoff statistics
      W<-(-log10(p.A)-apply(-log10(p.A_k),1,median))*(-log10(p.A)>=apply(-log10(p.A_k),1,max))
      MK.stat<-MK.statistic(-log10(p.A),-log10(p.A_k),method='median')
      
      temp.summary.window<-cbind(chr,window.summary,
                                   MK.stat,
                                   W,p.A,p.A_k,
                                   p.individual)
      colnames(temp.summary.window)[1:(grep('burden_MAF',colnames(temp.summary.window))-1)]<-c('chr','start','end','actual_start','actual_end',
                                                                                                   'kappa','tau',
                                                                                                   'W_KS','P_KS',paste0('P_KS_k',1:M))
      if(length(midout.dir)!=0){
        write.table(temp.summary.window,paste0(midout.dir,jobtitle,'_window_',chr,':',start,'-',end,'.txt'),sep='\t',row.names=F,col.names=T,quote=F)
      }
      
      result.summary.window<-rbind(result.summary.window,temp.summary.window)
    }
    
    #start<-end
  }

  return(list(result.single=result.summary.single,result.window=result.summary.window,variant.info=variant.info))
}


KS.test<-function(temp.G,temp.G_k,p.rare,p.rare_k,result.prelim,window.matrix,weight.matrix){
  
  mu<-result.prelim$nullglm$fitted.values;
  Y.res<-result.prelim$Y-mu;re.Y.res<-result.prelim$re.Y.res
  X0<-result.prelim$X0;outcome<-result.prelim$out_type
  M<-length(temp.G_k)
  
  #Burden test
  p.burden<-matrix(NA,ncol(window.matrix),ncol(weight.matrix))
  p.burden_k<-array(NA,dim=c(length(temp.G_k),ncol(window.matrix),ncol(weight.matrix)))
  for (k in 1:ncol(weight.matrix)){
    temp.window.matrix<-weight.matrix[,k]*window.matrix
    p.burden[,k]<-Get.p(temp.G%*%temp.window.matrix,result.prelim)
    temp<-c()
    for(i in 1:M){
      temp<-cbind(temp,Get.p(temp.G_k[[i]][]%*%temp.window.matrix,result.prelim=result.prelim))
      gc()
    }
    p.burden_k[,,k]<-t(temp)
  }
  
  #Dispersion test
  p.dispersion<-matrix(NA,ncol(window.matrix),ncol(weight.matrix))
  p.dispersion_k<-array(NA,dim=c(length(temp.G_k),ncol(window.matrix),ncol(weight.matrix)))
 
  #proc.time()
  if(outcome=='D'){v<-mu*(1-mu)}else{v<-rep(as.numeric(var(Y.res)),nrow(temp.G))}
  A<-crossprod(temp.G,v*temp.G)#t(temp.G)%*%(v*temp.G)
  B<-crossprod(temp.G,v*X0)#t(temp.G)%*%(v*X0)
  C<-result.prelim$inv.vX0#solve(t(X0)%*%(v*X0))
  K<-A-B%*%C%*%t(B) #here we use the same K for original and knockoffs, due to the exchangeability
  
  score<-t(temp.G)%*%Y.res#;re.score<-t(t(temp.G)%*%re.Y.res)
  score_k<-c()
  for(i in 1:M){
    score_k<-cbind(score_k,crossprod(temp.G_k[[i]][,],Y.res))
    gc()
  }

  for (k in 1:ncol(weight.matrix)){
    #print(k)
    p.dispersion[,k]<-Get.p.SKAT(score,K,window.matrix,weight=weight.matrix[,k],result.prelim)
    p.dispersion_k[,,k]<-t(sapply(1:length(temp.G_k),function(s){Get.p.SKAT(score_k[,s],K,window.matrix,weight=weight.matrix[,k],result.prelim)}))
  }
  #proc.time()
  
  p.V1<-Get.cauchy.scan(p.rare,window.matrix)
  p.V1_k<-apply(p.rare_k,2,Get.cauchy.scan,window.matrix=window.matrix)
  if(ncol(window.matrix)==1){p.V1_k<-matrix(p.V1_k,1,length(temp.G_k))}
  
  p.individual<-cbind(p.burden,p.dispersion,p.V1);
  colnames(p.individual)<-c(paste0('burden_',colnames(weight.matrix)),
                            paste0('dispersion_',colnames(weight.matrix)),
                            paste0('singleCauchy'))
  
  p.KS<-as.matrix(apply(p.individual,1,Get.cauchy))
  p.KS_k<-matrix(sapply(1:length(temp.G_k),function(s){apply(cbind(matrix(p.burden_k[s,,],dim(p.burden_k)[2],dim(p.burden_k)[3]),
                                                                matrix(p.dispersion_k[s,,],dim(p.dispersion_k)[2],dim(p.dispersion_k)[3]),
                                                                p.V1_k[,s]),1,Get.cauchy)}),dim(p.burden_k)[2],dim(p.burden_k)[1])
  
  return(list(p.KS=p.KS,p.KS_k=p.KS_k,p.individual=p.individual))
}



create.MK.AL <- function(X,pos,M,corr_max=0.75,maxN.neighbor=Inf,maxBP.neighbor=100000,corr_base=0.05,n.AL=floor(10*nrow(X)^(1/3)*log(nrow(X))),thres.ultrarare=25,R2.thres=0.75,method='shrinkage',bigmemory=F) {
  
  if(class(X)!='dgCMatrix'){X<-Matrix(X,sparse=T)} #convert it to sparse matrix format
  
  sparse.fit<-sparse.cor(X)
  cor.X<-sparse.fit$cor;cov.X<-sparse.fit$cov
  
  #svd to get leverage score, can be optimized;update: tried fast leveraging, but the R matrix is singular possibly because X is sparse.
  if(method=='svd.irlba'){
    svd.X.u<-irlba(X,nv=floor(sqrt(ncol(X)*log(ncol(X)))))$u
    h<-rowSums(svd.X.u^2)
    prob<-h/sum(h)
  }
  if(method=='svd.full'){
    svd.X.u<-svd(X)$u
    h<-rowSums(svd.X.u^2)
    prob<-h/sum(h)
  }
  if(method=='uniform'){
    h<-rep(1,nrow(X))
    prob<-h/sum(h)
  }
  if(method=='shrinkage'){
    svd.X.u<-irlba(X,nv=floor(sqrt(ncol(X)*log(ncol(X)))))$u
    h1<-rowSums(svd.X.u^2)
    h2<-rep(1,nrow(X))
    prob1<-h1/sum(h1)
    prob2<-h2/sum(h2)
    prob<-0.5*prob1+0.5*prob2
  }
  index.AL<-sample(1:nrow(X),min(n.AL,nrow(X)),replace = FALSE,prob=prob)
  w<-1/sqrt(n.AL*prob[index.AL])
  rm(svd.X.u) #remove temp file
  
  X.AL<-w*X[index.AL,]
  sparse.fit<-sparse.cor(X.AL)
  cor.X.AL<-sparse.fit$cor;cov.X.AL<-sparse.fit$cov
  skip.index<-colSums(X.AL!=0)<=thres.ultrarare #skip features that are ultra sparse, permutation will be directly applied to generate knockoffs
  
  Sigma.distance = as.dist(1 - abs(cor.X))
  if(ncol(X)>1){
    fit = hclust(Sigma.distance, method="single")
    corr_max = corr_max
    clusters = cutree(fit, h=1-corr_max)
  }else{clusters<-1}
  
  X_k<-list()
  for(k in 1:M){
    if(bigmemory==T){X_k[[k]]<-big.matrix(nrow=nrow(X),ncol=ncol(X),init=0)}else{
      X_k[[k]]<-matrix(0,nrow=nrow(X),ncol=ncol(X))
    }
  }

  index.exist<-c()
  for (k in unique(clusters)){
    cluster.fitted<-cluster.residuals<-matrix(NA,nrow(X),sum(clusters==k))
    for(i in which(clusters==k)){
      #print(i)
      rate<-1;R2<-1;temp.maxN.neighbor<-maxN.neighbor
      
      while(R2>=R2.thres){
        
        temp.maxN.neighbor<-floor(temp.maxN.neighbor/rate)
        
        index.pos<-which(pos>=max(pos[i]-maxBP.neighbor,pos[1]) & pos<=min(pos[i]+maxBP.neighbor,pos[length(pos)]))
        temp<-abs(cor.X[i,]);temp[which(clusters==k)]<-0;temp[-index.pos]<-0
        temp[which(temp<=corr_base)]<-0
        
        index<-order(temp,decreasing=T)
        if(sum(temp!=0,na.rm=T)==0 | temp.maxN.neighbor==0){index<-NULL}else{
          index<-setdiff(index[1:min(length(index),floor((nrow(X))^(1/3)),temp.maxN.neighbor,sum(temp!=0,na.rm=T))],i)
        }
        #index<-setdiff(index[1:min(length(index),sum(temp>corr_base),floor((nrow(X.AL))^(1/3)),maxN.neighbor)],i)
        
        y<-X[,i]
        if(length(index)==0){fitted.values<-0}
        if(i %in% skip.index){fitted.values<-0}
        if(!(i %in% skip.index |length(index)==0)){
          
          a<-proc.time()
          
          x.AL<-X.AL[,index,drop=F];
          n.exist<-length(intersect(index,index.exist))
          x.exist.AL<-matrix(0,nrow=nrow(X.AL),ncol=n.exist*M)
          if(length(intersect(index,index.exist))!=0){
            for(j in 1:M){ # this is the most time-consuming part
              x.exist.AL[,((j-1)*n.exist+1):(j*n.exist)]<-w*X_k[[j]][index.AL,intersect(index,index.exist),drop=F]
              #x.exist[,((j-1)*n.exist+1):(j*n.exist)]<-X_k[j,,intersect(index,index.exist),drop=F]
            }
          }
          y.AL<-w*X[index.AL,i];#x.exist.AL<-w*x.exist[index.AL,,drop=F];x.AL<-w*x[index.AL,,drop=F] #create re-scaled data
          
          b<-proc.time()
          #print(b[3]-a[3])
          
          a<-proc.time()
          
          temp.xy<-rbind(mean(y.AL),crossprod(x.AL,y.AL)/length(y.AL)-colMeans(x.AL)*mean(y.AL))
          temp.xy<-rbind(temp.xy,crossprod(x.exist.AL,y.AL)/length(y.AL)-colMeans(x.exist.AL)*mean(y.AL))
          
          temp.cov.cross<-sparse.cov.cross(x.AL,x.exist.AL)$cov
          temp.cov<-sparse.cor(x.exist.AL)$cov
          temp.xx<-cov.X.AL[index,index]
          temp.xx<-rbind(cbind(temp.xx,temp.cov.cross),cbind(t(temp.cov.cross),temp.cov))
          
          temp.xx<-cbind(0,temp.xx)
          temp.xx<-rbind(c(1,rep(0,ncol(temp.xx)-1)),temp.xx)
          
          svd.fit<-svd(temp.xx)
          v<-svd.fit$v
          cump<-cumsum(svd.fit$d)/sum(svd.fit$d)
          n.svd<-which(cump>=0.999)[1]
          svd.index<-intersect(1:n.svd,which(svd.fit$d!=0))
          temp.inv<-v[,svd.index,drop=F]%*%(svd.fit$d[svd.index]^(-1)*t(v[,svd.index,drop=F]))
          temp.beta<-temp.inv%*%temp.xy
          b<-proc.time()
          #print(b[3]-a[3])
          
          a<-proc.time()
          
          x<-X[,index,drop=F]
          temp.j<-1
          fitted.values<-temp.beta[1]+x%*%temp.beta[(temp.j+1):(temp.j+ncol(x)),,drop=F]-sum(colMeans(x)*temp.beta[(temp.j+1):(temp.j+ncol(x)),,drop=F])
          
          if(length(intersect(index,index.exist))!=0){
            temp.j<-temp.j+ncol(x)
            for(j in 1:M){
              temp.x<-X_k[[j]][,intersect(index,index.exist),drop=F]
              if(ncol(temp.x)>=1){
                fitted.values<-fitted.values+temp.x%*%temp.beta[(temp.j+1):(temp.j+ncol(temp.x)),,drop=F]-sum(colMeans(temp.x)*temp.beta[(temp.j+1):(temp.j+ncol(temp.x)),,drop=F])
              }
              temp.j<-temp.j+ncol(temp.x)
            }
          }
          
          b<-proc.time()
          #print(b[3]-a[3])
          
        }
        residuals<-as.numeric(y-fitted.values)
        #overfitted model
        R2<-1-var(residuals,na.rm=T)/var(y,na.rm=T)
        rate<-rate*2;temp.maxN.neighbor<-length(index)
      }
      
      #print(R2)
      
      #if(var(residuals,na.rm=T)/var(y,na.rm=T)<=0.05){fitted.values<-y;residuals<-y-fitted.values}
      #if(var(residuals,na.rm=T)/var(y,na.rm=T)>=0.95){fitted.values<-0;residuals<-y-fitted.values}
      #print(1-var(residuals,na.rm=T)/var(y,na.rm=T))
      
      cluster.fitted[,match(i,which(clusters==k))]<-as.vector(fitted.values)
      cluster.residuals[,match(i,which(clusters==k))]<-as.vector(residuals)
      
      index.exist<-c(index.exist,i)
    }
    #sample mutiple knockoffs
    a<-proc.time()
    
    cluster.sample.index<-sapply(1:M,function(x)sample(1:nrow(X)))
    for(j in 1:M){
      X_k[[j]][,which(clusters==k)]<-round(cluster.fitted+cluster.residuals[cluster.sample.index[,j],,drop=F],digits=1)
      #X_k[j,,which(clusters==k)]<-cluster.fitted+cluster.residuals[cluster.sample.index[,j],,drop=F]
    }
    b<-proc.time()
    #print(b[3]-a[3])
  }
  return(X_k)
}


SKAT_davies <- function(q,lambda,h = rep(1,length(lambda)),delta = rep(0,length(lambda)),sigma=0,lim=10000,acc=0.0001) {
  r <- length(lambda)
  if (length(h) != r) stop("lambda and h should have the same length!")
  if (length(delta) != r) stop("lambda and delta should have the same length!")
  out <- .C("qfc",lambdas=as.double(lambda),noncentral=as.double(delta),df=as.integer(h),r=as.integer(r),sigma=as.double(sigma),q=as.double(q),lim=as.integer(lim),acc=as.double(acc),trace=as.double(rep(0,7)),ifault=as.integer(0),res=as.double(0),PACKAGE="SKAT")
  out$res <- 1 - out$res
  return(list(trace=out$trace,ifault=out$ifault,Qq=out$res))
}

Get_Liu_PVal.MOD.Lambda<-function(Q.all, lambda, log.p=FALSE){
  param<-Get_Liu_Params_Mod_Lambda(lambda)
  Q.Norm<-(Q.all - param$muQ)/param$sigmaQ
  Q.Norm1<-Q.Norm * param$sigmaX + param$muX
  p.value<- pchisq(Q.Norm1,  df = param$l,ncp=param$d, lower.tail=FALSE, log.p=log.p)
  return(p.value)
}

Get_Liu_Params_Mod_Lambda<-function(lambda){
  ## Helper function for getting the parameters for the null approximation
  
  c1<-rep(0,4)
  for(i in 1:4){
    c1[i]<-sum(lambda^i)
  }
  
  muQ<-c1[1]
  sigmaQ<-sqrt(2 *c1[2])
  s1 = c1[3] / c1[2]^(3/2)
  s2 = c1[4] / c1[2]^2
  
  beta1<-sqrt(8)*s1
  beta2<-12*s2
  type1<-0
  
  #print(c(s1^2,s2))
  if(s1^2 > s2){
    a = 1/(s1 - sqrt(s1^2 - s2))
    d = s1 *a^3 - a^2
    l = a^2 - 2*d
  } else {
    type1<-1
    l = 1/s2
    a = sqrt(l)
    d = 0
  }
  muX <-l+d
  sigmaX<-sqrt(2) *a
  
  re<-list(l=l,d=d,muQ=muQ,muX=muX,sigmaQ=sigmaQ,sigmaX=sigmaX)
  return(re)
}

Get.p.SKAT<-function(score,K,window.matrix,weight,result.prelim){
  
  mu<-result.prelim$nullglm$fitted.values;Y.res<-result.prelim$Y-mu
  X0<-result.prelim$X0;outcome<-result.prelim$out_type
  
  Q<-as.vector(t(score^2)%*%(weight*window.matrix)^2)
  K.temp<-weight*t(weight*K)
  
  p<-rep(NA,length(Q))
  for(i in 1:length(Q)){
    #print(i)
    temp<-K.temp[window.matrix[,i]!=0,window.matrix[,i]!=0]
    if(sum(temp^2)==0){p[i]<-NA;next}
    
    lambda=eigen(temp,symmetric=T,only.values=T)$values
    if(sum(is.na(lambda))!=0){p[i]<-NA;next}
    
    temp.p<-SKAT_davies(Q[i],lambda,acc=10^(-6))$Qq
    
    if(temp.p > 1 || temp.p <= 0 ){
      temp.p<-Get_Liu_PVal.MOD.Lambda(Q[i],lambda)
    }
    p[i]<-temp.p
  }
  
  return(as.matrix(p))
}



#percentage notation
percent <- function(x, digits = 3, format = "f", ...) {
  paste0(formatC(100 * x, format = format, digits = digits, ...), "%")
}

sparse.cor <- function(x){
  n <- nrow(x)
  cMeans <- colMeans(x)
  covmat <- (as.matrix(crossprod(x)) - n*tcrossprod(cMeans))/(n-1)
  sdvec <- sqrt(diag(covmat)) 
  cormat <- covmat/tcrossprod(sdvec)
  list(cov=covmat,cor=cormat)
}

sparse.cov.cross <- function(x,y){
  n <- nrow(x)
  cMeans.x <- colMeans(x);cMeans.y <- colMeans(y)
  covmat <- (as.matrix(crossprod(x,y)) - n*tcrossprod(cMeans.x,cMeans.y))/(n-1)
  list(cov=covmat)
}




max.nth<-function(x,n){return(sort(x,partial=length(x)-(n-1))[length(x)-(n-1)])}

Get.p.base<-function(X,result.prelim){
  #X<-Matrix(X)
  mu<-result.prelim$nullglm$fitted.values;Y.res<-result.prelim$Y-mu
  outcome<-result.prelim$out_type
  if(outcome=='D'){v<-mu*(1-mu)}else{v<-rep(as.numeric(var(Y.res)),nrow(X))}
  A<-(t(X)%*%Y.res)^2
  B<-colSums(v*X^2)
  C<-t(X)%*%(v*result.prelim$X0)%*%result.prelim$inv.X0
  D<-t(t(result.prelim$X0)%*%as.matrix(v*X))
  p<-pchisq(as.numeric(A/(B-rowSums(C*D))),df=1,lower.tail=F)                     
  #p<-pchisq(as.numeric((t(X)%*%Y.res)^2/(apply(X*(v*X),2,sum)-apply(t(X)%*%(v*result.prelim$X0)%*%result.prelim$inv.X0*t(t(result.prelim$X0)%*%as.matrix(v*X)),1,sum))),df=1,lower.tail=F)
  #p[is.na(p)]<-NA
  return(as.matrix(p))
}

Get.p<-function(X,result.prelim){
  #X<-as.matrix(X)
  mu<-result.prelim$nullglm$fitted.values;Y.res<-result.prelim$Y-mu
  outcome<-result.prelim$out_type
  if(outcome=='D'){
    p<-ScoreTest_SPA(t(X),result.prelim$Y,result.prelim$X,method=c("fastSPA"),minmac=-Inf)$p.value
  }else{
    v<-rep(as.numeric(var(Y.res)),nrow(X))
    A<-(t(X)%*%Y.res)^2
    B<-colSums(v*X^2)
    C<-t(X)%*%(v*result.prelim$X0)%*%result.prelim$inv.X0
    D<-t(t(result.prelim$X0)%*%as.matrix(v*X))
    p<-pchisq(as.numeric(A/(B-rowSums(C*D))),df=1,lower.tail=F)                     
    #p<-pchisq(as.numeric((t(X)%*%Y.res)^2/(apply(X*(v*X),2,sum)-apply(t(X)%*%(v*result.prelim$X0)%*%result.prelim$inv.X0*t(t(result.prelim$X0)%*%as.matrix(v*X)),1,sum))),df=1,lower.tail=F)
  }
  return(as.matrix(p))
}

MK.statistic<-function (T_0,T_k,method='median'){
  T_0<-as.matrix(T_0);T_k<-as.matrix(T_k)
  T.temp<-cbind(T_0,T_k)
  T.temp[is.na(T.temp)]<-0
  
  which.max.alt<-function(x){
    temp.index<-which(x==max(x))
    if(length(temp.index)!=1){return(temp.index[2])}else{return(temp.index[1])}
  }
  kappa<-apply(T.temp,1,which.max.alt)-1
  
  if(method=='max'){tau<-apply(T.temp,1,max)-apply(T.temp,1,max.nth,n=2)}
  if(method=='median'){
    Get.OtherMedian<-function(x){median(x[-which.max(x)])}
    tau<-apply(T.temp,1,max)-apply(T.temp,1,Get.OtherMedian)
  }
  return(cbind(kappa,tau))
}

MK.threshold.byStat<-function (kappa,tau,M,fdr = 0.1,Rej.Bound=10000){
  b<-order(tau,decreasing=T)
  c_0<-kappa[b]==0
  ratio<-c();temp_0<-0
  for(i in 1:length(b)){
    #if(i==1){temp_0=c_0[i]}
    temp_0<-temp_0+c_0[i]
    temp_1<-i-temp_0
    temp_ratio<-(1/M+1/M*temp_1)/max(1,temp_0)
    ratio<-c(ratio,temp_ratio)
    if(i>Rej.Bound){break}
  }
  ok<-which(ratio<=fdr)
  if(length(ok)>0){
    #ok<-ok[which(ok-ok[1]:(ok[1]+length(ok)-1)<=0)]
    return(tau[b][ok[length(ok)]])
  }else{return(Inf)}
}

MK.threshold<-function (T_0,T_k, fdr = 0.1,method='median',Rej.Bound=10000){
  stat<-MK.statistic(T_0,T_k,method=method)
  kappa<-stat[,1];tau<-stat[,2]
  t<-MK.threshold.byStat(kappa,tau,M=ncol(T_k),fdr=fdr,Rej.Bound=Rej.Bound)
  return(t)
}


MK.q.byStat<-function (kappa,tau,M,Rej.Bound=10000){
  b<-order(tau,decreasing=T)
  c_0<-kappa[b]==0
  #calculate ratios for top Rej.Bound tau values
  ratio<-c();temp_0<-0
  for(i in 1:length(b)){
    #if(i==1){temp_0=c_0[i]}
    temp_0<-temp_0+c_0[i]
    temp_1<-i-temp_0
    temp_ratio<-(1/M+1/M*temp_1)/max(1,temp_0)
    ratio<-c(ratio,temp_ratio)
    if(i>Rej.Bound){break}
  }
  #calculate q values for top Rej.Bound values
  q<-rep(1,length(tau));index_bound<-max(which(tau[b]>0))
  for(i in 1:length(b)){
    temp.index<-i:min(length(b),Rej.Bound,index_bound)
    if(length(temp.index)==0){next}
    q[b[i]]<-min(ratio[temp.index])*c_0[i]+1-c_0[i]
    if(i>Rej.Bound){break}
  }
  return(q)
}



Get.cauchy<-function(p){
  p[p>0.99]<-0.99
  is.small<-(p<1e-16) & !is.na(p)
  is.regular<-(p>=1e-16) & !is.na(p)
  temp<-rep(NA,length(p))
  temp[is.small]<-1/p[is.small]/pi
  temp[is.regular]<-as.numeric(tan((0.5-p[is.regular])*pi))
  
  cct.stat<-mean(temp,na.rm=T)
  if(is.na(cct.stat)){return(NA)}
  if(cct.stat>1e+15){return((1/cct.stat)/pi)}else{
    return(1-pcauchy(cct.stat))
  }
}

Get.cauchy.scan<-function(p,window.matrix){
  p[p>0.99]<-0.99
  is.small<-(p<1e-16) & !is.na(p)
  temp<-rep(0,length(p))
  temp[is.small]<-1/p[is.small]/pi
  temp[!is.small]<-as.numeric(tan((0.5-p[!is.small])*pi))
  #window.matrix.MAC10<-(MAC>=10)*window.matrix0
  
  cct.stat<-as.numeric(t(temp)%*%window.matrix/apply(window.matrix,2,sum))
  #cct.stat<-as.numeric(t(temp)%*%window.matrix.MAC10/apply(window.matrix.MAC10,2,sum))
  is.large<-cct.stat>1e+15 & !is.na(cct.stat)
  is.regular<-cct.stat<=1e+15 & !is.na(cct.stat)
  pval<-rep(NA,length(cct.stat))
  pval[is.large]<-(1/cct.stat[is.large])/pi
  pval[is.regular]<-1-pcauchy(cct.stat[is.regular])
  return(pval)
}

Get.p.moment<-function(Q,re.Q){ #Q a A*q matrix of test statistics, re.Q a B*q matrix of resampled test statistics
  re.mean<-apply(re.Q,2,mean)
  re.variance<-apply(re.Q,2,var)
  re.kurtosis<-apply((t(re.Q)-re.mean)^4,1,mean)/re.variance^2-3
  re.df<-(re.kurtosis>0)*12/re.kurtosis+(re.kurtosis<=0)*100000
  re.p<-t(pchisq((t(Q)-re.mean)*sqrt(2*re.df)/sqrt(re.variance)+re.df,re.df,lower.tail=F))
  #re.p[re.p==1]<-0.99
  return(re.p)
}


Impute<-function(Z, impute.method){
  p<-dim(Z)[2]
  if(impute.method =="random"){
    for(i in 1:p){
      IDX<-which(is.na(Z[,i]))
      if(length(IDX) > 0){
        maf1<-mean(Z[-IDX,i])/2
        Z[IDX,i]<-rbinom(length(IDX),2,maf1)
      }
    }
  } else if(impute.method =="fixed"){
    for(i in 1:p){
      IDX<-which(is.na(Z[,i]))
      if(length(IDX) > 0){
        maf1<-mean(Z[-IDX,i])/2
        Z[IDX,i]<-2 * maf1
      }
    }
  } else if(impute.method =="bestguess") {
    for(i in 1:p){
      IDX<-which(is.na(Z[,i]))
      if(length(IDX) > 0){
        maf1<-mean(Z[-IDX,i])/2
        Z[IDX,i]<-round(2 * maf1)
      }
    }
  } else {
    stop("Error: Imputation method shoud be \"fixed\", \"random\" or \"bestguess\" ")
  }
  return(as.matrix(Z))
}


