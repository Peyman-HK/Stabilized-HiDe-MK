library(knockoff)
library(data.table)
library(glmnet)

`+` <- function(e1, e2) {
  if (is.character(e1) | is.character(e2)) {
    paste0(e1, e2)
  } else {
    base::`+`(e1, e2)
  }
}

source('/home/users/peymanhk/PROJECT_2/Simulated/GOI/KnockoffScreen_AL_vMar16-updated.R')

num_knock = 5 

path_data = 'C:/users/hosse/'
path_data = 'C:/Users/hosse/OneDrive/Desktop/De-randomized-HiDe-MK-main/Simulation data/class/'
indx <- 1

prt2 = 'FI_Class_' + indx + '.csv'
prt3 = 'Y_'+ indx +'.csv'
prt4 = 'Beta_'+ indx +'.csv'
path1 = path_data + prt2
path2 = path_data + prt3
path3 = path_data + prt4

W_all <- read.csv(path1, header = FALSE, check.names = TRUE, as.is=TRUE, sep = ",")
y <- read.csv(path2, header = FALSE, check.names = TRUE, as.is=TRUE, sep = ",")
Beta <- read.csv(path3, header = FALSE, check.names = TRUE, as.is=TRUE, sep = ",")

Feat_import <- data.matrix(W_all, rownames.force = NA)

Beta <- as.numeric(Beta$V1)
Temp = which( Beta!= 0, arr.ind=TRUE)
NZ_True = Temp
#print(NZ_True)
p = NROW(Beta)

Size_Orig = NROW(Feat_import)/(num_knock+1);
print(Size_Orig)

#####################################################################################

##############       		   Calculate FDR and Power  	       ##################

#####################################################################################


MK.threshold.byStat<-function (kappa,tau,M,fdr = val,Rej.Bound=10000){
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

MK.statistic<-function (T_0,T_k,method='median'){
  T_0<-as.matrix(T_0);T_k<-as.matrix(T_k)
  T.temp<-cbind(T_0,T_k)
  T.temp[is.na(T.temp)]<-0
  kappa<-apply(T.temp,1,which.max)-1
  if(method=='max'){tau<-apply(T.temp,1,max)-apply(T.temp,1,max.nth,n=2)}
  if(method=='median'){
    Get.OtherMedian<-function(x){median(x[-which.max(x)])}
    tau<-apply(T.temp,1,max)-apply(T.temp,1,Get.OtherMedian)
  }
  KS.stat<-cbind(kappa,tau)
  return(KS.stat)
}

MK.threshold<-function (T_0,T_k, fdr = val,method='median',Rej.Bound=10000){
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
  q<-rep(1,length(tau))
  for(i in 1:length(b)){
    q[b[i]]<-min(ratio[i:min(length(b),Rej.Bound)])*c_0[i]+1-c_0[i]
    if(i>Rej.Bound){break}
  }
  return(q)
}

###########################

T_0 = Feat_import[1:Size_Orig,]
Temp = Size_Orig+1
T_k = t(Feat_import[Temp:NROW(Feat_import),])
T_k <- t(matrix(T_k, nrow  = 5, byrow = TRUE))
#Diff = Size_Orig - p;
#T_0 = T_0[1:p]
#T_k = T_k[1:p,1:5]

T_0 = abs(T_0); 
T_k = abs(T_k); 


Kappa_Tau = MK.statistic(T_0,T_k,method='median')
print(Kappa_Tau)

Kappa = Kappa_Tau[,1]
Tau   = Kappa_Tau[,2]

M = 5
Target_FDR <-seq(0.01, 0.2, by=0.01) 

count <- 0
FDR <- numeric(length = length(Target_FDR))
Power <- numeric(length = length(Target_FDR))
q_val =MK.q.byStat(Kappa,Tau,M,Rej.Bound=10000)
print(q_val)

for (val in Target_FDR) {
  count = count+1
  Inds = q_val < val
  Temp = which( Inds!= FALSE, arr.ind=TRUE)
  NZ_Sel = Temp
  Total_sel = NROW(NZ_Sel)
  
  FDR[count] = sum(Beta[NZ_Sel] == 0)/max(1,Total_sel)
  Power[count] = sum(Beta[NZ_Sel] != 0)/sum(Beta != 0)
}

print("Multiple Knockoff")
print(FDR)
print(Power)

new_lineJ = t(c(FDR,Power))

Destination <- path_data + 'Stat_knockoff' + '.csv'
if (file.exists(Destination) == FALSE) 
  file.create(file=Destination)

write.table(new_lineJ,file=Destination, append=TRUE, sep=',', eol="\n", col.names=F, row.names=F)
Stat <- read.csv(file=Destination, header = FALSE, check.names = TRUE, as.is=TRUE, sep = ",")
colMeans(Stat)
