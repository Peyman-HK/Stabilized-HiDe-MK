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

source('/home/users/peymanhk/PROJECT_2/Simulated/GOI/KnockoffScreen_updated.R')

prt_result = '/home/users/peymanhk/PROJECT_2/Simulated/GOI/'


ML_Type = 'Class' # or 'Reg'

M = 5

path_data = 'C:/users/hosse/'
path_data = 'C:/Users/hosse/OneDrive/Desktop/De-randomized-HiDe-MK-main/Simulation data/class/'
indx <- 1

#############################
if (ML_Type == 'Class'){
prt2 = 'FI_Class_' + indx + '.csv'
} else {
prt2 = 'FI_Reg_' + indx + '.csv'
}
############################

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

Size_Orig = NROW(Feat_import)/(M + 1);
print(Size_Orig)

#####################################################################################

##############       		   Calculate FDR and Power  	       ##################

#####################################################################################

T_0 = Feat_import[1:Size_Orig,]
Temp = Size_Orig+1
T_k = t(Feat_import[Temp:NROW(Feat_import),])
T_k <- t(matrix(T_k, nrow  = M, byrow = TRUE))

T_0 = abs(T_0); 
T_k = abs(T_k); 


Kappa_Tau = MK.statistic(T_0,T_k,method='median')
print(Kappa_Tau)

Kappa = Kappa_Tau[,1]
Tau   = Kappa_Tau[,2]


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
if (ML_Type == 'Class'){
Destination <- prt_result + '/' + 'Stat_stabilize_HiDe_MK_Class.csv'
} else {
Destination <- prt_result + '/' + 'Stat_stabilize_HiDe_MK_Reg.csv'
}

if (file.exists(Destination) == FALSE) 
  file.create(file=Destination)

write.table(new_lineJ,file=Destination, append=TRUE, sep=',', eol="\n", col.names=F, row.names=F)
Stat <- read.csv(file=Destination, header = FALSE, check.names = TRUE, as.is=TRUE, sep = ",")
colMeans(Stat)
print("Done!!")

