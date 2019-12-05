library(caretEnsemble)
library(caret)
library(randomForest)
library(mlbench)
library(e1071)
library(klaR)
library(pROC)
library(kernlab)
library(neuralnet)
library(nnet)

#please set the working directory of the data here
#1 stratified sampling
LSelcdata_re=read.csv("Lselecdata_IV.csv")
LSelcdata_re=LSelcdata_re[,-1]
str(LSelcdata_re)
set.seed(800)
in_train=createDataPartition(LSelcdata_re$Class,p=0.7,list=FALSE)
trainset=LSelcdata_re[in_train,]
testset=LSelcdata_re[-in_train,]
table(trainset$Class)
table(testset$Class)
prop.table(table(trainset$Class))

#2 individual-------
#2.1 nb
m.nb=naiveBayes(trainset[,-10],trainset$Class,laplace = 1)

#training
pre.nb_train=predict(m.nb,trainset[,-10])
Matrix.nb_train=confusionMatrix(pre.nb_train,trainset[,10],positive="landslide")
Matrix.nb_train
probs.nb_train=predict(m.nb,trainset[,-10],type="raw")
probs.nb_train
nb.ROC_train=roc(trainset[,10],probs.nb_train[,1])
auc(nb.ROC_train)
ci.auc(nb.ROC_train)
plot(nb.ROC_train,main="nb.ROC_train",col="blue",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(nb.ROC_train,col="blue",lwd=2,xaxs="i",grid=c(0.1,0.1),legacy.axes=TRUE)

#validation
pre.nb=predict(m.nb,testset[,-10])
Matrix.nb=confusionMatrix(pre.nb,testset[,10],positive="landslide")
Matrix.nb
probs.nb=predict(m.nb,testset[,-10],type="raw")
probs.nb
nb.ROC=roc(testset[,10],probs.nb[,1])
nb.ROC
auc(nb.ROC)
ci.auc(nb.ROC)
plot(nb.ROC,main="nb.ROC",col="blue",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(nb.ROC,col="blue",lwd=2,xaxs="i",grid=c(0.1,0.1),legacy.axes=TRUE)
#2.2svm
?ksvm()
set.seed(200)
m.svm=ksvm(Class~., data=trainset,kernel="rbfdot",prob.model=TRUE,C=0.15)
m.svm
m.svm$finalModel
#training
pre.svm_train=predict(m.svm,trainset[,-10])
Matrix.svm_train=confusionMatrix(pre.svm_train,trainset[,10],positive="landslide")
Matrix.svm_train
probs.svm_train=predict(m.svm,trainset[,-10],type="prob")
probs.svm_train
svm.ROC_train=roc(trainset[,10],probs.svm_train[,1])
auc(svm.ROC_train)
ci.auc(svm.ROC_train)
plot(svm.ROC_train,main="svm_train.ROC",col="black",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(svm.ROC_train,col="black",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)

#validation
pre.svm=predict(m.svm,testset[,-10])
Matrix.svm=confusionMatrix(pre.svm,testset[,10],positive="landslide")
Matrix.svm
probs.svm=predict(m.svm,testset[,-10],type="prob")
probs.svm
svm.ROC=roc(testset[,10],probs.svm[,1])
auc(svm.ROC)
ci.auc(svm.ROC)
plot(svm.ROC,main="svm.ROC",col="black",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(svm.ROC,col="black",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)


#2.3lr
control=trainControl("repeatedcv", number=10, 
                     repeats=3,savePredictions=TRUE, classProbs=TRUE)
set.seed(300)
m.glm=train(Class~., data=trainset,method="glm",trControl=control,metric="Kappa")
m.glm

#training
pre.glm_train=predict(m.glm,trainset[,-10])
Matrix.glm_train=confusionMatrix(pre.glm_train,trainset[,10],positive="landslide")
Matrix.glm_train
probs.glm_train=predict(m.glm,trainset[,-10],type="prob")
probs.glm_train
glm.ROC_train=roc(trainset[,10],probs.glm_train[,1])
auc(glm.ROC_train)
ci.auc(glm.ROC_train)
plot(glm.ROC_train,main="glm.ROC_train",col="darkgoldenrod",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(glm.ROC_train,col="darkgoldenrod",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)
#validation
pre.glm=predict(m.glm,testset[,-10])
Matrix.glm=confusionMatrix(pre.glm,testset[,10],positive="landslide")
Matrix.glm
probs.glm=predict(m.glm,testset[,-10],type="prob")
probs.glm
glm.ROC=roc(testset[,10],probs.glm[,1])
auc(glm.ROC)
ci.auc(glm.ROC)
plot(glm.ROC,main="glm.ROC",col="darkgoldenrod",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(glm.ROC,col="darkgoldenrod",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)

#2.4ann
?nnet()
set.seed(40)
m.nnet=nnet(Class~., data=trainset,size=2,decay=0.001,maxit=30)
m.nnet

#training
pre.nnet_train=predict(m.nnet,trainset[,-10],type="class")
str(pre.nnet_train)
Matrix.nnet_train=confusionMatrix(as.factor(pre.nnet_train),trainset[,10],positive="landslide")
Matrix.nnet_train
probs.nnet_train=predict(m.nnet,trainset[,-10],type="raw")
probs.nnet_train
nnet.ROC_train=roc(trainset[,10],probs.nnet_train[,1])
auc(nnet.ROC_train)
ci.auc(nnet.ROC_train)
plot(nnet.ROC_train,main="nnet.ROC_train",col="cyan",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(nnet.ROC_train,col="cyan",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)
#validation
pre.nnet=predict(m.nnet,testset[,-10],type="class")
Matrix.nnet=confusionMatrix(as.factor(pre.nnet),testset[,10],positive="landslide")
Matrix.nnet
probs.nnet=predict(m.nnet,testset[,-10],type="raw")
probs.nnet
nnet.ROC=roc(testset[,10],probs.nnet[,1])
auc(nnet.ROC)
ci.auc(nnet.ROC)
plot(nnet.ROC,main="nnet.ROC",col="cyan",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(nnet.ROC,col="cyan",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)


#3 stacking-------
control=trainControl("repeatedcv", number=10, 
                     repeats=3,savePredictions=TRUE, classProbs=TRUE)


#3.1 stacking(all)
algorithmList3 = c("nb","svmRadial","glm","nnet")

set.seed(30)
?caretList()
models3 =caretList(Class~., data=trainset,trControl=control,
                   methodList=algorithmList3)
results3=resamples(models3)
modelCor(results3)
splom(results3)
parallelplot(results3)
summary(diff(results3))
?resamples()
?modelCor()
summary(results3)
stackControl3 = trainControl(method="repeatedcv", number=10, repeats=3, savePredictions="final", classProbs=TRUE)
set.seed(300)
stack.glm3=caretStack(models3, method="glm", metric="Accuracy", trControl=stackControl3)
print(stack.glm3)

#training
pre.stack3_train=predict(stack.glm3,trainset[,-10])
pre.stack3_train
Matrix.stack3_train=confusionMatrix(pre.stack3_train,trainset[,10],positive="landslide")
Matrix.stack3_train
probs.stack3_train=predict(stack.glm3,trainset[,-10],type="prob")
probs.stack3_train
stack.ROC3_train=roc(trainset[,10],probs.stack3_train)
auc(stack.ROC3_train)
ci.auc(stack.ROC3_train)
plot(stack.ROC3_train,main="SALN_train.ROC",col="red",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(stack.ROC3_train,col="red",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)

#validation
pre.stack3=predict(stack.glm3,testset[,-10])
pre.stack3
Matrix.stack3=confusionMatrix(pre.stack3,testset[,10],positive="landslide")
Matrix.stack3
probs.stack3=predict(stack.glm3,testset[,-10],type="prob")
probs.stack3
stack.ROC3=roc(testset[,10],probs.stack3)
auc(stack.ROC3)
ci.auc(stack.ROC3)
plot(stack.ROC3,main="SALN.ROC",col="red",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(stack.ROC3,col="red",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)


#3.2 stacking(without lr)
algorithmList4 = c("svmRadial","nb","nnet")
set.seed(350)
?caretList()
models4 =caretList(Class~., data=trainset,trControl=control,
                   methodList=algorithmList4)
results4=resamples(models4)
modelCor(results4)
summary(results4)
stackControl4 = trainControl(method="repeatedcv", number=10, repeats=3, savePredictions="final", classProbs=TRUE)
set.seed(300)
stack.glm4=caretStack(models4, method="glm", metric="Accuracy", trControl=stackControl4)
print(stack.glm4)


#training
pre.stack4_train=predict(stack.glm4,trainset[,-10])
pre.stack4_train
Matrix.stack4_train=confusionMatrix(pre.stack4_train,trainset[,10],positive="landslide")
Matrix.stack4_train
probs.stack4_train=predict(stack.glm4,trainset[,-10],type="prob")
probs.stack4_train
stack.ROC4_train=roc(trainset[,10],probs.stack4_train)
auc(stack.ROC4_train)
ci.auc(stack.ROC4_train)
plot(stack.ROC4_train,main="SAC.ROC_train",col="brown",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(stack.ROC4_train,col="brown",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)

#validation
pre.stack4=predict(stack.glm4,testset[,-10])
pre.stack4
Matrix.stack4=confusionMatrix(pre.stack4,testset[,10],positive="landslide")
Matrix.stack4
probs.stack4=predict(stack.glm4,testset[,-10],type="prob")
probs.stack4
stack.ROC4=roc(testset[,10],probs.stack4)
auc(stack.ROC4)
ci.auc(stack.ROC4)
plot(stack.ROC4,main="SAC.ROC",col="brown",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(stack.ROC4,col="brown",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)

#3.3 stacking(without lr and nb)
algorithmList5 = c("svmRadial","nnet")
set.seed(150)
?caretList()
models5 =caretList(Class~., data=trainset,trControl=control,
                   methodList=algorithmList5)
results5=resamples(models5)
modelCor(results5)
summary(results5)
stackControl5 = trainControl(method="repeatedcv", number=10, repeats=3, savePredictions="final", classProbs=TRUE)
set.seed(300)
stack.glm5=caretStack(models5, method="glm", metric="Accuracy", trControl=stackControl5)
print(stack.glm5)


#training
pre.stack5_train=predict(stack.glm5,trainset[,-10])
pre.stack5_train
Matrix.stack5_train=confusionMatrix(pre.stack5_train,trainset[,10],positive="landslide")
Matrix.stack5_train
probs.stack5_train=predict(stack.glm5,trainset[,-10],type="prob")
probs.stack5_train
stack.ROC5_train=roc(trainset[,10],probs.stack5_train)
auc(stack.ROC5_train)
ci.auc(stack.ROC5_train)
plot(stack.ROC5_train,main="SA.ROC_train",col="forestgreen",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(stack.ROC5_train,col="forestgreen",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)

#validation
pre.stack5=predict(stack.glm5,testset[,-10])
pre.stack5
Matrix.stack5=confusionMatrix(pre.stack5,testset[,10],positive="landslide")
Matrix.stack5
probs.stack5=predict(stack.glm5,testset[,-10],type="prob")
probs.stack5
stack.ROC5=roc(testset[,10],probs.stack5)
auc(stack.ROC5)
ci.auc(stack.ROC5)
plot(stack.ROC5,main="SA.ROC",col="forestgreen",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(stack.ROC5,col="forestgreen",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)



#3.4 stacking(without svm)
algorithmList = c("nb","glm","nnet")
set.seed(700)
?caretList()
models =caretList(Class~., data=trainset,trControl=control,
                  methodList=algorithmList)
results=resamples(models)
summary(results)
stackControl = trainControl(method="repeatedcv", number=10, repeats=3, savePredictions="final", classProbs=TRUE)
set.seed(300)
stack.glm=caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

#training
pre.stack_train=predict(stack.glm,trainset[,-10])
pre.stack_train
Matrix.stack_train=confusionMatrix(pre.stack_train,trainset[,10],positive="landslide")
Matrix.stack_train
probs.stack_train=predict(stack.glm,trainset[,-10],type="prob")
probs.stack_train
stack.ROC_train=roc(trainset[,10],probs.stack_train)
auc(stack.ROC_train)
ci.auc(stack.ROC_train)
plot(stack.ROC_train,main="SALCN_train.ROC",col="forestgreen",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(stack.ROC_train,col="forestgreen",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)

#validation
pre.stack=predict(stack.glm,testset[,-10])
pre.stack
Matrix.stack=confusionMatrix(pre.stack,testset[,10],positive="landslide")
Matrix.stack
probs.stack=predict(stack.glm,testset[,-10],type="prob")
probs.stack
stack.ROC=roc(testset[,10],probs.stack)
auc(stack.ROC)
ci.auc(stack.ROC)
plot(stack.ROC,main="SALCN.ROC",col="forestgreen",lwd=2,xaxs="i",legacy.axes=TRUE,print.auc=TRUE)
plot(stack.ROC,col="forestgreen",lwd=2,xaxs="i",legacy.axes=TRUE,add=TRUE)

