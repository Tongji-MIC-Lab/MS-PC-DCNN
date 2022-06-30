clear;
clc;

[testimgpath,testlabels]=textread('/home/tj/tpj/mltask/cifar10/cifar10_test.txt','%s %d');

multi_file='/home/tj/tpj/mltask/cifar10/cifar10_double_alex_nopre1_multiscale/';

scale32=load([multi_file,'average32']);
scale32=scale32.average32;

scale48=load([multi_file,'average48']);
scale48=scale48.average48;

scale64=load([multi_file,'average64']);
scale64=scale64.average64;

scale80=load([multi_file,'average80']);
scale80=scale80.average80;

average=(scale32+scale48+scale64+scale80)/4;

[m,n]=size(average);
correctnum1=0;
correctnum2=0;

for i=1:m
    [predicevalue1(i,1),predictlabel1(i,1)]=max(average(i,:));
    average(i,predictlabel1(i,1))=min(average(i,:));
    
    [predicevalue2(i,1),predictlabel2(i,1)]=max(average(i,:));
    average(i,predictlabel2(i,1))=min(average(i,:));
    
    [predicevalue3(i,1),predictlabel3(i,1)]=max(average(i,:));
    average(i,predictlabel3(i,1))=min(average(i,:));
    
    [predicevalue4(i,1),predictlabel4(i,1)]=max(average(i,:));
    average(i,predictlabel4(i,1))=min(average(i,:));
    
    [predicevalue5(i,1),predictlabel5(i,1)]=max(average(i,:));
    average(i,predictlabel5(i,1))=min(average(i,:));
    
    
end

for i=1:m
    if (predictlabel1(i,1)-1)==testlabels(i,1)
        correctnum1=correctnum1+1;
    end
end

for i=1:m
    if (predictlabel1(i,1)-1)==testlabels(i,1) || (predictlabel2(i,1)-1)==testlabels(i,1) || (predictlabel3(i,1)-1)==testlabels(i,1) || (predictlabel4(i,1)-1)==testlabels(i,1) || (predictlabel5(i,1)-1)==testlabels(i,1)
        correctnum2=correctnum2+1;
    end
end

accuracy1=correctnum1/m;
accuracy2=correctnum2/m;
