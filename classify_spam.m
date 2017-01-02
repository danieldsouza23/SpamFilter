clear all; clc; close all;
%%%%
load spamdata.mat
 TEST_SPAM = A_spam_test;
 TEST_NOT_SPAM = A_notspam_test;
 r = rank(Atrain(:,:,1)); % Rank is same for Atrain(:,:,1) and Atrain(:,:,2)
 for k = 1:r    
     % For loop to create Projection matrix
    for i=1:2
        input = Atrain(:,:,i);
        [U,S,V]= svd(input);
         Uk = U(:,1:k);
         Ukdag=pinv(Uk);
         Proj= Uk * Ukdag;
         P(:,:,i) = Proj;
    end
    % For loop to create SQERR for each of the Test Data Sequences
    for i = 1:2
       SQERR1(i,:) = sum((TEST_SPAM-squeeze(P(:,:,i))*TEST_SPAM).^2,1); %Sum of Squared Errors for TEST_SPAM vectors
       SQERR2(i,:) = sum((TEST_NOT_SPAM-squeeze(P(:,:,i))*TEST_NOT_SPAM).^2,1); %Sum of Squared Errors for TEST_SPAM vectors
    end
    
    [junk1,num1] = min(SQERR1,[],1); % Find Min Error for SQERR1
    [junk2,num2] = min(SQERR2,[],1); % Find Min Error for SQERR2

    spam_num1 = num1-1; % 0 for non-SPAM, 1 for SPAM. Expecting all 1's 
    spam_num2 = num2-1; % 0 for non-SPAM, 1 for SPAM. Expecting all 0's
   
    spam_num1_db(:,k) = spam_num1;
    spam_num2_db(:,k) = spam_num2;
    
    Pdk = sum(spam_num1)/906; %(Spam Correctly Classified)
    Pdf = sum(spam_num2)/906; %(Not spam incorrectly classifed as spam)

    plotpdkx(:,k) = Pdk; % Creating an array of values of Pdk for all k
    plotpdfx(:,k) = Pdf; % Creating an array of values of Pdf for all k
    basek(:,k) = k;      % Creating the 'k' x axis.

 end
 hold on 
 
%plot(basek,plotpdkx,'-ro')    % Plot of Pdk versus K
plot(basek,plotpdfx,'-gv')    % Plot of Pdf versus K
%plot(plotpdkx,plotpdfx,'-bx') % Plot of Pdk versus Pdf
%hold off
 
 %% Implement a mean-based classification

which_digits1 = spam_num1_db;
which_digits2 = spam_num2_db;

correct1 = sum(sum(which_digits1));
Pcorrect_mean1 = correct1/(906*k);

correct2 = sum(sum(which_digits2));
Pcorrect_mean2 = correct2/(906*k);

%hold on
%% For Plot Pd(k) vs k
%plot(basek,Pcorrect_mean1*ones(size(basek)),'red-','LineWidth',2)
%xlabel('k'), ylabel('Probability of Spam Correctly Classfied')
%% For Plot Pf(k) vs k
%plot(basek,Pcorrect_mean2*ones(size(basek)),'green-','LineWidth',2)
xlabel('k'), ylabel('Probability of NOT Spam incorrectly classfied as Spam')
%% For Plot Pd(k) vs Pf(k)
%xlabel('Pd(k)'), ylabel('Pf(k)')
hold off
%grid on

 
 %% Recursively finding the K such that Pf(k) <= 0.15 and Pd(k) is maximum
max = 0;
maxindex = -1;
for k = 1:r
    if(plotpdfx(k) <= 0.15) 
        if(plotpdkx(k) > max) 
            maxindex = k;
            max = plotpdkx(k);
        end
    end
end
kopt = maxindex;

   
        
    