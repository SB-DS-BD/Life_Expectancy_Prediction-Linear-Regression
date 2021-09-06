# ----- Linear Regression Modelling Project ----- #

#------------------->Basic Functional Form:
#Y=a0+a1X1+ a2X2+a3X3+..................aNXN+e, 
#Y=Dependent Var
#X1,X2,.......XN=Independent Var
#e= Error Term 


#---------------Problem Statement: 
# In this case study we got average life expectancy of people of 193 Countries. 
# We have to predict next year value using linear regression.

#-------Preparing the environment for MLRM------------#

list.of.packages <- c("boot", "car","QuantPsyc","lmtest","sandwich","vars","nortest","MASS","caTools")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos="http://cran.rstudio.com/")

library(boot) 
library(car)
library(QuantPsyc)
library(lmtest)
library(sandwich)
library(vars)
library(nortest)
library(Hmisc)
library(MASS)
library(caTools)
library(pastecs)
library(caret)
library(dplyr)

#--------Setting the Working Directory-----------#
Path<-"E:/IVY/Stat+R/Project/Linear reg"
setwd(Path)
getwd()

data=read.csv("Life_Expectancy_Data.csv")
data1=data #To create a backup of original data


#-------------Basic Exploration of the data------# 
str(data1) # 2938 obs. of  23 variables
summary(data1)
dim(data1)

#-------- making Year and Status as a categorical column (factor variable)
data1$Year<- as.factor(data1$Year)
data1$Status<- as.factor(data1$Status)

#Renaming four columns
colnames(data1)[12] <- "Under_five_Deaths"
colnames(data1)[16] <- "HIV_AIDS"
colnames(data1)[20] <- "Thinness1to19Years"
colnames(data1)[21] <- "Thinness5to9Years"

#------>Missing values Identification 
as.data.frame(colSums(is.na(data1)))

# Treatment
# checking the pattern of the dependent variable
hist.data.frame(data1["Life_Expectancy"])

#Imputing the values with the mean value of the series
data1[is.na(data1$Life_Expectancy),"Life_Expectancy"]=mean(data1$Life_Expectancy,na.rm=T)

# checking and showing that not any null values are present 
as.data.frame(colSums(is.na(data1)))

# checking distribution again
hist.data.frame(data1["Life_Expectancy"])

#----->Outlier checking
# boxplot
boxplot(data1$Life_Expectancy, horizontal = T)

#-----Treatment through quantile method

quantile(data1$Life_Expectancy,c(0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,
                                 0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99,1))

data1$Life_Expectancy<-ifelse(data1$Life_Expectancy>82,
                              82,data1$Life_Expectancy) # replace by capping

data1$Life_Expectancy<-ifelse(data1$Life_Expectancy<51.4,
                              51.4,data1$Life_Expectancy) # replace by flooring 


#Checking again
quantile(data1$Life_Expectancy,c(0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,
                                 0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99,1))

# checking boxplot again
boxplot(data1$Life_Expectancy, horizontal = T)

# statistics
stat.desc(data1)

#ANOVA Test
summary(aov(Life_Expectancy ~ data1$Year, data=data1)) # year col
summary(aov(Life_Expectancy ~ data1$Status, data=data1)) # status col
# both categorical col has good p values

# ---> EDA done....
# data1 is the final dataset for modeling

########### Model Building ###########
#------------Splitting the data into training and test data set
set.seed(123)#This is used to produce reproducible results, everytime we run the model

spl = sample.split(data1$Life_Expectancy, 0.7)
#Splits the overall data into train and test data in 70:30 ratio

original.data = subset(data1, spl == TRUE)
str(original.data)
dim(original.data)

test.data = subset(data1, spl == FALSE)
str(test.data)
dim(test.data)

# omit NA
original.data <- na.omit(original.data)
test.data <- na.omit(test.data)

#------------Fitting the model----------#
#Iteration.1 We start with testing all variables
options(scipen = 999)

LinearModel1=lm(Life_Expectancy~.,data=original.data)#'.', refers to all variables
summary(LinearModel1)

# dropping status that has no impact and the redundant variable country
#Iteration.2.

LinearModel2=lm(Life_Expectancy~	Year+	Adult_Mortality+	Infant_Deaths+	Alcohol+	
                  Percentage_Expenditure+	Hepatitis_B+	Measles+	BMI+	Under_five_Deaths+	Polio+	
                  Total_Expenditure+	Diphtheria+	HIV_AIDS+	GDP+	Per_Capita_GDP+ Population+	
                  Thinness1to19Years+	Thinness5to9Years+	Income_Composition_of_Resources+	Schooling,
                data=original.data)

summary(LinearModel2)

# dropping one by one
#Iteration.3.

LinearModel3= lm(Life_Expectancy~ 
                   I(Year==2007)+ I(Year==2008)+ I(Year==2009)+ 
                   I(Year==2010)+ I(Year==2011)+	I(Year==2012)+ I(Year==2013)+ I(Year==2014)+ 
                   Adult_Mortality +	BMI+	Under_five_Deaths+	Polio+	Total_Expenditure+	Diphtheria+	
                   HIV_AIDS+	Per_Capita_GDP+	Thinness1to19Years+	Income_Composition_of_Resources, 
                 data=original.data)

summary(LinearModel3)
# LinearModel3 is the best model till now.

# checking vif 
vif(LinearModel3)
# all good
# LinearModel3 is good from vif

# Get the predicted or fitted values (model3)
fitted(LinearModel3)
#par(mfrow=c(2,2))
par(mar=c(1,1,1,1))
plot(LinearModel3)

# Graphical display of significant variables
# scatter
ggplot(original.data) + geom_point(aes(x = Life_Expectancy, y = Year))
ggplot(original.data) + geom_point(aes(x = Life_Expectancy, y = Adult_Mortality))
ggplot(original.data) + geom_point(aes(x = Life_Expectancy, y = BMI))
ggplot(original.data) + geom_point(aes(x = Life_Expectancy, y = Under_five_Deaths))
ggplot(original.data) + geom_point(aes(x = Life_Expectancy, y = Polio))
ggplot(original.data) + geom_point(aes(x = Life_Expectancy, y = Total_Expenditure))
ggplot(original.data) + geom_point(aes(x = Life_Expectancy, y = Diphtheria))
ggplot(original.data) + geom_point(aes(x = Life_Expectancy, y = HIV_AIDS))
ggplot(original.data) + geom_point(aes(x = Life_Expectancy, y = Per_Capita_GDP))
ggplot(original.data) + geom_point(aes(x = Life_Expectancy, y = Thinness1to19Years))
ggplot(original.data) + geom_point(aes(x = Life_Expectancy, y = Income_Composition_of_Resources))


# MAPE
original.data$pred <- fitted(LinearModel3)
Actual_Pred<-select(original.data,c(Life_Expectancy,pred))
Actual_Pred$error<-Actual_Pred$Life_Expectancy-Actual_Pred$pred
write.csv(original.data,"mape3.csv", row.names = F)

#Calculating MAPE
attach(original.data)
MAPE<-print((sum((abs(Life_Expectancy-pred))/Life_Expectancy))/nrow(original.data))
# 4%

durbinWatsonTest(LinearModel3) # autocorelation
#Since, the p-value is <0.05, we reject H0. positive autocorrelation.

# Checking multicollinearity
vif(LinearModel3) # should be within 2. If it is greater than 10 then serious problem

# Homoscedasticity test
bptest(LinearModel3) # null hypothesis accepted and Homoscedasticity exists 

# normality test
resids <- LinearModel3$residuals
ad.test(resids) #get Anderson-Darling test for normality 
cvm.test(resids) #get Cramer-von Mises test for normaility 
lillie.test(resids) #get Lilliefors (Kolmogorov-Smirnov) test for normality 
pearson.test(resids) #get Pearson chi-square test for normaility 
sf.test(resids) #get Shapiro-Francia test for normaility 
qqnorm(resids)

# Variable Importance of the model
varImp(LinearModel3)

# Make predictions
predictions <- LinearModel3 %>% predict(test.data)
# Model performance

# (a) Prediction error, RMSE
RMSE(predictions, test.data$Life_Expectancy)

# (b) R-square
R2(predictions, test.data$Life_Expectancy)

# (c) Adj. R-square
summary(LinearModel3)$adj.r.squared

# coefficient
summary(LinearModel3)$coef # Interpret

###################################