# ENG201- Genre Analysis Project
## Jeffrey Cruz
#
#
#
#### Project Description:

Jeffrey Cruz
3/26/18
MAT 222: Group Project Write Up

 
 

The dataset given had 2340 rows and 82 columns, representing 2340 houses and 82 features of those houses. 

  ![alt text](https://i.imgur.com/A45ctsP.png "Seaborns Plot")


For this assignment we were to build a linear regression model for predicting the sale prices of new houses using only 5-15 of these variables. The process of picking which of the 82 variables to use is called features selection. In order to do my feature selection, I decided to fit the entire dataset to a Random Forest regression model. Fitting the dataset to a Random Forest model would be very helpful in terms of feature selection due to their innate interpretably (Random Forest models have a built in method for calculating feature importance). In addition, in order to increase the accuracy of my predictions, I normalized the output variable (Sale price) by taking it’s natural logarithm.


  
Distribution of output variable before normalization (above) and after normalization (below) and their QQ plots to show normality

 ![alt text](https://image.prntscr.com/image/JKqgfVHCSDO3bkjn9DQ8lw.png "Seaborns Plot")
 
  ![alt text](https://i.imgur.com/3K5eJht.png "Seaborns Plot")
 
 ![alt text](https://image.prntscr.com/image/qux_OJ9KT66MTf2UJ6qV4w.png "Seaborns Plot")
 
  ![alt text](https://i.imgur.com/qolPYaY.png "Seaborns Plot")


After performing all data preprocessing and fitting the dataset to a random forest model, I used it’s importance function to get the top 14 predictive variables which ended up being: 
1.	Overall.Qual
2.	Gr.Liv.Area
3.	Garage.Cars
4.	Total.Bsmt.SF
5.	1st.Flr.SF
6.	Lot.Area
7.	Year.Built
8.	BsmtFin.SF.1
9.	Year.Remod.Add
10.	Garage.Yr.Blt
11.	Overall.Cond
12.	Central.Air
13.	Bsmt.Qual
14.	Bsmt.Unf.SF

HeatMap Correlation Matrix:

  ![alt text](https://i.imgur.com/PWz5HB4.png "Seaborns Plot")


I also normalize these variables (except for Central Air and Bsmt Qual) by subtracting their mean and dividing by their standard deviation, this is done to get more uniform coefficients for the linear regression model. For Central Air I treat Y as 2, and N as 1, and for Bsmt Qual I do (TA = 6, PO = 5, GD = 4, FA = 3, EX = 2, Blank = 1, NA = 0), After fitting the normalized dataset of these 14 variables to a linear regression model, I get this graph: 

 
 ![alt text](https://i.imgur.com/WeSRGBm.png "Residuals")

 
There are two outliers where my linear regression model made poor predictions. This can be for various reasons (I’m not using a variable that are crucial to those two houses, or there were special circumstances for those houses that are not present in the data like a discounted sale). But overall most of my predictions were centered around 0, within 0.5 residuals. (Again, this model is predicting ln(saleprice), not the sale price itself. To get back the sale price, just take the exponential).







The linear regression equation for this model is:

ln(SalePrice) = 11.853712323 +
 0.123934337((Overall.Qual - 6.095726)/1.3963486) + 
 0.136089772((Gr.Liv.Area - 1493.636325)/484.9349280)  + 
0.034304434((Garage.Cars - 1.768803)/0.7591321) + 
0.076780929((Total.Bsmt.SF - 1048.171795)/425.1529679) + 
0.009308631((1st.Flr.SF - 1154.117521) / 378.2119002) + 
0.028618693((Lot.Area - 10140.485470)/8345.8225594) +
 0.075797256((Year.Built - 1971.394444)/30.2048084) + 
0.021313824((BsmtFin.SF.1-443.111538)/438.1302054) +
 0.024709399((Year.Remod.Add - 1984.134615)/20.8984863) + 
0.004113766((Garage.Yr.Blt - 1978.168321)/24.3972066) + 
0.053811831((Overall.Cond - 5.567949)/1.1185016) + 
0.099941783(Central.Air) + 
-0.005866364(Bsmt.Qual) + 
-0.027382521((Bsmt.Unf.SF-553.861111)/441.5821364)

Using this equation manually for the first example in the dataset gives:

ln(SalePrice) = 11.853712323 +
 0.123934337((5  - 6.095726)/1.3963486) + 
 0.136089772((896 - 1493.636325)/484.9349280)  + 
0.034304434((1 - 1.768803)/0.7591321) + 
0.076780929((882 - 1048.171795)/425.1529679) + 
0.009308631((896 - 1154.117521) / 378.2119002) + 
0.028618693((11622 - 10140.485470)/8345.8225594) +
 0.075797256((1961 - 1971.394444)/30.2048084) + 
0.021313824((468 - 443.111538)/438.1302054) +
 0.024709399((1961 - 1984.134615)/20.8984863) + 
0.004113766((1961 - 1978.168321)/24.3972066) + 
0.053811831((6  - 5.567949)/1.1185016) + 
0.099941783(2) + 
-0.005866364(6) + 
-0.027382521((270 - 553.861111)/441.5821364)



= 11.853712323 + 0.123934337(-0.784708429) +  0.136089772(-1.232405196) 
  	+ 0.034304434(-1.012739925) + 0.076780929(-0.390851781)
	+ 0.009308631(-0.682468006) + 0.028618693(0.1775157) 
+ 0.075797256(-0.34413211) + 0.021313824(0.056806085) 
+  0.024709399(-1.106999574) + 0.004113766(-0.703700286) 
+ 0.053811831(0.386276859) + 0.099941783(2)
+ -0.005866364(6) + -0.027382521(-0.642827433)
	
= 11.670670469219339515. 

This is the natural log, so to get back the original sale price, we take the exponential of this result.

Y_pred = e^11.670670469219339515 = 117086.7590377469988

So to see our error, we subtract this from the actual sale price of 105000.

Error = y_true – y_pred = 105000 - 117086.7590377469988 = -12086.7590377469988

So our model does reasonable well for the first example, the prediction 11.5% off from the true value, meaning it is 88.5% accurate for this example. The summary of my regressor can be found below

 ![alt text](https://i.imgur.com/BGFwPBO.png "Residuals")


 
