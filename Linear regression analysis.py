import scipy.stats
import numpy as np 
from matplotlib import pyplot as plt

# UniCode for Sigma & Beta symbol.
sigma_S = "\u03A3"
beta_S = "\u03B2"
beta0_S = "\u03B2\u2080"
beta1_S = "\u03B2\u2081"
beta2_S = "\u03B2\u2082"
power2_S = "\u00B2"
power_minus1_S = "\u207B\u00B9"

#----------------------------- הכנסת הנתונים ------------------------------
data1 = np.array([ [41200,2400],
                   [50300,2700],
                   [55000,2850],
                   [66000,4950],
                   [44500,3100],
                   [37700,2500],
                   [73500,5106],
                   [37500,3100],
                   [56700,3400],
                   [35600,1750] ])


[rows,cols] = np.shape(data1)

 # הפרדת העמודות
X = data1[:,:-1]
Y = data1[:,-1]

#------------ הצגת גרף של הוצאות כפונקציה של הכנסות -------------

print("\nSection A - סעיף א")

# ציור נקודות בצבע ציאן (Cyan) בהתאם לערכי X ו-Y שהוזנו. X[:,0] מייצג את כל הערכים בעמודה הראשונה של X.
plt.plot(X[:,0], Y, "c*")

# הגדרת טווח הציר האופקי (X) מ-0 עד 80,000
plt.xlim(0, 80000)

# הגדרת טווח הציר האנכי (Y) מ-0 עד 6,000
plt.ylim(0, 6000)

# יצירת מערך של ערכי X לצורך ציור הקו המותאם מ-20,000 עד 79,000 בצעדים של 0.05
x_graph = np.arange(20000, 79000, 0.05)

# חישוב ערכי Y באמצעות הנוסחה המותאמת לקו, עבור כל ערך של X במערך x_graph
y_graph = -494.97550229 + (0.07226305 * x_graph)

# ציור הקו המותאם על הגרף בצבע שחור
plt.plot(x_graph, y_graph, "k-")

# הוספת תווית לציר X
plt.xlabel("X - Income")

# הוספת תווית לציר Y
plt.ylabel("Y - Expenditure")

# הוספת כותרת לגרף
plt.title('Expenditure as a function of Income')

# הצגת רשת על הגרף לקלות קריאה
plt.grid()

# הצגת הגרף
plt.show()

#--------------- סעיף ב- מציאת משוואת קו הרגרסיה הלינארית ---------------
print("\n ---------- Section B - סעיף ב ----------")

# יצירת מטריצת עמודה מ-Y
Y = np.reshape(Y,[rows,1])
# יצירת מטריצת עמודה של אחדים באורך שווה למספר השורות ב-X
ones = np.ones([rows,1])
# הוספת עמודת האחדים ל-X כעמודה ראשונה
X = np.hstack([ones,X])

# הדפסת מטריצת X והוקטור Y (מוגבל בתגובה זו)
#print("Matrix X:\n",X)
#print("\nVector Y:\n",Y)

# חישוב המכפלה הטרנספוזיציונית של X ב-X
XTX = np.matmul(X.T,X)
print("\nXTX = X.T * X:\n",XTX)

# חישוב המכפלה הטרנספוזיציונית של X ב-Y
XTY = np.matmul(X.T,Y)
print("\nXTY = X.T * Y:\n",XTY)

# חישוב ההופכי של המטריצה XTX
inv_XTX = np.linalg.inv(XTX)
print(f"\n(XTX){power_minus1_S}:\n",inv_XTX)

# חישוב וקטור הבטא (המקדמים) על ידי כפל ההופכי של XTX ב-XTY
betas = np.matmul(inv_XTX,XTY)
print(f"\nbetas: {beta_S} = (XTX){power_minus1_S} * XTY:")
print(f"{beta0_S} =",betas[0],f"\n{beta1_S} =",betas[1])

# לחשב את השגיאות בין הערכים הנצפים לערכים שהמודל חזה

#e = Y - y_hat  # מחשב את ההפרש בין הערכים האמיתיים (Y) לבין הערכים שנחזו על ידי המודל (y_hat). שורה זו מוגדרת כהערה ולכן לא תבוצע

#print("\ne = Y - y_hat:\n",e)  # מדפיס את ההפרשים (השגיאות) שחושבו. גם שורה זו מוגדרת כהערה ולכן לא תבוצע

#---------------------- סעיף ג- מציאת SST SSE SSR R^2 ---------------------
print("\n ---------- Section C - סעיף ג ----------")

# הגדרות למדדים שונים
print(f"SSR = {sigma_S}((y_hat - y_mean)**2)")
print(f"SST = {sigma_S}((Y - y_mean)**2)")
print(f"SSE = {sigma_S}((Y - y_hat)**2)")

# הצגת אופציה נוספת לחישוב SSE
print("Option 2: SEE = np.matmul(e.T,e)")
print(f"R{power2_S} = 1 - SSE/SST")

# הצגת אופציה נוספת לחישוב R מרובע
print(f"Option 2: R{power2_S} = SSR/SST\n")

# חישוב הערכים המתבקשים של y_hat ו-y_mean

# חישוב הערכים המתוחזים על ידי המודל
y_hat = np.matmul(X,betas)

 # חישוב הממוצע של ערכי Y
y_mean = np.mean(Y)

# חישוב והדפסת SSR, הסכום הריבועי של השאריות
SSR = np.sum((y_hat-y_mean)**2)
print("SSR =",SSR)

# חישוב והדפסת SST, הסכום הריבועי הכולל
SST = np.sum((Y-y_mean)**2)
print("SST =",SST)

# חישוב והדפסת SSE, הסכום הריבועי של השגיאות
SSE = np.sum((Y-y_hat)**2)
print("SSE =",SSE)

# חישוב והדפסת R בריבוע, מדד לתיאור ההתאמה של המודל לנתונים
R_squared = 1 - SSE/SST
print(f"\nR{power2_S} =", R_squared)

#-------------------- סעיף ד- הסבר על הקשר בין שני המשתנים הלינארים --------------------
print("\n ---------- Section D - סעיף ד ----------")
# חישוב מקדם הקורלציה (r) מתוך R מרובע.
# מקדם הקורלציה הוא השורש הריבועי של R מרובע.
r = R_squared**0.5
print("r =", r)

#------- ה- ניבוי ההוצאות של עובד שמכניס 60,000 דולרים בשנה -------
print("\n ---------- Section E - סעיף ה ----------")

# הדפסת הנוסחה של קו הרגרסיה. נוסחה זו משמשת לניבוי ערכי Y (הוצאות) מתוך ערכי X (הכנסה)
print(f"Regression line: Y = {beta0_S} + {beta1_S}X")

# הצגת הערכים של הפרמטרים בקו הרגרסיה עבור עובד עם הכנסה של 60,000 דולר
print(f"X = 60,000 \n{beta0_S} =",betas[0],f" \n{beta1_S} =",betas[1])

# חישוב ההוצאות הצפויות לעובד עם הכנסה של 60,000 דולר
h = betas[0] + betas[1]*60000

# הדפסת ההוצאות הצפויות עבור העובד
print("\nThe Expenditure will be:", h)

#---------------------------------2 ---------------------------------

# יצירת מערך נתונים עם נתוני משקל, מרחק נסיעה ורמת זיהום
data2 = np.array([ [1000,790,99],
                   [1250,1165,95],
                   [1000,929,96],
                   [900,865,90],
                   [1500,1135,105],
                   [2200,1280,104],
                   [2100,1605,115],
                   [1650,1525,108]  ])

# שמירת מספר השורות והעמודות במערך
[rows,cols] = np.shape(data2)

# הפרדת המערך למשתנים בלתי תלויים (X) ותלוי (Y)
X = data2[:,:-1]  # כל העמודות פרט לאחרונה
Y = data2[:,-1]  # רק העמודה האחרונה

# הדפסת כותרת לסעיף א', חלק א'
print("\n ---------- Section A, Part A - סעיף א, חלק א ----------")

# ציור נקודות על גרף להצגת רמת הזיהום כפונקציה של משקל המכונית
plt.plot(X[:,1],Y,"g*")  # ציור נקודות בצבע ירוק
plt.xlim(0,2500)  # הגדרת טווח ציר ה-X
plt.ylim(0,150)  # הגדרת טווח ציר ה-Y

# חישוב קו הרגרסיה לפי המקדמים שנמצאו מראש
Q2P1_x_graph = np.arange(200, 2400, 0.05)  # יצירת סדרה של ערכי X לצורך ציור הקו
Q2P1_y_graph = 74.95090209 + (0.00588758 * Q2P1_x_graph) + (0.01557099 * Q2P1_x_graph)  # חישוב ערכי Y לפי נוסחת הרגרסיה
plt.plot(Q2P1_x_graph, Q2P1_y_graph,"k-")  # ציור קו הרגרסיה

# הוספת תוויות וכותרת לגרף
plt.xlabel("X1- Weight")  # תווית ציר X
plt.ylabel("Y- Pollution level- Co2")  # תווית ציר Y
plt.title('Pollution level (Co2) as a function of car weight')  # כותרת הגרף
plt.grid()  # הצגת רשת על הגרף
plt.show()  # הצגת הגרף

#הצגת גרף של רמת הזיהום (Y) כפונקציה של נפח (X2)המכונית
print("\n ---------- Section A, Part B - סעיף א, חלק ב ----------")

# ציור נקודות על גרף להצגת רמת הזיהום כפונקציה של נפח המכונית

plt.plot(X[:,0],Y,"r*")  # ציור נקודות בצבע אדום
plt.xlim(0,2500)  # הגדרת טווח ציר ה-X
plt.ylim(0,150)  # הגדרת טווח ציר ה-Y

# כאן יש טעות בקוד. הנוסחה לחישוב Q2P2_y_graph אמורה להתייחס לנפח ולא למשקל.
# הנוסחה כתובה שוב כמו למשקל, אבל היא צריכה להשתמש במשתנה הנפח (X2) במקום.
# לכן, ההסבר יתייחס לקוד כאילו היה מופיע כאן קוד נכון לחישוב בהתאם לנפח.

Q2P2_x_graph = np.arange(200, 2400, 0.05)  # יצירת סדרה של ערכי X לצורך ציור הקו
Q2P2_y_graph = 74.95090209 + (0.00588758 * Q2P2_x_graph) + (0.01557099 * Q2P2_x_graph)  # חישוב ערכי Y לפי נוסחת הרגרסיה שאמורה להיות מתוקנת לנפח
plt.plot(Q2P2_x_graph, Q2P2_y_graph,"k-")  # ציור קו הרגרסיה

# הוספת תוויות וכותרת לגרף

plt.xlabel("X2- Volume")  # תווית ציר X, המציינת את נפח המכונית
plt.ylabel("Y- Pollution level- Co2")  # תווית ציר Y, המציינת את רמת הזיהום
plt.title('Pollution level (Co2) as a function of car volume')  # כותרת הגרף, המציינת את נושא הגרף
plt.grid()  # הצגת רשת על הגרף לנוחות קריאה
plt.show()  # הצגת הגרף

#---------------- מציאת משוואת קו הרגרסיה הלינארית (מודל) --------------
print("\n ---------- Section B - סעיף ב ----------")

# הפיכת וקטור Y למטריצה עם עמודה אחת
Y = np.reshape(Y,[rows,1])

# יצירת מטריצת עמודה של אחדים לצורך הוספת המונח החופשי למודל
ones = np.ones([rows,1])

# הוספת העמודה של אחדים למטריצת התכונות X על מנת להתחשב במונח החופשי במודל
X = np.hstack([ones,X])

# חישוב המכפלה הטרנספוזיציונית של X ב-X
XTX = np.matmul(X.T,X)
print("\nXTX = X.T * X:\n",XTX)

# חישוב המכפלה הטרנספוזיציונית של X ב-Y
XTY = np.matmul(X.T,Y)
print("\nXTY = X.T * Y:\n",XTY)

# חישוב ההופכי של המטריצה XTX
inv_XTX = np.linalg.inv(XTX)
print(f"\n(XTX)^-1:\n",inv_XTX)

# חישוב המקדמים (בטאס) של המודל על ידי הכפלת ההופכי של XTX ב-XTY
betas = np.matmul(inv_XTX,XTY)
print(f"\nbetas: β = (XTX)^-1 * XTY")
# הדפסת המקדמים שנמצאו
print(f"β₀ =",betas[0],f"\nβ₁ =",betas[1],f"\nβ₂ =",betas[2])

#-------------------------- סעיף ג- חישוב Adj R^2 -------------------------
print("\n ---------- Section C - סעיף ג ----------")

# הצגת הנוסחאות לחישוב SSR, SST, SSE ו-R^2 ללא חישוב ממשי
print(f"SSR = {sigma_S}((y_hat - y_mean)**2)")
print(f"SST = {sigma_S}((Y - y_mean)**2)")
print(f"SSE = {sigma_S}((Y - y_hat)**2)")
print(f"R{power2_S} = 1 - SSE/SST")
print(f"Adj R{power2_S} = 1 - ((n-1) / (n-p-1)) * (1-R{power2_S})\n")

# חישוב ערכי חזוי (y_hat) על ידי הכפלת מטריצת התכונות במקדמים
y_hat = np.matmul(X,betas)
# חישוב הממוצע של הערכים התצפיים
y_mean = np.mean(Y)

# חישוב Sum of Squares Regression (SSR)
SSR = np.sum((y_hat-y_mean)**2)
print("SSR =",SSR)

# חישוב Total Sum of Squares (SST)
SST = np.sum((Y-y_mean)**2)
print("SST =",SST)

# חישוב Sum of Squares Error (SSE)
SSE = np.sum((Y-y_hat)**2)
print("SSE =",SSE)

# חישוב של R squared (R^2)
R_squared = 1 - SSE/SST
print(f"R{power2_S} =", R_squared)

# מספר התצפיות (n) ומספר המשתנים המסבירים (p)
n = rows # מספר השורות
p = cols-1 # מספר העמודות פחות 1

# חישוב של R squared מתואם (Adjusted R^2)
Adj_R_squared = 1- ((n-1) / (n-p-1)) * (1-R_squared)
print(f"\nAdj R{power2_S} =",Adj_R_squared)

#-------------------- סעיף ד- חיזוי רמת זיהום של מכונית מסוימת --------------------
print("\n ---------- Section D - סעיף ד ----------")

# חישוב ה-MSE (Mean Squared Error) של המודל
MSE = SSE/n

# חישוב ה-RMSE (Root Mean Squared Error), שהוא השורש הריבועי של ה-MSE
RMSE = MSE**0.5
print("RMSE =", RMSE)

#-------------------- סעיף ה- בדיקת מובהקות לפיצ'רים --------------------
print("\n ---------- Section E - סעיף ה ----------")

# הצגת נוסחת השונות
print("Variance = SSE / (n-p-1)\n")

# חישוב שונות השגיאה
variance = SSE / (n-p-1)
print("Variance =",variance)

# חישוב מטריצת השונות-שוני
var_covr_Matrix = variance * inv_XTX
print(f"\nVariance Covariance Matrix = Variance * (XTX)^{-1}\n", var_covr_Matrix)

# חישוב ערך t לבדיקת המובהקות של המקדם הראשון
t1 = betas[1] / np.sqrt(var_covr_Matrix[1,1])
print(f"\nbeta1: t1 =",t1)

# חישוב P-value לערך t הראשון
P_value_t1 = 2*scipy.stats.t.sf(abs(t1),df = n-p-1)
print("t1: P_value  =", P_value_t1)

# חישוב ערך t לבדיקת המובהקות של המקדם השני
t2 = betas[2] / np.sqrt(var_covr_Matrix[2,2])
print(f"\nbeta2: t2 =", t2)

# חישוב P-value לערך t השני
P_value_t2 = 2 * scipy.stats.t.sf(abs(t2), df = n-p-1)
print("t2: P_value =", P_value_t2)

# הגדרת רמת המובהקות אלפא
alpha = 0.05
print(f"\nHypothesis testing for beta1 against alpha")

# בדיקה אם לדחות את השערת האפס עבור המקדם הראשון
if(P_value_t1<alpha):
    print(" t1 --> Reject H0")
else:
    print(" t1 --> Can't Reject H0")

print(f"\nHypothesis testing for beta2 against alpha")
# בדיקה אם לדחות את השערת האפס עבור המקדם השני
if(P_value_t2<alpha):
    print(" t2 --> Reject H0")
else:
    print(" t2 --> Can't Reject H0")

    #-------------------- סעיף ו- בדיקת מובהקות למודל ---------------------
print("\n ---------- Section F - סעיף ו ----------")

# חישוב ערך ה-F לבדיקת מובהקות המודל כולו
F = (R_squared/p)/((1-R_squared)/(n-p-1))
print("F =",F)

# חישוב P-value לערך ה-F
P_value_F = 1 - scipy.stats.f.cdf(abs(F),p,n-p-1)
print("F: P_value =", P_value_F)

print("\nHypothesis testing for the entire model against alpha")

# בדיקה אם לדחות את השערת האפס לכל המודל
if(P_value_F<alpha):
    print(" F test --> Reject H0")
else:
    print(" F test --> Can't Reject H0") 
    
    #---------------
    
    
    
    #-------------------- סעיף ו- בדיקת מובהקות למודל ---------------------
print("\n ---------- Section F - סעיף ו ----------")

# חישוב P-value לערך ה-F
P_value_F = scipy.stats.f.sf(7.621,2,8-2-1)
print("F: P_value new=", P_value_F)

print("\nHypothesis testing for the entire model against alpha")

# בדיקה אם לדחות את השערת האפס לכל המודל
if(P_value_F<alpha):
    print(" F test --> Reject H0")
else:
    print(" F test --> Can't Reject H0") 