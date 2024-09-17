#TASK 1 Lists, Dictionaries, Tuples


nums = [3, 5, 7, 8, 12]
print(f"\nQuestion-1.1")
cube = []
for i in nums:
    cube.append(i*i*i)
print( cube)

#Question-1.2

dict = {}
print(f"\nQuestion-1.2")
print("original dictionary: ", dict)

dict['parrot'] = 2 
dict['goat'] = 4 
dict['spider'] = 8  
dict['crab'] = 10 

print("updated dictionary: ", dict)

#Question-1.3
dict = {}

dict['parrot'] = 2 
dict['goat'] = 4 
dict['spider'] = 8  
dict['crab'] = 10 
 
print(f"\nQuestion-1.3")
res = 0
for key, value in dict.items():
   
    print(f"Animal:{key}    Legs:{value}")
    res += value
print("sum of legs",res)

#Question-1.4
print(f"\nQuestion-1.4")
A = (3, 9, 4,[5,6])
print(A)

A[3][0] = 8
print(A)

# Question 1.5
print(f"\nQuestion-1.5")
del A

#Question-1.6
print(f"\nQuestion-1.6")
B = ('a', 'p', 'p', 'l', 'e')
p_count = B.count('p')
print(p_count)


#Question-1.7
print(f"\nQuestion-1.7")

B = ('a', 'p', 'p', 'l', 'e')
i_index = B.index('l')
print(i_index)

#TASK-2(Numpy)

#Question-2.1
print("\nQuestion-2.1")
import numpy as np # type: ignore
A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print(A)
z = np.array([1, 0, 1])

#Question-2.2
print("\nQuestion-2.2")
b = A[:2, :2]
print(b)

#Question-2.3
print("\nQuestion-2.3")
C = np.zeros_like(A)
print(C)

#Question-2.4
print("\nQuestion-2.4")
for i in range(A.shape[1]):  # Loop over the columns
    C[:, i] = A[:, i] + z  # Add vector z to each column and store it in C
print(C)

#Question-2.5
print("\nQuestion-2.5")
X = np.array([[1,2],[3,4]])
Y = np.array([[5,6],[7,8]])
v = np.array([9,10])
sum_result = X+Y
print("Sum of X and Y is:", sum_result)

#Question-2.6
print("\nQuestion-2.6")

mul_result = X*Y
print("Product of X and Y is:", mul_result)

#Question-2.7
print("\nQuestion-2.7")
sqrt_result = np.sqrt(Y)
print("Elemant wise square root of X and Y is:", sqrt_result)

#Question-2.8
print("\nQuestion-2.8")
dot_p_result = np.dot(X, v)
print("Product of X and v is:", dot_p_result)

#Question-2.9
print("\nQuestion-2.9")
column_s_result = np.sum(X, axis=0) #For row axis=1
print("Sum of each column of X:\n", column_s_result)



#TASK-3(Functions and Loops)

#Question-3.1
print("\nQuestion-3.1")
def Compute(Dis, T):
    Vel = Dis / T
    return Vel
distance = 80
time = 4
Vel = Compute(distance, time)
print(f"The velocity is {Vel} m/s.")

#Question-3.2
print("\nQuestion-3.2")

even_num = [2, 4, 6, 8, 10, 12]
def mult(even_num):
    product = 1
    for num in even_num:
        product *= num
    return product
product_of_even_num = mult(even_num)
print(product_of_even_num)

#TASK-4(Pandas)
#Question-4.1
print("\nQuestion-4.1")
import pandas as pd
import numpy as py
d = {'C1': [1, 2, 3, 5, 5], 'C2': [6, 7, 5, 4, 8], 'C3': [7, 9, 8, 6, 5],'C4': [7, 5, 2, 8, 8],}
df = pd.DataFrame(data=d)
print(df)

print(df.iloc[:2])
#Question-4.2
print("\nQuestion-4.2")
#print (df.head(2))
print("\nsecond_column:", df['C2'])

#Question-4.3
print("\nQuestion-4.3")
df.rename(columns={'C3': 'B3'}, inplace=True)
print(df)

#Question-4.4
print("\nQuestion-4.4")
lst = df.sum()
df['Sum'] = lst
print(df)


#Question-4.5
print("\nQuestion-4.5")
df['Sum'] = df.sum(axis=1)
print(df)

#Question-4.6
print("\nQuestion-4.6")
df = pd.read_csv(r'hello_sample.csv')

# Question-4.7
print("\nQuestion-4.7")
print(df.to_string())

# Question-4.8
print("\nQuestion-4.8")
print(df.tail(2))

# Question-4.9
print("\nQuestion-4.9")
print(df.info())

# Question-4.10
print("\nQuestion-4.10")
print(df.shape)

# Question-4.11
print("\nQuestion-4.11")
df_sorted = df.sort_values(by='Weight')
print(df_sorted)

# Question-4.12
print("\nQuestion-4.12")
# Check for missing values using isnull()
print("\nMissing values:")
print(df.isnull())

# Drop rows with missing values using dropna()
df_dropped = df.dropna()

print("\nDataFrame after dropping rows with missing values:")
print(df_dropped)

