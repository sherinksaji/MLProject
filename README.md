# 50.007 Machine Learning Project 2022

## Dependencies needed to install

It is strongly recommended to establish a pip environment for executing all the files. The code was generated using python 3.11.4 and any python version 3.7+ should be consistent with executing our code.

We have listed down the imports for the packages we are using in our project code itself. Here's a list of the pacakges for your reference:

 1. Pandas
 2. NumPy
 3. Regular expression

These packages can be installed via the general command:

**MacOS/Linux**

    pip3 install pandas
    pip3 install numpy
    pip3 install regex
    
**Windows**

    pip install pandas
    pip install numpy
    pip install regex

## Instructions to run our code

Please follow the below instructions for setting our code:

 Utilize <b> git bash (Windows) or powershell or Mac/Linux:</b>
 
Step 1: ```git clone https://github.com/sherinksaji/MLProject.git```
Step 2: ```cd MLProject```
Step 3: ``` git checkout main```

Please follow the below instructions for running our code as well as evalResult:

Note: If you are running Windows, use  `python`  instead of  `python3`  for each command specified below (unless you have aliased it on your system)

### Part 1:

**For ES dataset**:
```
python part1.py
python  evalResult.py  Data/ES/dev.out  Data/ES/dev.p1.out
```

**For RU dataset**:
```
python part1.py
python  evalResult.py  Data/RU/dev.out  Data/RU/dev.p1.out
```

### Part 2:

**For ES dataset**:
```
python part2.py
python  evalResult.py  Data/ES/dev.out  Data/ES/dev.p2.out
```

**For RU dataset**:
```
python part2.py
python  evalResult.py  Data/RU/dev.out  Data/RU/dev.p2.out
```

### Part 3:

**For Evaluating ES 2nd-best k**:
```
python evalResult.py Data/ES/dev.out Data/ES/dev.p3.2nd.out
```

**For Evaluating ES 8th-best k:**:
```
python evalResult.py Data/ES/dev.out Data/ES/dev.p3.8th.out
```

**For Evaluating RU 2nd-best k:**:
```
python evalResult.py Data/RU/dev.out Data/RU/dev.p3.2nd.out
```

**For Evaluating RU 8th-best k:**:
```
python evalResult.py Data/RU/dev.out Data/RU/dev.p3.8th.out
```

### Part 4:

In part4, we have two main files you can test on:
 1. dev.in
 2. test.in

You can choose whichever file to test on by modifying the flag variable (acts as a toggle) in the following code in part4.py:
```
# set to dev for analyzing dev.in dataset
# set to test for analyzing test.in dataset
flag = 'test'
```

Similarly, you can choose whichever dataset you would like to train and test for by modifying the lang variable (acts as a toggle) in the followign code in part4.py:

```
#set to ES/RU for whichever dataset you wish to analyze
lang = 'ES'
```

Before running the EvalScript, you must run part4.py without debugging. 

**For ES dataset**:
```
python part4.py
python  evalResult.py  Data/ES/dev.out  Data/ES/dev.p4.out
```
**For RU dataset**:
```
python part4.py
python  evalResult.py  Data/RU/dev.out  Data/RU/dev.p4.out
```

Running test data for Part 4:

**For ES dataset**:
```
python part4.py
python  evalResult.py  Data/ES/dev.out  Test/ES/test.p4.out
```
**For RU dataset**:
```
python part4.py
python  evalResult.py  Data/RU/dev.out  Test/RU/test.p4.out
```


## Team Members

-   Mehta Yash Piyush: 1006516
-   Sherin Karuvallil Saji: 1005228
-   Tang Heng: 1006102
-  Venkatakrishnan Logganesh: 1006050