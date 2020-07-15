# Project 2: Two Deep Learning Models for Wisconsin Breast Cancer Dataset

## Description
See project-2.pdf in the repository for details.

## Instructions
1. To begin, you need three files (and, of course, ```proj2-8.py``` and ```proj2-16.py```):
    ```
    x_test.csv
    y_test.csv
    fraction_xy.py
    ```

2. Run fraction_xy.py to get a random subset of testing data as training data.
    *Windows*
    ```bash
    python fraction_xy.py x_test.csv y_test.csv <0.08 or 0.16> <seed-value>
    ```

    *Linux*
    ```bash
    python3 fraction_xy.py x_test.csv y_test.csv <0.08 or 0.16> <seed-value>
    ```
    and then rename the generated files to ```x_train8.csv``` and ```y_train8.csv```  or ```x_train16.csv``` and ```y_train16.csv```, depending on which one you generated for.

3. Run the program by typing in the command line/terminal:

    *Windows*
    ```bash
    python proj2-8.py
    ```
    or
    ```bash
    python proj2-16.py
    ```

    *Linux*
    ```bash
    python3 proj2-8.py
    ```
    or
    ```bash
    python3 proj2-16.py
    ```