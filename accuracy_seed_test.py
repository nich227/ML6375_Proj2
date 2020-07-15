from subprocess import Popen, PIPE, STDOUT
import sys
import re

accuracy_for_seed_8 = []
accuracy_for_seed_16 = []

for i in range(0, 20):
    print('On seed:', i)
    Popen(['python', 'fraction_xy.py', 'x_test.csv', 'y_test.csv', '0.08', str(i)], stdout=PIPE, stderr=STDOUT).wait()
    Popen(['python', 'rename_files.py', '8', str(i)], stdout=PIPE, stderr=STDOUT).wait()
    
    Popen(['python', 'fraction_xy.py', 'x_test.csv', 'y_test.csv', '0.16', str(i)], stdout=PIPE, stderr=STDOUT).wait()
    Popen(['python', 'rename_files.py', '16', str(i)], stdout=PIPE, stderr=STDOUT).wait()
    
    for line in Popen(['python', 'nkc160130-8.py'], stdout=PIPE, stderr=STDOUT).stdout:
        line = line.decode("utf-8")
        if re.compile('0.\d+').match(line):
            accuracy_for_seed_8.append(float(line))
    
    for line in Popen(['python', 'nkc160130-16.py'], stdout=PIPE, stderr=STDOUT).stdout:
        line = line.decode("utf-8")
        if re.compile('0.\d+').match(line):
            accuracy_for_seed_16.append(float(line))

i = 0
for (accuracy_8, accuracy_16) in zip(accuracy_for_seed_8, accuracy_for_seed_16):
    print('Accuracy for seed', str(i) + ':')
    print('8:', accuracy_8)
    print('16:', accuracy_16)
    print('----------------------------------------')
    i+=1