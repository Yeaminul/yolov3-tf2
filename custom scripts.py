import random
import os


########## manual train test splitter ##############

fname = []
for files in os.listdir(r'D:\Python World\yolo_tensorflow\yolov3-tf2\data\waste\Annotations'):
    # print(files.split('.')[0])
    fname.append(files.split('.')[0])

listoffiles = random.sample(fname, len(fname))
trainset = listoffiles[0:int(len(listoffiles) * 0.80)]
testset = listoffiles[int(len(listoffiles) * 0.80):-1]
print(len(trainset))
print(trainset)
print(len(testset))
print(testset)




##### change file contents in place ##########

dir = r'D:\Python World\yolo_tensorflow\yolov3-tf2\data\waste\Annotations\\'
os.chdir(dir)

def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)

# inplace_change(dir+'glass1.xml', '<name>4</name>', '<name>glass</name>')

for files in os.listdir(dir):
    inplace_change(files, '<name>plastice</name>', '<name>plastic</name>')

