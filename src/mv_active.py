import os

counter = 0
try:
    with open('records/counter', 'r') as file:
        counter = int(file.read())
except:
    print('Cannot open counter file: records/actice/counter')

counter += 1

src = 'records/active'
des = 'records/record_{}'.format(counter)
cmd = 'mv {} {}'.format(src, des)
if os.system(cmd) == 0:
    print('\n\nMoved {} to {}\n'.format(src, des))
else:
    print('Move failed. CMD: ', cmd)
    exit()

cmd = 'echo {} > records/counter'.format(counter)
if os.system(cmd) != 0:
    print('Counter increasing failed. CMD: ', cmd)
    exit()