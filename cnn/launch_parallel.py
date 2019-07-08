from subprocess import Popen 

cmd = ['python', 'generate_data_multithread.py']
threads = 8

for i in range(threads):
    Popen(cmd, cwd='/Users/bmmorris/git/orso/cnn/')