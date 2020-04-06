import numpy as np
import os
from kadai3 import Input_data

#結果をファイルに書き込む

class Write_output:
    def __init__(self):
        pass
    def write(self,result):
        tmp= os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        path= "../result.txt"
        S= "test\n"+str(result)
        with open(path, mode='w') as f:
            f.write(S)

if __name__=="__main__":
    #testデータで結果を出力し、ファイルに書き込む
    a= Write_output()
    a.write(6)
