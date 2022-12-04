import subprocess
import tempfile
import os
from time import time
import pandas as pd

DIR = ""
OLD_TRAIN_DATA =  DIR+"train_vul_funcs.csv"
OLD_TEST_DATA =   DIR+"test_vul_funcs.csv"
OLD_VAL_DATA =    DIR+"val_vul_funcs.csv"

TRAIN_DATA =  DIR+"train_vul_funcs_cpg.csv"
TEST_DATA =   DIR+"test_vul_funcs_cpg.csv"
VAL_DATA =    DIR+"val_vul_funcs_cpg.csv"

OLD_DATA_PATH = OLD_TRAIN_DATA
NEW_DATA_PATH = TRAIN_DATA
TMP = "tmp3"
cpg_col = 'cpg'
CPG_SEP = '\n--====--\n'

def exist_cpg(i, data, processed_data):
  func = data["processed_func"][i]
  flaw_lines = data["flaw_lines"][i]
 
  res = processed_data.index[processed_data['processed_func'] == func].tolist() 
  if len(res) == 0:
    return -1
  return res[0] 

if __name__ == '__main__':
  data = pd.read_csv(OLD_DATA_PATH, encoding='utf-8')
  print(len(data))
  processed_data = pd.read_csv("temp_vul_func_train.csv", encoding='utf-8')
  fs = []
  if cpg_col not in data.columns:
    for i in range(0, len(data)):
      fs.append("")
    data[cpg_col] = fs

  joern_cmd = "/content/drive/MyDrive/linevd/storage/external/joern-cli/joern"
  script = "/content/drive/MyDrive/linevd/storage/external/get_func_graph.scala"

  counter = 0
  start = time()
  count = 0
  for i in range(0, len(data)):
      print("processing...", i)
      idx_processed = exist_cpg(i, data, processed_data)
      if idx_processed != -1:
        data[cpg_col][i] = processed_data.at[idx_processed, "cpg"]
        continue
      file_name = data['file_name'][i]
      if not isinstance(file_name, str):
        file_name = "train.c"
      code = str(data['processed_func'][i])
      cpg = str(data[cpg_col][i])
      if len(code) > 100 and len(cpg) < 5:
        # temp_dir = tempfile.TemporaryDirectory()
        f = open(TMP+"/"+os.path.basename(file_name), "a")
        f.write(str(code))
        f.close()
        
        cmd = joern_cmd +" --script "+script+" --params='filename="+f.name+"'"
        # print(cmd)
        # subprocess.getoutput(cmd)
        os.system(cmd)

        if os.path.isfile(f.name+".nodes.json"):
          nodes_file = open(f.name+".nodes.json", "r")
          edges_file = open(f.name+".edges.json", "r")
          cpg = nodes_file.read()+CPG_SEP+edges_file.read()
          data[cpg_col][i] = cpg

          if data[cpg_col][i]:
            counter = counter+1
            print("----> ROW " + str(i) + " is ok")
            avg = (time()-start)/(counter)
            print("FILE: "+ file_name)
            print("----> AVG: " +  str(avg) + "s")
            nodes_file.close()
            edges_file.close()

            os.remove(f.name+".nodes.json")
            os.remove(f.name+".edges.json")
            os.remove(f.name)
            
          else:
            print("ERROR: " + file_name)
  data.to_csv(NEW_DATA_PATH, encoding='utf-8')