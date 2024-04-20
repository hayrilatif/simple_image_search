import sys
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import tqdm

class Meta():
    def __init__(self):
        self.search_index=None
        self.model_path=None
meta=Meta()

class Interpreter():
    def __init__(self):
        self.tf_interpreter=None
        self.input_details=None
        self.output_details=None
interpreter=Interpreter()

class VectorContainer():
    def __init__(self):
        self.vector_list={}
    def get_json_serializable(self):
        return {i:list(self.vector_list[i].astype(str)) for i in self.vector_list}
    def set_from_json_serializable(self,d):
        self.vector_list={i:np.array(d[i]).astype(np.float32) for i in d}
vector_container=VectorContainer()

pcommand=sys.argv
if len(pcommand)>1:
    preset_path=pcommand[1]
    with open(preset_path,"r") as f:
        presets=json.loads(f.read())
        
        meta.search_index=presets["search_index"]
        meta.model_path=presets["model_path"]

if len(pcommand)>2:
    vectors_filename=pcommand[2]
    with open(vectors_filename,"r") as f:
        vector_file=json.loads(f.read())
        vector_container.set_from_json_serializable(vector_file)



print("Hello! Welcome to Simple Image Search v1")
print("Author: hayrilatif")
print(".")

def command_set(parsed_command):
    if parsed_command[1]=="search_index":
        meta.search_index=parsed_command[2]
        
    if parsed_command[1]=="model_path":
        meta.model_path=parsed_command[2]
    
    print(".")

def command_load_preset(parsed_command):
    with open(parsed_command[1],"r") as f:
        presets=json.loads(f.read())
        meta.search_index=presets["search_index"]
        meta.model_path=presets["model_path"]
    print(".")

def command_save_preset(parsed_command):
    jf=json.dumps({"search_index":meta.search_index,"model_path":meta.model_path})
    with open(parsed_command[1],"wt") as f:
        f.write(jf)
    print(".")

def command_load_vectors(parsed_command):
    with open(parsed_command[1],"r") as f:
        vector_file=json.loads(f.read())
        vector_container.set_from_json_serializable(vector_file)
    print(".")

def command_save_vectors(parsed_command):
    jf=json.dumps(vector_container.get_json_serializable())
    with open(parsed_command[1],"wt") as f:
        f.write(jf)
    print(".")

def command_prepare_model(parsed_command):
    interpreter.tf_interpreter=tf.lite.Interpreter(model_path=meta.model_path)
    interpreter.tf_interpreter.allocate_tensors()
    
    interpreter.input_details=interpreter.tf_interpreter.get_input_details()
    interpreter.output_details=interpreter.tf_interpreter.get_output_details()
    print(".")
    
def predict(x):
    p=[]
    for x_i in x:
        interpreter.tf_interpreter.set_tensor(interpreter.input_details[0]["index"],x_i[np.newaxis,...].astype(np.float32))
        interpreter.tf_interpreter.invoke()    
        p.append(interpreter.tf_interpreter.get_tensor(interpreter.output_details[0]["index"]))
    return np.array(p)

def command_create_vectors(parsed_command):
    image_paths=glob.glob(meta.search_index+"/*")
    
    arrays=[]
    for path in image_paths:
        array=cv2.resize(cv2.imread(path),(32,32))
        arrays.append(array)
    image_batch=((np.array(arrays)/255.)-0.5)*2.
    
    vectors=predict(image_batch).squeeze()
    
    for path,vector in zip(image_paths,vectors):
        vector_container.vector_list[path]=vector
    print(".")
        
def command_find_nearest(parsed_command):
    print("Searching...")
    
    vector=predict((((cv2.resize(cv2.imread(parsed_command[1]),(32,32))[np.newaxis,...]/255.)-0.5)*2.)).squeeze()
    
    scores=[]
    for v in tqdm.tqdm([vector_container.vector_list[i] for i in vector_container.vector_list]):
        scores.append(((v-vector)**2).sum())
    
    index=scores.index(min(scores))
    
    print("Distance: ",min(scores))
    print("Path: ",list(vector_container.vector_list.keys())[index])
    print(".")

def command_exit(parsed_command):exit()

#Command Index
command_dict={
    "set":command_set,
    "load_preset":command_load_preset,
    "save_preset":command_save_preset,
    "load_vectors":command_load_vectors,
    "save_vectors":command_save_vectors,
    "prepare_model": command_prepare_model,
    "create_vectors": command_create_vectors,
    "find_nearest": command_find_nearest,
    "exit":command_exit
}

#Engine
while True:
    command=input("command: ")
    parsed_command=command.split(" ")
    
    command_dict[parsed_command[0]](parsed_command)