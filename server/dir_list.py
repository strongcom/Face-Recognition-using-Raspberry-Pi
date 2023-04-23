import os

def make_dir_list(path) :
    files = os.listdir(path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
    
    #print(files_dir)
    return files_dir

def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + "\n"
    return result.strip()


path = './dataset/face-datasets'
#Stringname = listToString(make_dir_list(path))
#print(Stringname)

name = make_dir_list(path)
print(len(name))
