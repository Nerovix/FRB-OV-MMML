import os



patients=[]
for npyfiles in os.listdir(os.getcwd()+"/Extracted_Features_use"):
    if npyfiles[-4:]=='.npy':
        patients.append(npyfiles[:-4])
# print(patients)

for patient in patients:
    os.system("rm ./use/* -rf");
    os.makedirs("./use/"+patient);
    os.system("cp data_cna.txt ./use/"+patient+"/data_cna.txt")
    print(patient,end=" ")
    os.system("python ./use.py > use_out.txt")
    with open("use_out.txt","r") as f:
        content=f.read()
        print(content)
    
