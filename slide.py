import os
# from PathOmics.PathOmics.split_tiles_utils.Customized_resnet import resnet50_baseline

# model = resnet50_baseline(pretrained=True)

path = "./DATA/COAD/"

outputPath = './TILLED/'

cnt = 0

for f in os.listdir(path):
    cnt += 1
    if f[-4:] == ".txt" or cnt < 480:
        continue
    inputPath = path + '/' + f
    cmd = "python Tiling.py -i {} -o {} -ws 512".format(inputPath, outputPath)
    os.system(cmd)
    
    

# images = [
#     f for f in os.listdir(path)
#         if os.isfile(os.join(path, f)) and 
#             f[-4:] == ".jpg"
# ]

