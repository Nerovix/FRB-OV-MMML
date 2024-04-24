from data_loading_utils.load_files import *
from data_loading_utils.Customized_Dataset import *
import pandas as pd
from model_and_training_utils.PathOmics_Survival_model import *
import torch
import os
from extract3_use import extract_use

def run_model(model,data_WSI, data_omic, device, model_mode = 'finetune'):
    model.eval()    
    data_WSI = data_WSI.squeeze().to(device) if data_WSI is not None else None
    data_omic = [i.squeeze().to(device) for i in data_omic] if data_omic is not None else None
    with torch.no_grad():
        hazards, S, Y_hat, _, _ = model(x_path=data_WSI, x_omic=data_omic, x_cluster = [], mode = model_mode)
    return Y_hat.item()


def main():
    patient_name=extract_use()
    
    z_score_path_folder=os.getcwd()+'/use/'+patient_name;
    z_score_path=[]
    for file in os.listdir(z_score_path_folder):
        if(file[-4:]=='.txt'):
            z_score_path.append(file)
    

    assert len(z_score_path)<=1,'wtf, too many .txt gene file'
    haveomics=len(z_score_path)

    if haveomics==1:
        z_score_path=z_score_path_folder+'/'+z_score_path[0]

    feature_folder=os.getcwd()+'/Extracted_Features_use'
    if os.path.exists(feature_folder+'/'+patient_name+'.npy'):
        havepath=1
    else:
        havepath=0
    
    assert havepath==1 or haveomics==1,'no path no omics, U R joking'

    if haveomics==1:
        df_cna = pd.read_csv(z_score_path, sep='\s+',engine='python')
        
        tmp=df_cna.columns
        tmpp='404notfound'
        for col in tmp:
            if col[0:12]==patient_name:
                tmpp=col
                break

        assert(tmpp!='404notfound'),'wrong name in .txt head'
        tobedropped=[]
        for col in df_cna.columns:
            if col!=tmpp and col != 'Hugo_Symbol':
                tobedropped.append(col)
        df_cna = df_cna.drop(tobedropped, axis=1)
        df_cna = df_cna.rename(columns={tmpp:patient_name+'-01'})
        df_cna = df_cna[df_cna['Hugo_Symbol'].notna()].dropna()
        df_cna = df_cna.set_index('Hugo_Symbol')
        df_gene = df_cna
    else:
        df_gene=None

    dict_family_genes = load_gene_family_info('gene_family')
    x_omics=load_genomics_z_score(df_gene,patient_name,dict_family_genes)
    x_path=load_feature(feature_folder,patient_name)
    
    if x_omics is not None:
        tmp=[]
        for f in x_omics:
            if f.ndim==1:
                tmp.append(torch.unsqueeze(f,0))
        x_omics=tmp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model=PathOmics_Surv(device=device,
                         fusion='concat',
                         omic_sizes=[i.shape[1] for i in x_omics] if haveomics==1 else [],
                         model_size_omic='small',
                         omic_bag='SNN',
                         use_GAP_in_pretrain=True,
                         proj_ratio=1,
                         image_group_method='random')
    model_path=os.getcwd()+'/reproduce_experiments/COAD_reproduce/1_model_PathOmics_OmicBag_SNN_FusionType_concat_OmicType_CNA/fold_3_finetune_model_11.pt'
    ckpnt=torch.load(model_path)
    model.load_state_dict(ckpnt['model_state_dict'],strict=False)
    model.to(device=device)

    print(run_model(model,x_path,x_omics,device))

if __name__ == "__main__":
    main()
