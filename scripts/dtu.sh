dataset_folder=/media/baowen/data/dataset/dtu/DTU_mask
for scene in 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122
do
    python train.py -s ${dataset_folder}/scan${scene} -m output/dtu/scan${scene} -r 2 --use_decoupled_appearance 3
    python mesh_extract.py -m output/dtu/scan${scene}
    python evaluate_dtu_mesh.py -m output/dtu/scan${scene}
done