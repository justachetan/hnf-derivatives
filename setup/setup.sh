# setup train environment
conda env create -f train_environment.yml
conda activate hnf-train
pip install -r train_requirements.txt
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=hnf-train
conda deactivate
echo "hnf-train is ready to use! activate with: conda activate hnf-train"

# setup render environment
conda env create -f render_environment.yml
conda activate hnf-render
pip install -r render_requirements.txt
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=hnf-render
conda deactivate
echo "hnf-render is ready to use! activate with: conda activate hnf-render"
