cd vamos_ws/src

# Install trajectory packages
pip install -e trajectory_prediction/
pip install -e trajectory_projection/

cd vamos_src/src/vamos

python dowload_hf_model.py --model_id "mateoguaman/vamos_navigation_only"
