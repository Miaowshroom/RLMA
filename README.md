## RLMA
This code is for the paper: Minimalistic Attacks: How Little it Takes to Fool a Deep Reinforcement Learning Policy

# Usage

	# install environment from .yml
	conda env create -f env.yml

	# source env
	source activate RLMA

	# install some special packages that could not be installed from pip
	pip install -r req.txt

	# remember to deactive the environment when you don't need it any more or switch to other tasks :)
	conda deactivate

# Notice
	Some code modifications have been done and the new code is in Minimalistic_Attack.py.

# Dependencies:
	python 3.6
	tensorflow
	gym[atari]
	stable_baselines
	cv2
	pybrain
	multiprocessing
	matplotlib

	[more details in req.txt]
# Run the code with command(to load the author's models): 
	python Minimalistic_Attack.py    -g   'give a name you want/this will create the folder name'  -a   'not important'   -n   pixels_to_attack   -t   0.9   -r   'SAC-discrete (root name for the folder)' --customized_path "./SAC_model/saved_models/sac_discrete_atari_BeamRider-v4/sac_discrete_atari_BeamRider-v4_s3/tf1_save5"

# Run the code with command(to load the author's models): 
	python Minimalistic_Attack.py    -g   'Pong'  -a   'dqn'   -n   5   -t   0.9   -r   'dqnrun'
	python Minimalistic_Attack.py    -g   'Pong'  -a   'a2c'   -n   5   -t   0.9   -r   'a2crun'
	python Minimalistic_Attack.py   -g   'Pong'  -a   'ppo2'  -n   5   -t   0.9   -r   'ppo2run'
	python Minimalistic_Attack.py   -g   'Pong'  -a   'acktr' -n   5   -t   0.9   -r   'acktrrun'


# Run from shell
	cd ~/RLMA
	chmod +x run_dqn_attack.sh
	bash run_dqn_attack.sh
	
# Results
	After running, the results will be saved in the folder results/runname/.
	The video for the atari game playing is also saved in the folder results/videos_runname/.
	
