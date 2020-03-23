## RLMA
This code is for the paper: Minimalistic Attacks: How Little it Takes to Fool a Deep Reinforcement Learning Policy

# Usage

	# install conda and run in the repo
	conda env create -f env.yml

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
# Run the code with command:
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
	
