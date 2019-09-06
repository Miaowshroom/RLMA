## RLMA
This code is for the paper: Minimalistic Attacks: How Little it Takes to Fool a Deep Reinforcement Learning Policy

# Dependencies:
	python 3.6
	tensorflow
	gym[atari]
	stable_baselines
	cv2
	pybrain
	multiprocessing
	matplotlib
# Run the code with command:
	python DQN_Minimalistic_Attack.py    -g   'Pong'  -a   'dqn'   -n   5   -t   0.9   -r   'dqnrun'
	python A2C_Minimalistic_Attack.py    -g   'Pong'  -a   'a2c'   -n   5   -t   0.9   -r   'a2crun'
	python PPO2_Minimalistic_Attack.py   -g   'Pong'  -a   'ppo2'  -n   5   -t   0.9   -r   'ppo2run'
	python ACKTR_Minimalistic_Attack.py  -g   'Pong'  -a   'acktr' -n   5   -t   0.9   -r   'acktrrun'
# Run from shell
	cd ~/RLMA
	chmod +x run_dqn_attack.sh
	bash run_dqn_attack.sh
	
# Results
	After running, the results will be saved in the folder results/runname/.
	The video for the atari game playing is also saved in the folder results/videos_runname/.
	
