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
python DQN_Minimalistic_Attack.py -g 'Pong' -a 'dqn' -n 5 -t 0.9 -r 'testrun'
