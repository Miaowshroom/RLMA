Newly trained A2C models. Previous models under this directory were moved to ./backup.


# cmd to run attack
```
python Minimalistic_Attack.py -g 'Assault.pkl' -a 'a2c' -t 0.5689810415866755 --use_gpu --customized_xf -n 2 --deterministic --r A2C_Assault_n2_deter

python Minimalistic_Attack.py -g 'Assault.pkl' -a 'a2c' -t 0.5689810415866755 --use_gpu --customized_xf -n 2 --r A2C_Assault_n2_stoch
```