PYTHONPATH=. python3 -m kbc.learn kbc/data/NELL --model ComplEx --max_epochs 100 --model_save_schedule 50 --valid 3 --reg 0.05 --batch_size 100 --rank 100
PYTHONPATH=. python3 -m kbc.learn kbc/data/NELL --model ComplEx --max_epochs 100 --model_save_schedule 50 --valid 3 --reg 0.05 --batch_size 100 --rank 200
PYTHONPATH=. python3 -m kbc.learn kbc/data/NELL --model ComplEx --max_epochs 100 --model_save_schedule 50 --valid 3 --reg 0.05 --batch_size 1000 --rank 500
PYTHONPATH=. python3 -m kbc.learn kbc/data/NELL --model ComplEx --max_epochs 100 --model_save_schedule 50 --valid 3 --reg 0.05 --batch_size 1000 --rank 1000

PYTHONPATH=. python3 -m kbc.learn kbc/data/FB15k-237 --model ComplEx --max_epochs 100 --model_save_schedule 50 --valid 3 --reg 0.05 --batch_size 500 --rank 100
PYTHONPATH=. python3 -m kbc.learn kbc/data/FB15k-237 --model ComplEx --max_epochs 100 --model_save_schedule 50 --valid 3 --reg 0.05 --batch_size 1000 --rank 200
PYTHONPATH=. python3 -m kbc.learn kbc/data/FB15k-237 --model ComplEx --max_epochs 100 --model_save_schedule 50 --valid 3 --reg 0.05 --batch_size 500 --rank 500
PYTHONPATH=. python3 -m kbc.learn kbc/data/FB15k-237 --model ComplEx --max_epochs 100 --model_save_schedule 50 --valid 3 --reg 0.05 --batch_size 1000 --rank 1000

PYTHONPATH=. python3 -m kbc.learn kbc/data/FB15k --model ComplEx --max_epochs 100 --model_save_schedule 50 --valid 3 --reg 0.001 --batch_size 1000 --rank 100
PYTHONPATH=. python3 -m kbc.learn kbc/data/FB15k --model ComplEx --max_epochs 100 --model_save_schedule 50 --valid 3 --reg 0.005 --batch_size 1000 --rank 200
PYTHONPATH=. python3 -m kbc.learn kbc/data/FB15k --model ComplEx --max_epochs 100 --model_save_schedule 50 --valid 3 --reg 0.005 --batch_size 500 --rank 500
PYTHONPATH=. python3 -m kbc.learn kbc/data/FB15k --model ComplEx --max_epochs 100 --model_save_schedule 50 --valid 3 --reg 0.01 --batch_size 100 --rank 1000
