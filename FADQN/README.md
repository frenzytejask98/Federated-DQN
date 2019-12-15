# FADQN
Federated Averaging Deep Q-Network

```
pip install tensorflow
pip install gym
pip install dill
```
## How to run ##

    cd baselines/deepq/experiments/
    python async_fed_avg.py --config_file config.ini --config DEFAULT --job_name "ps" --task_index 0 --seed 1
    python async_fed_avg.py --config_file config.ini --config DEFAULT --job_name "worker" --task_index 0 --seed 1
    python async_fed_avg.py --config_file config.ini --config DEFAULT --job_name "worker" --task_index 1 --seed 1
    python async_fed_avg.py --config_file config.ini --config DEFAULT --job_name "worker" --task_index 2 --seed 1
    .
    .
    
* `config_file` is the name of the config file for the actors.
    * Defaults to `config.ini`
    
* `config` is the section in the config file to override the default values with.
    * Defaults to `DEFAULT`. Use `async` for the `[async]` section and `sync` for the `[sync]`

* `job_name` is the type of job the current thread (worker or ps) should perform.

* `task_index` is the index of the current server's IP in the list for its job (ps or worker).
    * Defaults to `0`. Worker 0 will become the chief with extra responsibilities like cummulating the results.

* `seed` is the seed for all the randomness in the server.

The asynchronous Federated Architecture script is located at
    baselines/deepq/experiments/async_fed_avg.py
