# INSTRUCTIONS.md

- context
  - `/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/03_trading_experiment.py` has the cartpole solved by DQN and I want to convert it into DQN solving a daily bitcoin trading problem
  - a sample of my data is exemplified in `/home/dusoudeth/Documentos/github/monnaie-par-renforcement/docs/data_description_01.md`

- instructions: modify `03_trading_experiment.py` in order to:
  - change the cartpole by the bitcoin position trading environment
  - to start, the action space should be [BUY, HOLD, SELL] and the state space should be based in `close`, `volume` and `fng_value`