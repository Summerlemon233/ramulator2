#!/usr/bin/env bash

# Direct-key single tuple.
python3 pregen_ramulator_bank.py \
  --task-mode direct \
  --L 2048 \
  --nhead 16 \
  --dhead 80 \
  --dbyte 2 \
  --pim-type BA \
  --power-constraint 1 \
  --workers 96 \
  --flush-every 200 \
  --ramulator-out /home/lizhuoran200/vstack/attacc_simulator/ramulator.out \
  --tmp-dir /home/lizhuoran200/vstack/attacc_simulator/ramulator2/trace_gen/tmp

# Direct-key sweep with list/range inputs.
python3 pregen_ramulator_bank.py \
  --task-mode direct \
  --L 1:31000 \
  --nhead 12,16 \
  --dhead 80 \
  --dbyte 2 \
  --pim-type BA \
  --power-constraint 1 \
  --workers 96 \
  --flush-every 200 \
  --ramulator-out /home/lizhuoran200/vstack/attacc_simulator/ramulator.out \
  --tmp-dir /home/lizhuoran200/vstack/attacc_simulator/ramulator2/trace_gen/tmp

# Legacy derived-mode compatibility example.
# python3 pregen_ramulator_bank.py \
#   --task-mode derived \
#   --model GPT-175B \
#   --ngpu 8 \
#   --num-hbm 8 \
#   --batch-min 16 \
#   --batch-max 16 \
#   --seqlen-min 1 \
#   --seqlen-max 74372 \
#   --maxlen-floor 4096 \
#   --dbyte 2 \
#   --power-modes 1 \
#   --workers 96 \
#   --flush-every 200 \
#   --ramulator-out /home/lizhuoran200/vstack/attacc_simulator/ramulator.out \
#   --tmp-dir /home/lizhuoran200/vstack/attacc_simulator/ramulator2/trace_gen/tmp
