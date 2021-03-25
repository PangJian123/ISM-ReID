# supervised training on Market-1501
python tools/pre_train.py --config-file ./configs/Hazy_Market1501/server_baseline_bagtricks_R50.yml --info "baseline" INPUT.DO_AUTOAUG True INPUT.DO_CJ True OUTPUT_DIR "/home/pj/fast-reid-master/logs/hazy-market1501/bagtricks_R50/baseline/"
# supervised training on DukeMTMC
python tools/pre_train.py --config-file ./configs/Hazy_DukeMTMC/server_baseline_bagtricks_R50.yml --info "baseline"  INPUT.DO_AUTOAUG True INPUT.DO_CJ True OUTPUT_DIR "/home/pj/fast-reid-master/logs/hazy-dukemtmc/bagtricks_R50/baseline/"