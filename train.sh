# training ISM on Market-1501->hazy_DukeMTMC
python tools/train.py  --info "ISM" --config-file ./configs/Hazy_Market1501/server_ism_bagtricks_R50.yml
# training ISM on DukeMTMC->hazy_Market-1501
python tools/train.py  --info "ISM" --config-file ./configs/Hazy_DukeMTMC/server_ism_bagtricks_R50.yml