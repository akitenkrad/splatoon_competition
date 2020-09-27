# splatoon_competition
codes for prob.space splatoon competition

## How to run pytorch model train

```bash
python -m models.pytorch.train \
       --ds-path dataset/train_data.csv \
       --epochs 100 \
       --batch-size 8 \
       --tf-block-size 12
```

## How to collect external data

### get main weapon data from https://www.ikaclo.jp/2/weapons/

```bash
python -m get_main_weapons --output main_weapon_data.csv
```

### get sub weapon data from https://www.ikaclo.jp/2/weapons/sub/

```bash
python -m get_sub_weapons --output sub_weapon_data.csv
```

### get special weapon data from https://www.ikaclo.jp/2/weapons/special/

```bash
python -m get_special_weapons --output special_weapon_data.csv
```