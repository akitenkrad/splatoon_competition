# splatoon_competition
codes for prob.space splatoon competition

## How to collect external data

### get main weapon data from https://www.ikaclo.jp/2/weapons/

```
python -m get_main_weapons --output main_weapon_data.csv
```

### get sub weapon data from https://www.ikaclo.jp/2/weapons/sub/

```
python -m get_sub_weapons --output sub_weapon_data.csv
```

### get special weapon data from https://www.ikaclo.jp/2/weapons/special/

```
python -m get_special_weapons --output special_weapon_data.csv
```