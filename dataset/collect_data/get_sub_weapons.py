from pathlib import Path
import urllib
import json
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

class SubWeapon(object):
    def __init__(self, name, min_attack, max_attack, ink_consumption, available_num, iine):
        self.name = name
        self.min_attack = min_attack
        self.max_attack = max_attack
        self.ink_consumption = ink_consumption
        self.available_num = available_num
        self.iine = iine
        
    def __str__(self):
        return '<SubWeapon {}>'.format(self.name)
    
    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        res = {
            'name': self.name,
            'min_attack': self.min_attack,
            'max_attack': self.max_attack,
            'ink_consumption': self.ink_consumption,
            'available_num': self.available_num,
            'iine': self.iine
        }
        return res
    
def collect(output):
    
    output = Path(output)
    assert output.suffix == '.csv'
    
    sub_weapons = []
    r = urllib.request.Request('https://www.ikaclo.jp/2/weapons/sub/')
    with urllib.request.urlopen(r) as u:
        soup = BeautifulSoup(u, 'lxml')
        table = soup.find('table').find('tbody')

        for tr in table.find_all('tr'):
            td_all = tr.find_all('td')
            name = td_all[0].find_all('a')[-1].get_text().strip()
            attack = td_all[1].get_text().strip()
            if '～' in attack:
                min_attack, max_attack = [int(float(a.strip())) for a in attack.split('～')]
            elif attack != '-':
                min_attack, max_attack = int(float(attack)), int(float(attack))
            else:
                min_attack, max_attack = np.nan, np.nan

            ink_consumption = td_all[2].get_text().strip()
            available_num = td_all[3].get_text().strip().replace('コ', '')
            if available_num == '-':
                available_num = np.nan
            iine = int(td_all[4].find('span').get_text())

            sub_weapons.append(SubWeapon(name, min_attack, max_attack, ink_consumption, available_num, iine))
            
    weapon_data = pd.DataFrame([w.to_dict() for w in sub_weapons])
    weapon_data.to_csv(str(output.absolute()), index=False, header=True)
    
    print('save weapon data ->', str(output))
    
def build_parser():
    parser = ArgumentParser('collect external weapon data from https://www.ikaclo.jp/2/weapons/sub/')
    parser.add_argument('--output', type=str, default='sub_weapon_data.csv', help='output file')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = build_parser()
    
    collect(args.output)