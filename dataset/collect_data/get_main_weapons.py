from pathlib import Path
import urllib
import json
from argparse import ArgumentParser
import pandas as pd
from bs4 import BeautifulSoup

class MainWeapon(object):
    def __init__(self, name, sub_weapon, special_weapon, category, attack_range, attack_power, attack_rounds, iine):
        self.name = name
        self.sub_weapon = sub_weapon
        self.special = special_weapon
        self.category = category
        self.range = attack_range
        self.power = attack_power
        self.rounds_per = attack_rounds
        self.iine = iine
        
    def __str__(self):
        return '<MainWeapon {}>'.format(self.name)
    
    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        res = {
            'name': self.name,
            'sub_weapon': self.sub_weapon,
            'special': self.special,
            'range': self.range,
            'power': self.power,
            'rounds_per': self.rounds_per,
            'iine': self.iine
        }
        return res

def collect(output):
    
    output = Path(output)
    assert output.suffix == '.csv'
    
    main_weapons = []
    r = urllib.request.Request('https://www.ikaclo.jp/2/weapons/')
    with urllib.request.urlopen(r) as u:
        soup = BeautifulSoup(u, 'lxml')
        table = soup.find('table').find('tbody')

        for tr in table.find_all('tr'):
            td_all = tr.find_all('td')
            name = td_all[0].find('span').get_text().strip()
            sub_special = [span.get_text().strip() for span in td_all[1].find_all('span')]
            category = td_all[2].get_text().strip()
            weapon_range = int(td_all[3].get_text())
            weapon_attack = int(td_all[4].get_text())
            rounds_per = int(td_all[5].get_text())
            iine = int(td_all[6].find_all('span')[0].get_text())

            main_weapons.append(MainWeapon(name, sub_special[0], sub_special[1], category, weapon_range, weapon_attack, rounds_per, iine))
    
    weapon_data = pd.DataFrame([w.to_dict() for w in main_weapons])
    weapon_data.to_csv(str(output.absolute()), index=False, header=True)
    
    print('save weapon data ->', str(output))
    
def build_parser():
    parser = ArgumentParser('collect external weapon data from https://www.ikaclo.jp/2/weapons/')
    parser.add_argument('--output', type=str, default='main_weapon_data.csv', help='output file')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = build_parser()
    
    collect(args.output)