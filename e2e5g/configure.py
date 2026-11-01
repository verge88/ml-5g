"""Generate static deployment configuration, never learn policy from NF replies."""
from pathlib import Path
import yaml

ADDR={'amf':'127.0.0.5','smf':'127.0.0.4','upf':'127.0.0.7','udm':'127.0.0.12','udr':'127.0.0.20','pcf':'127.0.0.13','bsf':'127.0.0.15'}
ROLES=['nrf','ausf','nssf','udr','udm','pcf','bsf','upf','smf','amf']

def generate(root=Path('/config')):
    root.mkdir(parents=True,exist_ok=True)
    for role in ROLES:
        if role in ['nrf','ausf','nssf']:
            d=yaml.safe_load(Path(f'/opt/lab/cu5g/{role}.yaml').read_text())
            if role=='nrf':
                d['nrf']['time']={'nf_instance':{'heartbeat':1}}
                d['nrf']['serving']=[{'plmn_id':{'mcc':999,'mnc':70}}]
        else:
            text=Path(f'/src/open5gs/configs/open5gs/{role}.yaml.in').read_text().split('################################################################################')[0]
            text=text.replace('@localstatedir@','/var').replace('@sysconfdir@','/etc')
            d=yaml.safe_load(text); cfg=d[role]
            if 'sbi' in cfg: cfg['sbi']['client']={'nrf':[{'uri':'http://127.0.0.1:7777'}]}
            if role in ['udr','pcf']: d['db_uri']='mongodb://database/open5gs'
            if role=='udm': cfg.pop('hnet',None)  # test subscription uses null SUCI scheme
            if role=='amf': cfg['ngap']['server']=[{'address':'172.28.0.10'}]
            if role in ['upf','smf']:
                cfg['session']=[{'subnet':'10.45.0.0/16','gateway':'10.45.0.1'}]
                if role=='upf': cfg['gtpu']['server']=[{'address':'172.28.0.10'}]
                if role=='smf': cfg.pop('freeDiameter',None); cfg['dns']=['10.45.0.1']
        d['global']={'max':{'ue':32}}
        (root/f'{role}.yaml').write_text(yaml.safe_dump(d,sort_keys=False))

def subscriber(imsi='999700000000001'):
    return dict(imsi=imsi,security=dict(k='465B5CE8B199B49FAA5F0A2EE238A6BC',amf='8000',op=None,opc='E8ED289DEBA952E4283B54E88E6183CA'),
        ambr=dict(downlink=dict(value=1,unit=3),uplink=dict(value=1,unit=3)),
        slice=[dict(sst=1,default_indicator=True,session=[dict(name='internet',type=3,
            qos=dict(index=9,arp=dict(priority_level=8,pre_emption_capability=1,pre_emption_vulnerability=1)),
            ambr=dict(downlink=dict(value=1,unit=3),uplink=dict(value=1,unit=3)))])],
        access_restriction_data=32,subscriber_status=0,network_access_mode=0,subscribed_rau_tau_timer=12,schema_version=1,__v=0)
