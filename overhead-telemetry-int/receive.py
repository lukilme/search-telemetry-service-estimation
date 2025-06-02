#!/usr/bin/env python3
import argparse
import sys
import socket
import random
import struct
import os
import csv
import time

from scapy.all import Packet, bind_layers, XByteField, FieldLenField, BitField, ShortField, IntField, PacketListField, Ether, IP, UDP, sendp, get_if_hwaddr, sniff


class InBandNetworkTelemetry(Packet):
    fields_desc = [ BitField("switchID_t", 0, 31),
                    BitField("ingress_port",0, 9),
                    BitField("egress_port",0, 9),
                    BitField("egress_spec", 0, 9),
                    BitField("ingress_global_timestamp", 0, 48),
                    BitField("egress_global_timestamp", 0, 48),
                    BitField("enq_timestamp",0, 32),
                    BitField("enq_qdepth",0, 19),
                    BitField("deq_timedelta", 0, 32),
                    BitField("deq_qdepth", 0, 19)
                  ]
    """any thing after this packet is extracted is padding"""
    def extract_padding(self, p):
                return "", p

class nodeCount(Packet):
  name = "nodeCount"
  fields_desc = [ ShortField("count", 0),
                  PacketListField("INT", [], InBandNetworkTelemetry, count_from=lambda pkt:(pkt.count*1))]
  
  
def get_field_names(packet):
  if packet.haslayer(InBandNetworkTelemetry):
    field_names = ['timestamp']
    fields = [field.name for field in packet[InBandNetworkTelemetry].fields_desc]
    switch_count = len(packet[nodeCount].INT)
    
    for i in range(switch_count):
      field_names.extend(fields)
    return field_names
  return []

def save_to_csv(filename, field_names, packet_data):
  file_exists = os.path.isfile('data_int.csv')
    
  with open('data_int.csv', 'a', newline='') as file:
    writer = csv.writer(file)
        
    if not file_exists:
      writer.writerow(field_names)

    writer.writerow(packet_data)

def handle_pkt(pkt):
  pkt.show2()
  if pkt.haslayer(nodeCount):
    timestamp = int(time.time())
    packet_data = [timestamp]
    node_count = pkt[nodeCount].count
    fields = [field.name for field in pkt[InBandNetworkTelemetry].fields_desc]
    field_names = get_field_names(pkt)

    for i in range(node_count):
      int_pkt = pkt[nodeCount].INT[i]
      packet_data.extend([getattr(int_pkt, field, "") for field in fields])

    save_to_csv('data_int.csv', field_names, packet_data)
  
def main():

  iface = 'enp0s8'
  bind_layers(IP, nodeCount, proto = 253)
  sniff(filter = "ip proto 253", iface = iface, prn = lambda x: handle_pkt(x))

if __name__ == '__main__':
    main()