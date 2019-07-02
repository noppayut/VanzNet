# VanzNet
A character-level RNN for generating motorcycle thug (Vanz: แว๊น) names (in Thai)

## Training data
Training data is made from Facebook names of members of various Thai motorcycle thug Facebook groups which might violate privacy issues.
Contact me directly if you are interested.

## Nets
- `vanznet.py`, `vanznet_nb.py` : vanilla RNN implementation
- `vanznet_lstm.py`, `vanznet_lstm_nb.py` : LSTM implementation

## Example

Starting letter | Vanilla RNN | LSTM
----- | ----- | -----
ก | การัก เอ็มใจร้าย | กิตติศักดิ์ กันทะวัง
ค | คำเพิ เอ็ม | คน บ้า
ม | มี่ เฉยย | มินิ กองฟาง
จ | จัน รัง สายสุข | จั้ก บุญจอง
ว | วัน พลู | วัน ทูกแป้ว
ร | รัก' เอ็น | รักกันเมื่อยังหาย ใจ
บ | บอร เทียย | บอย ก็กี
