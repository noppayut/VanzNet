# VanzNet
A character-level RNN for generating motorcycle thug (Vanz: แว๊น) names (in Thai)

## Training data
Training data is made from Facebook names of several Thai motorcycle thug Facebook group members which might violate privacy issues.
Contact me directly if you are interested.

## Nets
- `vanznet.py`, `vanznet_nb.py` : vanilla RNN implementation
- `vanznet_lstm.py`, `vanznet_lstm_nb.py` : LSTM implementation

## Example

Starting letter | Vanilla RNN | LSTM
----- | ----- | -----
ก | การัก เอ็มใจร้าย | กู เจ๋ง
ค | คำเพิ เอ็ม | คนชั่วคราว พระเมืองต
ม | มี่ เฉยย | มอส คุง
จ | จัน รัง สายสุข | จั้ก บุญจอง
ว | วัน พลู | วัน ทูกแป้ว
ร | รัก' เอ็น | รักกันเมื่อยังหาย ใจ
บ | บอร เทียย | บอย ก็กี

## Make your own vanz nickname

Name | Vanz name (LSTM)
---- | ----
บอย | บอย กลับ' ต้า
ฟ้า | ฟ้า เถื่อนน
วิน | วิน ที่สุดฟง
จอน | จอน จัด
เต้ย | เต้ย จัดหั้ย
โอ๊ต | โอ๊ต หนองใหล่สับ
โอ | โอ คนเดิม นะครับ
เอิร์ธ | เอิร์ธ' เก่ง
