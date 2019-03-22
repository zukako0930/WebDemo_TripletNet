import base64
import struct

#Base64でエンコードする画像
target_file=r"./original.jpg"
#エンコード保存先
encode_file=r"./encode.txt"

with open(target_file, 'rb') as f:
    data = f.read()

    #Base64で画像をエンコード
encode=base64.b64encode(data)
with open(encode_file,"wb") as f:
    f.write(encode)
    
