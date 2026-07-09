import os

file_path = r"c:\Users\Deng2\Desktop\temp\kuafu\stm32_firmware\MDK-ARM\F407ZG.uvprojx"

# Test different encodings
encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'gbk', 'gb2312']
for enc in encodings:
    try:
        with open(file_path, 'r', encoding=enc) as f:
            content = f.read(500)
            print(f"Encoding {enc}: Success!")
            print(content[:200])
            break
    except Exception as e:
        print(f"Encoding {enc}: Failed ({e})")
